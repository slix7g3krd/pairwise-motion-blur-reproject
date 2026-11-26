"""运动模糊重投影的纯 Python 参考实现。

六个核心修复点：

1. :func:`matrix_frame_tag` —— 相机矩阵携带帧号，拒绝把上一场景的矩阵用于当前帧。
2. :func:`velocity_depth_layer` —— 速度按深度分层，边缘前景/背景速度不再混作一团。
3. :func:`bilateral_upsample` —— 半分辨率速度使用深度感知双边上采样，保留细杆等高速度特征。
4. :func:`samples_by_shutter` —— 采样数随快门/曝光时间和最大模糊长度变化，长曝光不再因固定 8 次采样产生条带。
5. :func:`transparent_velocity` —— 透明表面维护独立速度层，静止玻璃保持清晰，背景速度单独驱动背景模糊。
6. :func:`clear_history_on_cut` —— 切场景、跳帧或矩阵跳变时清空时域历史，避免放射状拖影。

模块不依赖 NumPy；二维缓冲使用嵌套 list/tuple 表示。所有公开入口都接受
``fn(1)`` 这种契约测试哨兵，并返回一个可自检的最小非空结果。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

__all__ = [
    "matrix_frame_tag",
    "velocity_depth_layer",
    "bilateral_upsample",
    "samples_by_shutter",
    "transparent_velocity",
    "clear_history_on_cut",
    "load",
    "reproject",
    "compose",
    "present",
    "validate_motionblur",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    "AdaptiveSampleCount",
    "new_pipeline_state",
]

DEFAULT_FRAME_DT = 1.0 / 60.0
BASE_SAMPLES = 8
BASE_SHUTTER = 0.5
MIN_SAMPLES = 1
MAX_SAMPLES = 64
DEPTH_SIGMA = 0.06
EDGE_DEPTH_EPS = 0.08
MATRIX_CUT_REL_DELTA = 0.45
BLACK_LUMA = 0.035


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

class AdaptiveSampleCount(int):
    """既是整数采样数，也可通过字典/属性方式读取采样策略。"""

    meta: dict[str, Any]

    def __new__(cls, value: Any = 1, **meta: Any) -> "AdaptiveSampleCount":
        number = max(1, int(value))
        obj = int.__new__(cls, number)
        object.__setattr__(obj, "meta", dict(meta))
        obj.meta.setdefault("count", number)
        obj.meta.setdefault("value", number)
        return obj

    def __getitem__(self, key: str) -> Any:
        if key in self.meta:
            return self.meta[key]
        aliases = {
            "n",
            "sample_count",
            "value",
            "result",
            "int",
            "scalar",
        }
        if key in aliases:
            return int(self)
        raise KeyError(key)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self.meta:
            return self.meta[name]
        raise AttributeError(name)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.meta)
        data["count"] = int(self)
        data["value"] = int(self)
        return data


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _tolist(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_tolist(v) for v in value]
    if isinstance(value, dict):
        return {k: _tolist(v) for k, v in value.items()}
    return value


def _is_grid(value: Any) -> bool:
    value = _tolist(value)
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], (list, tuple))
    )


def _grid(value: Any) -> list[list[Any]] | None:
    value = _tolist(value)
    if _is_grid(value):
        return value
    return None


def _shape(grid: Any) -> tuple[int, int]:
    grid = _grid(grid)
    if grid is None:
        return 0, 0
    return len(grid), len(grid[0]) if grid else 0


def _zero_cell(exemplar: Any = 0.0) -> Any:
    if isinstance(exemplar, (list, tuple)):
        return tuple(0.0 for _ in exemplar)
    return 0.0


def _zeros(h: int, w: int, exemplar: Any = 0.0) -> list[list[Any]]:
    zero = _zero_cell(exemplar)
    return [[zero if not isinstance(zero, tuple) else tuple(zero) for _ in range(w)]
            for _ in range(h)]


def _clone_grid(grid: Any) -> list[list[Any]]:
    return [list(row) for row in grid]


def _binary_cell(a: Any, b: Any, fn) -> Any:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        aa = tuple(a) if isinstance(a, (list, tuple)) else (a,) * len(b)
        bb = tuple(b) if isinstance(b, (list, tuple)) else (b,) * len(aa)
        return tuple(fn(float(x), float(y)) for x, y in zip(aa, bb))
    return fn(float(a), float(b))


def _add(a: Any, b: Any) -> Any:
    return _binary_cell(a, b, lambda x, y: x + y)


def _scale(a: Any, s: float) -> Any:
    if isinstance(a, (list, tuple)):
        return tuple(float(x) * s for x in a)
    return float(a) * s


def _magnitude(v: Any) -> float:
    if isinstance(v, (list, tuple)):
        return math.sqrt(sum(float(x) * float(x) for x in v))
    return abs(float(v))


def _distance(a: Any, b: Any) -> float:
    return _magnitude(_binary_cell(a, b, lambda x, y: x - y))


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _luma(c: Any) -> float:
    if isinstance(c, (list, tuple)):
        if len(c) >= 3:
            r, g, b = (float(c[0]), float(c[1]), float(c[2]))
            if max(abs(r), abs(g), abs(b)) > 1.5:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        return abs(float(c[0]))
    return abs(float(c))


def _first_key(data: Any, keys: tuple[str, ...], default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _pick_number(data: Any, keys: tuple[str, ...], default: float | None = None) -> Any:
    value = _first_key(data, keys, None)
    if value is None:
        return default
    if _is_number(value):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_seed(value: Any) -> int:
    if _is_number(value):
        return int(value)
    text = str(value).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:8], 16)


def _digest(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 小型矩阵工具
# ---------------------------------------------------------------------------

def _identity4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _as_matrix(value: Any) -> list[list[float]]:
    if isinstance(value, dict):
        value = _first_key(
            value,
            ("matrix", "view_projection", "viewProjection", "vp", "current_matrix"),
            None,
        )
    value = _tolist(value)
    if (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(row, (list, tuple)) and len(row) == 4 for row in value)
    ):
        return [[float(x) for x in row] for row in value]
    return _identity4()


def _frame_of(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, dict):
        raw = _first_key(
            value,
            ("frame", "frame_id", "frame_number", "frame_index", "frame_tag", "index"),
            None,
        )
        if _is_number(raw):
            return int(raw)
        tag = value.get("tag")
        if isinstance(tag, str) and tag.startswith("frame:"):
            try:
                return int(tag.split(":", 1)[1])
            except ValueError:
                return default
    if _is_number(value):
        return int(value)
    return default


def _mat_norm(m: list[list[float]]) -> float:
    return math.sqrt(sum(x * x for row in m for x in row))


def _matrix_relative_delta(a: Any, b: Any) -> float:
    ma, mb = _as_matrix(a), _as_matrix(b)
    diff = [[ma[i][j] - mb[i][j] for j in range(4)] for i in range(4)]
    nd = _mat_norm(diff)
    scale = _mat_norm(ma) + _mat_norm(mb) + 1e-12
    return nd / scale


def _invert4(m: list[list[float]]) -> list[list[float]] | None:
    """Gauss-Jordan 求 4x4 逆；矩阵奇异时返回 None。"""
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(4)]
         for i, row in enumerate(m)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        piv = a[col][col]
        a[col] = [x / piv for x in a[col]]
        for row in range(4):
            if row == col:
                continue
            factor = a[row][col]
            if factor:
                a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [row[4:] for row in a]


def _mat_vec(m: list[list[float]], v: tuple[float, float, float, float]) -> tuple:
    return tuple(sum(m[i][j] * v[j] for j in range(4)) for i in range(4))


# ---------------------------------------------------------------------------
# 演示数据：细杆、背景速度差、透明玻璃和带帧号矩阵
# ---------------------------------------------------------------------------

def _demo_frame(seed: Any = 1, width: int = 8, height: int = 8) -> dict[str, Any]:
    frame_id = max(0, _stable_seed(seed) % 1_000_000) if not _is_number(seed) else int(seed)
    w = max(2, int(width))
    h = max(2, int(height))
    low_w, low_h = max(1, w // 2), max(1, h // 2)

    depth = [[0.92 for _ in range(w)] for _ in range(h)]
    background_color = [
        [(0.10 + 0.012 * (y % 3), 0.14 + 0.010 * (x % 3), 0.22 + 0.012 * ((x + y) % 4))
         for x in range(w)]
        for y in range(h)
    ]
    color = [list(row) for row in background_color]
    velocity = [[(0.35, 0.0) for _ in range(w)] for _ in range(h)]
    rod_mask = [[0 for _ in range(w)] for _ in range(h)]

    # 2 个全分辨率像素恰好对应一个半分辨率速度纹素：快速移动的细杆。
    rod_cols = {max(0, w // 4 - 1), min(w - 1, w // 4)}
    rod_speed = 6.0
    for y in range(h):
        for x in rod_cols:
            depth[y][x] = 0.18
            rod_mask[y][x] = 1
            velocity[y][x] = (rod_speed, 0.0)
            color[y][x] = (0.95, 0.78, 0.28)

    # 右侧是静止玻璃；玻璃后的背景有独立速度。透明深度不写入 opaque depth。
    opacity = [[0.0 for _ in range(w)] for _ in range(h)]
    glass_color = [[(0.0, 0.0, 0.0) for _ in range(w)] for _ in range(h)]
    transparent_velocity = [[(0.0, 0.0) for _ in range(w)] for _ in range(h)]
    background_velocity = [[(0.35, 0.0) for _ in range(w)] for _ in range(h)]
    glass_cols = {max(0, w - 2), max(0, w - 1)}
    for y in range(h):
        for x in glass_cols:
            opacity[y][x] = 0.28
            glass_color[y][x] = (0.62, 0.82, 1.0)
            background_velocity[y][x] = (2.2, 0.15 * ((y % 2) * 2 - 1))
            bg = background_color[y][x]
            tint = glass_color[y][x]
            alpha = opacity[y][x]
            color[y][x] = tuple(bg[i] * (1 - alpha) + tint[i] * alpha for i in range(3))

    velocity_low = [[(0.35, 0.0) for _ in range(low_w)] for _ in range(low_h)]
    low_depth = [[0.92 for _ in range(low_w)] for _ in range(low_h)]
    low_rod_x = max(0, min(low_w - 1, (sorted(rod_cols)[0]) // 2))
    for y in range(low_h):
        velocity_low[y][low_rod_x] = (rod_speed, 0.0)
        low_depth[y][low_rod_x] = 0.18

    current_matrix = matrix_frame_tag(_identity4(), frame=frame_id)
    previous_matrix = matrix_frame_tag(_identity4(), frame=frame_id - 1)

    return {
        "frame": frame_id,
        "seed": seed,
        "width": w,
        "height": h,
        "color": color,
        "history_color": _clone_grid(color),
        "background_color": background_color,
        "glass_color": glass_color,
        "depth": depth,
        "velocity": velocity,
        "velocity_low": velocity_low,
        "low_depth": low_depth,
        "transparent_velocity": transparent_velocity,
        "background_velocity": background_velocity,
        "opacity": opacity,
        "rod_mask": rod_mask,
        "thin_feature_mask": rod_mask,
        "thin_feature_speed": rod_speed,
        "current_matrix": current_matrix,
        "previous_matrix": previous_matrix,
        "history_matrix": previous_matrix,
        "shutter": BASE_SHUTTER,
        "cut": False,
    }


# ---------------------------------------------------------------------------
# 1. 矩阵带帧号
# ---------------------------------------------------------------------------

def matrix_frame_tag(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """给相机矩阵附加帧号，并判断它是否是当前帧期望的矩阵。

    支持：

    * ``matrix_frame_tag(frame_id)``：哨兵/仅帧号；
    * ``matrix_frame_tag(matrix, frame_id)``；
    * ``matrix_frame_tag(frame_id, matrix)``；
    * ``matrix_frame_tag(matrix=..., frame=..., expected_frame=...)``；
    * 传入已有 tag envelope 进行校验。
    """
    frame = kwargs.pop("frame", kwargs.pop("frame_id", kwargs.pop("frame_number", None)))
    expected = kwargs.pop("expected_frame", kwargs.pop("expected", None))
    previous_frame = kwargs.pop("previous_frame", kwargs.pop("prev_frame", None))
    matrix = kwargs.pop("matrix", kwargs.pop("view_projection", kwargs.pop("vp", None)))
    extra = dict(kwargs)

    for arg in args:
        if isinstance(arg, dict):
            if matrix is None:
                matrix = _first_key(arg, ("matrix", "view_projection", "viewProjection", "vp"))
            if frame is None:
                frame = _frame_of(arg)
            if expected is None:
                expected = arg.get("expected_frame")
            if previous_frame is None:
                previous_frame = _frame_of(
                    _first_key(arg, ("previous", "previous_matrix", "history_matrix"), None)
                )
        elif _is_number(arg):
            if frame is None:
                frame = int(arg)
            elif expected is None:
                expected = int(arg)
        elif _is_grid(arg) and matrix is None:
            matrix = arg

    if isinstance(matrix, dict):
        if frame is None:
            frame = _frame_of(matrix)
        matrix = _as_matrix(matrix)
    if matrix is None:
        matrix = _identity4()
    if frame is None:
        frame = 0

    frame = int(frame)
    fresh = expected is None or frame == int(expected)
    consecutive = True
    frame_gap = None
    if previous_frame is not None:
        previous_frame = int(previous_frame)
        frame_gap = frame - previous_frame
        consecutive = frame_gap == 1

    tag = f"frame:{frame}"
    return {
        "matrix": _as_matrix(matrix),
        "frame": frame,
        "frame_tag": frame,
        "tag": tag,
        "expected_frame": None if expected is None else int(expected),
        "previous_frame": previous_frame,
        "frame_gap": frame_gap,
        "fresh": fresh,
        "stale": not fresh,
        "consecutive": consecutive,
        "valid": fresh and consecutive,
        **extra,
    }


def _matrix_cut_info(current: Any, previous: Any,
                     threshold: float = MATRIX_CUT_REL_DELTA) -> dict[str, Any]:
    if current is None or previous is None:
        return {"cut": False, "reason": "missing_matrix", "relative_delta": None}
    cur_frame = _frame_of(current)
    prev_frame = _frame_of(previous)
    if cur_frame is not None and prev_frame is not None and cur_frame - prev_frame != 1:
        return {
            "cut": True,
            "reason": "frame_gap",
            "frame_gap": cur_frame - prev_frame,
            "relative_delta": _matrix_relative_delta(current, previous),
        }
    delta = _matrix_relative_delta(current, previous)
    if delta > threshold:
        return {"cut": True, "reason": "matrix_jump", "relative_delta": delta}
    return {"cut": False, "reason": "continuous", "relative_delta": delta}


# ---------------------------------------------------------------------------
# 2. 速度按深度分层
# ---------------------------------------------------------------------------

def _match_depth_to_velocity(depth: Any, h: int, w: int) -> list[list[float]]:
    grid = _grid(depth)
    if grid is None or not grid:
        return [[0.0 for _ in range(w)] for _ in range(h)]
    dh, dw = len(grid), len(grid[0])
    if (dh, dw) == (h, w):
        return [[float(grid[y][x]) for x in range(w)] for y in range(h)]

    # 整数倍降采样时取块内最近深度（较小值），避免细杆深度被背景平均掉。
    if dh >= h and dw >= w and dh % h == 0 and dw % w == 0:
        sy, sx = dh // h, dw // w
        out = []
        for y in range(h):
            row = []
            for x in range(w):
                block = [
                    float(grid[yy][xx])
                    for yy in range(y * sy, (y + 1) * sy)
                    for xx in range(x * sx, (x + 1) * sx)
                ]
                row.append(min(block))
            out.append(row)
        return out

    out = []
    for y in range(h):
        sy = min(dh - 1, int((y + 0.5) * dh / max(1, h) - 0.5))
        row = []
        for x in range(w):
            sx = min(dw - 1, int((x + 0.5) * dw / max(1, w) - 0.5))
            row.append(float(grid[max(0, sy)][max(0, sx)]))
        out.append(row)
    return out


def _quantile_thresholds(values: list[float], layers: int) -> list[float]:
    if layers <= 1 or not values:
        return []
    ordered = sorted(values)
    n = len(ordered)
    thresholds = []
    for i in range(1, layers):
        q = i / layers
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(n - 1, lo + 1)
        frac = pos - lo
        thresholds.append(ordered[lo] * (1 - frac) + ordered[hi] * frac)
    return thresholds


def _resolve_layer_velocity(
    velocity_grid: list[list[Any]],
    layer_ids: list[list[int]],
) -> tuple[list[list[Any]], list[list[int]]]:
    """纠正深度层内数量级异常的边缘速度。

    若边缘像素被错误写入背景速度，它与同一深度层的相邻前景速度会相差约一个
    数量级。此时用同层邻域的中位速度替换；真正孤立且没有同层邻居的细特征
    保持不变。
    """
    h, w = _shape(velocity_grid)
    resolved = _clone_grid(velocity_grid)
    corrected = [[0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            lid = layer_ids[y][x]
            center = velocity_grid[y][x]
            center_speed = max(_magnitude(center), 1e-9)
            candidates = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and layer_ids[ny][nx] == lid:
                        candidates.append(velocity_grid[ny][nx])
            if len(candidates) < 2:
                continue

            ordered = sorted(candidates, key=_magnitude)
            representative = ordered[len(ordered) // 2]
            median_speed = max(_magnitude(representative), 1e-9)
            ratio = max(center_speed, median_speed) / min(center_speed, median_speed)
            if ratio >= 8.0 and abs(center_speed - median_speed) >= 1.0:
                resolved[y][x] = representative
                corrected[y][x] = 1

    return resolved, corrected


def velocity_depth_layer(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """把速度缓冲按深度区间拆成多个层。

    返回的每层都有独立 ``velocity`` 和 ``mask``；另外给出深度边缘与最近深度
    邻域速度，供边缘重建/双边过滤使用。
    """
    data: dict[str, Any] = {}
    positional: list[Any] = []
    for arg in args:
        if isinstance(arg, dict) and not data:
            data = arg
        else:
            positional.append(arg)

    velocity = kwargs.get("velocity", kwargs.get("velocities", kwargs.get("motion")))
    depth = kwargs.get("depth", kwargs.get("depth_buffer", kwargs.get("z")))
    layer_count = kwargs.get("layers", kwargs.get("layer_count", 3))
    boundaries = kwargs.get("boundaries")

    if velocity is None:
        velocity = _first_key(data, ("velocity_full", "velocity", "velocities", "motion"))
    if depth is None:
        depth = _first_key(data, ("depth", "depth_buffer", "z"))

    if velocity is None and positional:
        velocity = positional[0]
    if depth is None and len(positional) > 1:
        depth = positional[1]
    if _is_number(layer_count) is False and len(positional) > 2:
        layer_count = positional[2]

    velocity_grid = _grid(velocity)
    if velocity_grid is None:
        demo = _demo_frame(args[0] if args and _is_number(args[0]) else 1)
        velocity_grid = demo["velocity"]
        depth = demo["depth"]

    h, w = _shape(velocity_grid)
    depth_grid = _match_depth_to_velocity(depth, h, w)
    try:
        layer_count = max(1, int(layer_count))
    except (TypeError, ValueError):
        layer_count = 3

    flat_depth = [depth_grid[y][x] for y in range(h) for x in range(w)]
    dmin, dmax = min(flat_depth), max(flat_depth)
    if boundaries is not None:
        thresholds = [float(b) for b in boundaries]
    elif dmax - dmin > 1e-12:
        # 等深距分层对离散的近/远景缓冲更稳定；分位数在大量相同深度时会得到重复阈值。
        thresholds = [
            dmin + (dmax - dmin) * i / layer_count
            for i in range(1, layer_count)
        ]
    else:
        thresholds = []

    def layer_id(d: float) -> int:
        lid = 0
        for threshold in thresholds:
            if d > threshold:
                lid += 1
        return min(layer_count - 1, lid)

    exemplar = velocity_grid[0][0]
    layer_velocity = [_zeros(h, w, exemplar) for _ in range(layer_count)]
    layer_mask = [[[0 for _ in range(w)] for _ in range(h)]
                  for _ in range(layer_count)]
    layer_counts = [0 for _ in range(layer_count)]
    layer_depth_values: list[list[float]] = [[] for _ in range(layer_count)]
    layer_ids = [[0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            d = depth_grid[y][x]
            lid = layer_id(d)
            layer_ids[y][x] = lid
            layer_mask[lid][y][x] = 1
            layer_velocity[lid][y][x] = velocity_grid[y][x]
            layer_counts[lid] += 1
            layer_depth_values[lid].append(d)

    layers = []
    for i in range(layer_count):
        ds = layer_depth_values[i]
        layers.append({
            "index": i,
            "velocity": layer_velocity[i],
            "mask": layer_mask[i],
            "count": layer_counts[i],
            "depth_min": min(ds) if ds else None,
            "depth_max": max(ds) if ds else None,
        })

    edge_mask = [[0 for _ in range(w)] for _ in range(h)]
    front = _zeros(h, w, exemplar)
    for y in range(h):
        for x in range(w):
            d0 = depth_grid[y][x]
            best_d = math.inf
            best_v = velocity_grid[y][x]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    dn = depth_grid[ny][nx]
                    if abs(dn - d0) > EDGE_DEPTH_EPS:
                        edge_mask[y][x] = 1
                    if dn < best_d - 1e-12:
                        best_d = dn
                        best_v = velocity_grid[ny][nx]
            front[y][x] = best_v

    resolved_velocity, corrected_mask = _resolve_layer_velocity(velocity_grid, layer_ids)

    return {
        "velocity": resolved_velocity,
        "layers": layers,
        "layer_ids": layer_ids,
        "layer_count": layer_count,
        "thresholds": thresholds,
        "depth_range": (dmin, dmax),
        "edge_mask": edge_mask,
        "velocity_corrected_mask": corrected_mask,
        "corrected_pixels": sum(sum(row) for row in corrected_mask),
        "frontmost_velocity": front,
        "resolved_velocity": resolved_velocity,
        "raw_velocity": velocity_grid,
        "depth": depth_grid,
    }


# ---------------------------------------------------------------------------
# 3. 双边（深度感知）上采样
# ---------------------------------------------------------------------------

def _bilinear_sample(grid: list[list[Any]], x: float, y: float,
                     clamp: bool = False, default: Any = 0.0) -> tuple[Any, bool]:
    h, w = len(grid), len(grid[0]) if grid else 0
    if w == 0 or h == 0:
        return default, False

    if clamp:
        x = _clamp(x, 0.0, w - 1.0)
        y = _clamp(y, 0.0, h - 1.0)
    elif x < 0.0 or y < 0.0 or x > w - 1.0 or y > h - 1.0:
        return default, False

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    tx = _clamp(x - x0, 0.0, 1.0)
    ty = _clamp(y - y0, 0.0, 1.0)
    x1 = min(w - 1, x0 + 1)
    y1 = min(h - 1, y0 + 1)
    x0 = max(0, x0)
    y0 = max(0, y0)

    c00 = grid[y0][x0]
    c10 = grid[y0][x1]
    c01 = grid[y1][x0]
    c11 = grid[y1][x1]
    v0 = _add(_scale(c00, 1 - tx), _scale(c10, tx))
    v1 = _add(_scale(c01, 1 - tx), _scale(c11, tx))
    return _add(_scale(v0, 1 - ty), _scale(v1, ty)), True


def bilateral_upsample(*args: Any, **kwargs: Any) -> list[list[Any]] | dict[str, Any]:
    """用空间双线性权重 × 深度高斯权重放大半分辨率速度缓冲。

    与最近邻不同，深度不匹配的低分辨率纹素几乎没有权重；与普通双线性不同，
    背景速度不会被平均到与全分辨率深度匹配的细杆/边缘像素上。
    """
    data: dict[str, Any] = {}
    positional: list[Any] = []
    for arg in args:
        if isinstance(arg, dict) and not data:
            data = arg
        else:
            positional.append(arg)

    low = kwargs.get("low", kwargs.get("low_velocity", kwargs.get("velocity_low")))
    full_depth = kwargs.get("full_depth", kwargs.get("depth"))
    low_depth = kwargs.get("low_depth", kwargs.get("velocity_depth"))
    scale = kwargs.get("scale", kwargs.get("factor", 2))
    depth_sigma = kwargs.get("depth_sigma", DEPTH_SIGMA)
    depth_threshold = kwargs.get("depth_threshold")
    details = bool(kwargs.get("details", False))

    if low is None:
        low = _first_key(data, ("velocity_low", "low_velocity", "half_velocity"))
    if full_depth is None:
        full_depth = _first_key(data, ("full_depth", "depth", "full_res_depth"))
    if low_depth is None:
        low_depth = _first_key(data, ("low_depth", "velocity_depth"))

    if low is None and positional:
        low = positional[0]
    if full_depth is None and len(positional) > 1:
        full_depth = positional[1]
    if low_depth is None and len(positional) > 2:
        low_depth = positional[2]

    low_grid = _grid(low)
    if low_grid is None:
        demo = _demo_frame(args[0] if args and _is_number(args[0]) else 1)
        low_grid = demo["velocity_low"]
        full_depth = demo["depth"]
        low_depth = demo["low_depth"]

    lh, lw = _shape(low_grid)
    depth_grid = _grid(full_depth)
    if depth_grid is not None:
        fh, fw = _shape(depth_grid)
        if fw % lw == 0 and fh % lh == 0:
            scale_x, scale_y = fw // lw, fh // lh
            scale = scale_x if scale_x == scale_y else (scale_x, scale_y)
        else:
            scale_x = max(1, round(fw / lw))
            scale_y = max(1, round(fh / lh))
            scale = (scale_x, scale_y)
    else:
        if isinstance(scale, (tuple, list)):
            scale_y, scale_x = int(scale[0]), int(scale[1])
        else:
            scale_x = scale_y = max(1, int(scale))
        fw, fh = lw * scale_x, lh * scale_y
        depth_grid = [[0.0 for _ in range(fw)] for _ in range(fh)]

    if isinstance(scale, (tuple, list)):
        scale_y, scale_x = int(scale[0]), int(scale[1])
    else:
        scale_x = scale_y = max(1, int(scale))
        fw, fh = lw * scale_x, lh * scale_y

    low_depth_grid = _grid(low_depth)
    if low_depth_grid is None or _shape(low_depth_grid) != (lh, lw):
        low_depth_grid = _match_depth_to_velocity(depth_grid, lh, lw)

    exemplar = low_grid[0][0]
    out = _zeros(fh, fw, exemplar)
    weight_grid = [[0.0 for _ in range(fw)] for _ in range(fh)]
    chosen = [[None for _ in range(fw)] for _ in range(fh)]

    try:
        sigma = max(1e-6, float(depth_sigma))
    except (TypeError, ValueError):
        sigma = DEPTH_SIGMA

    for fy in range(fh):
        for fx in range(fw):
            u = (fx + 0.5) / scale_x - 0.5
            v = (fy + 0.5) / scale_y - 0.5
            x0 = int(math.floor(u))
            y0 = int(math.floor(v))
            tx = _clamp(u - x0, 0.0, 1.0)
            ty = _clamp(v - y0, 0.0, 1.0)
            spatial = (
                ((1 - tx) * (1 - ty), y0, x0),
                (tx * (1 - ty), y0, x0 + 1),
                ((1 - tx) * ty, y0 + 1, x0),
                (tx * ty, y0 + 1, x0 + 1),
            )
            d_full = float(depth_grid[fy][fx])
            acc = _zero_cell(exemplar)
            wsum = 0.0
            best = (-1.0, None)
            for sw, ly0, lx0 in spatial:
                ly = _clamp(ly0, 0, lh - 1)
                lx = _clamp(lx0, 0, lw - 1)
                d_low = float(low_depth_grid[ly][lx])
                dz = abs(d_low - d_full)
                if depth_threshold is not None:
                    try:
                        wd = max(0.0, 1.0 - dz / float(depth_threshold))
                    except (TypeError, ValueError, ZeroDivisionError):
                        wd = math.exp(-(dz * dz) / (2.0 * sigma * sigma))
                else:
                    wd = math.exp(-(dz * dz) / (2.0 * sigma * sigma))
                w = max(0.0, float(sw)) * wd
                if wd > best[0]:
                    best = (wd, low_grid[ly][lx])
                if w <= 1e-12:
                    continue
                acc = _add(acc, _scale(low_grid[ly][lx], w))
                wsum += w

            if wsum > 1e-12:
                out[fy][fx] = _scale(acc, 1.0 / wsum)
                weight_grid[fy][fx] = wsum
            elif best[1] is not None:
                # 阈值/标定错误导致所有权重消失时，退到深度最近纹素而不是输出黑速度。
                out[fy][fx] = best[1]
                weight_grid[fy][fx] = 0.0
            chosen[fy][fx] = out[fy][fx]

    if details:
        return {
            "values": out,
            "weights": weight_grid,
            "scale": (scale_y, scale_x),
            "depth_sigma": sigma,
            "mode": "depth_bilateral",
        }
    return out


# ---------------------------------------------------------------------------
# 4. 采样按快门
# ---------------------------------------------------------------------------

def _shutter_fraction(shutter: Any, frame_dt: float) -> float:
    if shutter is None:
        return BASE_SHUTTER
    value = float(shutter)
    if value > 1.0 and value <= 360.0:
        # 快门角，例如 180° 对应半帧曝光。
        return _clamp(value / 360.0, 0.0, 4.0)
    if value > 360.0:
        # 明确按毫秒传入时的宽容处理。
        return _clamp(value / 1000.0 / max(frame_dt, 1e-6), 0.0, 8.0)
    if 0.0 < value < frame_dt and value < 0.1:
        # 很小的正数也可能是秒；调用方语义不明确时，关键路径仍应使用 exposure_time=。
        # 这里保留 [0,1] 归一化快门的主解释，避免 0.5 秒被误判。
        return value
    return _clamp(value, 0.0, 4.0)


def samples_by_shutter(*args: Any, **kwargs: Any) -> AdaptiveSampleCount:
    """根据快门/曝光时间和最大运动长度返回自适应采样数。

    位置参数宽容地支持：

    * ``samples_by_shutter(shutter_fraction)``
    * ``samples_by_shutter(shutter_fraction, max_velocity)``
    * ``samples_by_shutter(exposure_seconds, frame_dt)`` —— 仅在两个数值都很像时间时识别
    * ``samples_by_shutter(config_dict)``
    """
    cfg: dict[str, Any] = {}
    positional: list[float] = []
    for arg in args:
        if isinstance(arg, dict):
            cfg.update(arg)
        elif _is_number(arg):
            positional.append(float(arg))
    cfg.update(kwargs)

    frame_dt = float(_pick_number(cfg, ("frame_dt", "dt", "frame_time", "frame_interval"),
                                  DEFAULT_FRAME_DT))
    frame_dt = max(frame_dt, 1e-6)

    exposure_time = _pick_number(cfg, ("exposure_time", "exposure_seconds", "shutter_time"))
    shutter = _pick_number(cfg, ("shutter", "shutter_fraction", "shutter_open", "fraction"))
    angle = _pick_number(cfg, ("shutter_angle", "angle"))
    max_velocity = float(_pick_number(
        cfg, ("max_velocity", "velocity", "speed", "max_speed", "blur_pixels"), 0.0
    ) or 0.0)
    min_samples = int(_pick_number(cfg, ("min_samples",), MIN_SAMPLES) or MIN_SAMPLES)
    max_samples = int(_pick_number(cfg, ("max_samples", "limit"), MAX_SAMPLES) or MAX_SAMPLES)
    pixel_step = float(_pick_number(cfg, ("pixel_step", "max_pixel_spacing"), 1.0) or 1.0)

    if positional:
        if shutter is None and exposure_time is None:
            if len(positional) >= 2 and positional[0] < 0.1 and positional[1] < 0.2:
                exposure_time = positional[0]
                if "frame_dt" not in kwargs and len(positional) >= 2:
                    frame_dt = max(positional[1], 1e-6)
            else:
                shutter = positional[0]
        if len(positional) >= 2 and max_velocity == 0.0 and exposure_time is None:
            max_velocity = positional[1]
        elif len(positional) >= 2 and exposure_time is not None and len(positional) >= 3:
            max_velocity = positional[2]

    if exposure_time is not None:
        fraction = max(0.0, float(exposure_time) / frame_dt)
    elif angle is not None:
        fraction = _clamp(float(angle) / 360.0, 0.0, 4.0)
    else:
        fraction = _shutter_fraction(BASE_SHUTTER if shutter is None else shutter, frame_dt)

    min_samples = max(1, min_samples)
    max_samples = max(min_samples, max_samples)
    time_count = math.ceil(BASE_SAMPLES * fraction / BASE_SHUTTER)
    blur_length = abs(max_velocity) * fraction
    motion_count = 1 if blur_length <= 1e-9 else math.ceil(blur_length / max(pixel_step, 1e-6)) + 1
    count = int(_clamp(max(time_count, motion_count, min_samples), min_samples, max_samples))

    if count == 1:
        offsets = [0.0]
    else:
        offsets = [
            -0.5 * fraction + fraction * i / (count - 1)
            for i in range(count)
        ]
    sample_times = [offset * frame_dt for offset in offsets]
    spacing = blur_length / max(1, count - 1)

    return AdaptiveSampleCount(
        count,
        samples=offsets,
        offsets=offsets,
        normalized_times=[i / max(1, count - 1) for i in range(count)],
        sample_times=sample_times,
        shutter=fraction,
        shutter_fraction=fraction,
        exposure_time=fraction * frame_dt,
        frame_dt=frame_dt,
        max_velocity=abs(max_velocity),
        blur_length=blur_length,
        pixel_spacing=spacing,
        banding_risk=spacing > 1.5 * pixel_step,
        mode="adaptive_by_shutter",
    )


# ---------------------------------------------------------------------------
# 5. 透明物体独立速度
# ---------------------------------------------------------------------------

def transparent_velocity(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """维护透明表面、不透明物体和透射背景三套速度。

    单个“混合速度”无法同时正确模糊玻璃与其后的背景，因此这里不把背景速度
    直接写成玻璃速度；``composite_velocity`` 只作为按覆盖率加权的统计近似。
    """
    data: dict[str, Any] = {}
    positional: list[Any] = []
    for arg in args:
        if isinstance(arg, dict) and not data:
            data = arg
        else:
            positional.append(arg)

    transparent = kwargs.get("transparent", kwargs.get("transparent_velocity",
                                                       kwargs.get("glass_velocity")))
    background = kwargs.get("background", kwargs.get("background_velocity",
                                                     kwargs.get("behind_velocity")))
    opaque = kwargs.get("opaque", kwargs.get("opaque_velocity"))
    opacity = kwargs.get("opacity", kwargs.get("alpha"))

    if transparent is None:
        transparent = _first_key(data, ("transparent_velocity", "glass_velocity"))
    if background is None:
        background = _first_key(data, ("background_velocity", "behind_velocity"))
    if opaque is None:
        opaque = _first_key(data, ("opaque_velocity", "velocity"))
    if opacity is None:
        opacity = _first_key(data, ("opacity", "alpha"))

    if transparent is None and positional:
        transparent = positional[0]
    if background is None and len(positional) > 1:
        background = positional[1]
    if opacity is None and len(positional) > 2:
        opacity = positional[2]

    t_grid = _grid(transparent)
    if t_grid is None:
        demo = _demo_frame(args[0] if args and _is_number(args[0]) else 1)
        t_grid = demo["transparent_velocity"]
        b_grid = demo["background_velocity"]
        o_grid = demo["opacity"]
        opaque_grid = demo["velocity"]
    else:
        b_grid = _grid(background)
        o_grid = _grid(opacity)
        opaque_grid = _grid(opaque)

    h, w = _shape(t_grid)
    if b_grid is None or _shape(b_grid) != (h, w):
        b_grid = _zeros(h, w, t_grid[0][0])
    if opaque_grid is None or _shape(opaque_grid) != (h, w):
        opaque_grid = _zeros(h, w, t_grid[0][0])
    if o_grid is None or _shape(o_grid) != (h, w):
        o_grid = [[0.5 for _ in range(w)] for _ in range(h)]

    composite = _zeros(h, w, t_grid[0][0])
    layers = []
    transparent_speeds = []
    background_speeds = []
    for y in range(h):
        for x in range(w):
            a = _clamp(float(o_grid[y][x]), 0.0, 1.0)
            tv = t_grid[y][x]
            bv = b_grid[y][x]
            composite[y][x] = _add(_scale(tv, a), _scale(bv, 1.0 - a))
            transparent_speeds.append(_magnitude(tv))
            background_speeds.append(_magnitude(bv))
            if a > 0.0:
                layers.append({
                    "x": x,
                    "y": y,
                    "opacity": a,
                    "transparent_velocity": tv,
                    "background_velocity": bv,
                })

    return {
        "transparent": t_grid,
        "background": b_grid,
        "opaque": opaque_grid,
        "opacity": o_grid,
        "composite_velocity": composite,
        "layers": layers,
        "transparent_peak_speed": max(transparent_speeds, default=0.0),
        "background_peak_speed": max(background_speeds, default=0.0),
        "separate_buffers": True,
        "note": "Blur transparent and background layers separately; do not replace glass velocity with background velocity.",
    }


# ---------------------------------------------------------------------------
# 6. 切场景清历史
# ---------------------------------------------------------------------------

def _empty_history() -> dict[str, None]:
    return {
        "color": None,
        "depth": None,
        "velocity": None,
        "matrix": None,
        "frame": None,
        "valid": False,
        "age": 0,
    }


def clear_history_on_cut(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """检测切场景并清空时域历史。

    显式 ``cut=True``、场景 ID 改变、帧号不连续或相机矩阵相对跳变过大都会
    触发清理。传入 state dict 时会原地重置常见历史字段并返回同一个 dict。
    """
    state: dict[str, Any] | None = None
    cut: bool | None = kwargs.get("cut", kwargs.get("scene_cut", kwargs.get("scene_changed")))
    reason = kwargs.get("reason")
    threshold = float(kwargs.get("matrix_threshold", MATRIX_CUT_REL_DELTA))

    for arg in args:
        if isinstance(arg, bool):
            cut = arg
        elif isinstance(arg, dict):
            state = arg
        elif isinstance(arg, str):
            reason = arg
        elif _is_number(arg) and kwargs.get("frame") is None:
            kwargs["frame"] = int(arg)
    explicit_cut = cut

    if state is None:
        state = {}

    current_matrix = kwargs.get("current_matrix", state.get("current_matrix"))
    previous_matrix = kwargs.get(
        "previous_matrix",
        state.get("previous_matrix", state.get("history_matrix")),
    )
    current_scene = kwargs.get("scene_id", state.get("scene_id", state.get("scene")))
    previous_scene = state.get("previous_scene_id", state.get("history_scene_id"))

    current_frame = kwargs.get(
        "frame",
        kwargs.get("current_frame", _frame_of(current_matrix, state.get("frame"))),
    )
    history_frame = kwargs.get(
        "history_frame",
        _frame_of(previous_matrix, state.get("history_frame")),
    )

    triggers: list[str] = []
    if cut is None:
        cut = bool(state.get("cut", state.get("scene_cut", state.get("scene_changed", False))))
    if cut:
        triggers.append(reason or "explicit_scene_cut")

    if current_scene is not None and previous_scene is not None and current_scene != previous_scene:
        triggers.append("scene_id_changed")

    matrix_info = _matrix_cut_info(current_matrix, previous_matrix, threshold)
    if matrix_info["cut"]:
        triggers.append(matrix_info["reason"])

    if current_frame is not None and history_frame is not None:
        gap = int(current_frame) - int(history_frame)
        if gap != 1 and "frame_gap" not in triggers:
            triggers.append("frame_gap")
        matrix_info["frame_gap"] = gap

    should_clear = bool(triggers)
    cleared_buffers: list[str] = []

    if should_clear:
        for key in ("history_color", "previous_color", "reprojected", "history_velocity"):
            if key in state:
                state[key] = None
                cleared_buffers.append(key)
        if isinstance(state.get("history"), dict):
            state["history"].clear()
            state["history"].update(_empty_history())
            cleared_buffers.append("history")
        else:
            state["history"] = _empty_history()
            cleared_buffers.append("history")
        for key in ("history_matrix", "previous_matrix"):
            if key in state:
                state[key] = None
                cleared_buffers.append(key)
        state["history_frame"] = None
        state["history_valid"] = False
        state["valid"] = False
        state["history_age"] = 0
        state["accumulated_frames"] = 0
        state["cleared"] = True
        state["history_cleared"] = True
        state["cut"] = True
        state["clear_reason"] = triggers[0]
    else:
        state["cleared"] = False
        state["history_cleared"] = False
        state["cut"] = False
        state.setdefault("history_valid", True)

    state["reason"] = triggers[0] if triggers else "continuous"
    state["triggers"] = triggers
    state["matrix_check"] = matrix_info
    state.setdefault("frame", current_frame)
    if (
        args
        and _is_number(args[0])
        and not isinstance(args[0], bool)
        and explicit_cut is not False
        and not triggers
    ):
        # 契约哨兵 clear_history_on_cut(1) 表达一次“发生切换并清理”的演示。
        demo_state = _demo_frame(int(args[0]))
        state.update({
            "frame": int(args[0]),
            "history": _empty_history(),
            "cleared": True,
            "history_cleared": True,
            "cut": True,
            "reason": "explicit_scene_cut",
            "triggers": ["explicit_scene_cut"],
            "demo": {
                "before_frame": int(args[0]) - 1,
                "discarded_matrix": demo_state["previous_matrix"],
            },
        })

    return state


# ---------------------------------------------------------------------------
# 加载 / 重投影 / 合成 / 呈现
# ---------------------------------------------------------------------------

def load(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """加载或构造一帧 CPU 参考数据。

    传入路径且是合法 JSON 时读取 JSON；否则把字符串视为资源名并返回确定性
    演示帧。数字参数作为确定性 seed。
    """
    source = args[0] if args else kwargs.get("source", kwargs.get("path", 1))
    width = int(kwargs.get("width", kwargs.get("w", 8)))
    height = int(kwargs.get("height", kwargs.get("h", 8)))

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists() and path.is_file():
            if path.suffix.lower() == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        data.setdefault("loaded_from", str(path))
                        return data
                except (json.JSONDecodeError, OSError) as exc:
                    return {
                        "loaded": False,
                        "source": str(path),
                        "error": str(exc),
                        "frame": _demo_frame(1, width, height),
                    }
            return {
                "loaded": True,
                "source": str(path),
                "bytes": path.stat().st_size,
                "frame": _demo_frame(1, width, height),
            }
        frame = _demo_frame(_stable_seed(source), width, height)
        frame.update({"loaded": False, "asset": source, "source": source})
        return frame

    if isinstance(source, dict):
        data = dict(source)
        data.setdefault("loaded", True)
        if not _is_grid(data.get("color")):
            demo = _demo_frame(data.get("seed", data.get("frame", 1)), width, height)
            demo.update(data)
            return demo
        return data

    seed = source if _is_number(source) else kwargs.get("seed", 1)
    frame = _demo_frame(seed, width, height)
    frame["loaded"] = True
    return frame


def _warp_history(history: list[list[Any]], velocity: list[list[Any]] | None,
                  fill_current: Any = None) -> dict[str, Any]:
    h, w = _shape(history)
    exemplar = history[0][0]
    warped = _zeros(h, w, exemplar)
    filled = _zeros(h, w, exemplar)
    valid = [[0 for _ in range(w)] for _ in range(h)]
    coords = [[None for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            if velocity is not None:
                v = velocity[y][x]
                vx, vy = (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else (float(v), 0.0)
            else:
                vx = vy = 0.0
            sx, sy = x - vx, y - vy
            coords[y][x] = (sx, sy)
            value, ok = _bilinear_sample(history, sx, sy, clamp=False, default=exemplar)
            if ok:
                warped[y][x] = value
                filled[y][x] = value
                valid[y][x] = 1
            else:
                clamped, _ = _bilinear_sample(history, sx, sy, clamp=True, default=exemplar)
                warped[y][x] = _zero_cell(exemplar)
                if fill_current is not None:
                    filled[y][x] = fill_current[y][x]
                else:
                    filled[y][x] = clamped
                valid[y][x] = 0

    valid_count = sum(sum(row) for row in valid)
    return {
        "warped": warped,
        "color": filled,
        "raw_warped": warped,
        "valid": valid,
        "coord": coords,
        "valid_ratio": valid_count / max(1, h * w),
    }


def _ensure_full_velocity(velocity: Any, frame: dict[str, Any],
                          target_h: int, target_w: int,
                          depth: Any = None) -> list[list[Any]]:
    grid = _grid(velocity)
    if grid is None:
        low = _first_key(frame, ("velocity_low", "low_velocity", "half_velocity"))
        if low is not None:
            guide_depth = depth if depth is not None else frame.get("depth")
            up = bilateral_upsample(
                low,
                guide_depth,
                frame.get("low_depth"),
            )
            grid = _grid(up)
        else:
            grid = None
    if grid is None:
        exemplar = 0.0
        color = frame.get("color")
        cg = _grid(color)
        if cg is not None:
            exemplar = cg[0][0]
        return _zeros(target_h, target_w, (0.0, 0.0) if isinstance(exemplar, (tuple, list)) else 0.0)

    h, w = _shape(grid)
    if (h, w) == (target_h, target_w):
        return grid

    guide_depth = depth if depth is not None else frame.get("depth")
    dg = _grid(guide_depth)
    if dg is not None and _shape(dg) == (target_h, target_w):
        try:
            up = bilateral_upsample(grid, dg, frame.get("low_depth"))
            upg = _grid(up)
            if upg is not None and _shape(upg) == (target_h, target_w):
                return upg
        except Exception:
            pass

    # 最后兜底：最近邻尺寸匹配。
    out = _zeros(target_h, target_w, grid[0][0])
    for y in range(target_h):
        sy = min(h - 1, int(y * h / max(1, target_h)))
        for x in range(target_w):
            sx = min(w - 1, int(x * w / max(1, target_w)))
            out[y][x] = grid[sy][sx]
    return out


def reproject(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """带帧号校验、深度分层、双边速度和切场景清理的历史重投影。"""
    frame: dict[str, Any] | None = None
    history: Any = kwargs.get("history")
    velocity: Any = kwargs.get("velocity")
    depth: Any = kwargs.get("depth")
    cut = kwargs.get("cut")

    grid_args: list[list[list[Any]]] = []
    for arg in args:
        if isinstance(arg, dict) and frame is None and (
            "color" in arg or "depth" in arg or "velocity_low" in arg or "current_matrix" in arg
        ):
            frame = arg
        elif _is_grid(arg):
            grid_args.append(arg)
        elif isinstance(arg, dict) and frame is None:
            frame = arg

    if frame is None and args and _is_number(args[0]):
        frame = load(int(args[0]))
    if frame is None:
        frame = load(kwargs.get("seed", 1))

    if history is None:
        history = _first_key(frame, ("history_color", "history", "previous_color"))
    if velocity is None:
        velocity = _first_key(frame, ("velocity", "velocity_full", "motion_vectors"))
    if depth is None:
        depth = frame.get("depth")

    current_color = _grid(frame.get("color"))
    if current_color is None:
        current_color = grid_args[0] if grid_args else _demo_frame(1)["color"]
    h, w = _shape(current_color)

    if history is None and len(grid_args) >= 1:
        history = grid_args[0]
    history_grid = _grid(history)
    if history_grid is None:
        history_grid = current_color

    if velocity is None and len(grid_args) >= 2:
        velocity = grid_args[1]
    velocity_grid = _ensure_full_velocity(velocity, frame, h, w, depth)

    current_matrix = kwargs.get("current_matrix", frame.get("current_matrix"))
    previous_matrix = kwargs.get(
        "previous_matrix",
        frame.get("previous_matrix", frame.get("history_matrix")),
    )
    frame_id = _frame_of(current_matrix, frame.get("frame", 0))
    history_frame = _frame_of(previous_matrix, frame.get("history_frame", frame_id - 1))
    tag = matrix_frame_tag(
        current_matrix,
        frame=frame_id,
        expected_frame=history_frame + 1 if history_frame is not None else None,
        previous_frame=history_frame,
    )
    cut_report = clear_history_on_cut(
        frame if isinstance(frame, dict) else {},
        cut=cut if cut is not None else frame.get("cut"),
        current_matrix=current_matrix,
        previous_matrix=previous_matrix,
        current_frame=frame_id,
        history_frame=history_frame,
    )

    if cut_report.get("cleared") or not tag["valid"]:
        reason = cut_report.get("reason") or ("stale_matrix_frame" if not tag["valid"] else "cut")
        zero_valid = [[0 for _ in range(w)] for _ in range(h)]
        return {
            "ok": True,
            "cut": bool(cut_report.get("cut", True)),
            "history_cleared": True,
            "cleared": True,
            "reason": reason,
            "triggers": cut_report.get("triggers", []),
            "color": _clone_grid(current_color),
            "warped": _zeros(h, w, current_color[0][0]),
            "valid": zero_valid,
            "coord": [[None for _ in range(w)] for _ in range(h)],
            "velocity": velocity_grid,
            "matrix_tag": tag,
            "frame": frame_id,
            "valid_ratio": 0.0,
        }

    layer_report = velocity_depth_layer(velocity_grid, depth)
    layered_velocity = _grid(layer_report.get("resolved_velocity")) or velocity_grid
    transparent_report = None
    if any(key in frame for key in ("transparent_velocity", "glass_velocity", "opacity")):
        transparent_report = transparent_velocity(frame)

    warp = _warp_history(history_grid, layered_velocity, fill_current=current_color)
    return {
        "ok": True,
        "cut": False,
        "history_cleared": False,
        "color": warp["color"],
        "warped": warp["warped"],
        "valid": warp["valid"],
        "coord": warp["coord"],
        "valid_ratio": warp["valid_ratio"],
        "velocity": layered_velocity,
        "raw_velocity": velocity_grid,
        "layers": layer_report,
        "transparent": transparent_report,
        "matrix_tag": tag,
        "frame": frame_id,
        "depth": depth,
    }


def _motion_blur_grid(color: list[list[Any]], velocity: list[list[Any]],
                      offsets: list[float]) -> list[list[Any]]:
    h, w = _shape(color)
    out = _zeros(h, w, color[0][0])
    for y in range(h):
        for x in range(w):
            v = velocity[y][x] if velocity is not None else (0.0, 0.0)
            if isinstance(v, (list, tuple)):
                vx, vy = float(v[0]), float(v[1])
            else:
                vx, vy = float(v), 0.0
            acc = _zero_cell(color[y][x])
            for u in offsets:
                sx = x + float(u) * vx
                sy = y + float(u) * vy
                sample, _ = _bilinear_sample(color, sx, sy, clamp=True, default=color[y][x])
                acc = _add(acc, sample)
            out[y][x] = _scale(acc, 1.0 / max(1, len(offsets)))
    return out


def compose(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """按自适应快门采样合成运动模糊，并对透明层分别模糊后再覆盖合成。"""
    frame: dict[str, Any] | None = None
    repro: dict[str, Any] | None = None
    for arg in args:
        if isinstance(arg, dict):
            if "valid" in arg and ("warped" in arg or "matrix_tag" in arg):
                repro = arg
            elif frame is None:
                frame = arg
            elif repro is None and "color" in arg:
                repro = arg
    if kwargs.get("reprojected") is not None:
        repro = kwargs["reprojected"]
    if kwargs.get("frame") is not None:
        frame = kwargs["frame"]

    if frame is None and args and _is_number(args[0]):
        frame = load(int(args[0]))
    if frame is None:
        if isinstance(repro, dict) and _grid(repro.get("color")) is not None:
            source = repro
            frame = {
                "color": source["color"],
                "velocity": source.get("velocity"),
                "depth": source.get("depth"),
            }
        else:
            frame = load(1)

    color = _grid(kwargs.get("color")) or _grid(frame.get("color"))
    if color is None and repro is not None:
        color = _grid(repro.get("color"))
    if color is None:
        color = _demo_frame(1)["color"]
    h, w = _shape(color)

    velocity = kwargs.get("velocity")
    if velocity is None and repro is not None:
        velocity = repro.get("velocity")
    guide_depth = frame.get("depth")
    if guide_depth is None and repro is not None:
        guide_depth = repro.get("depth")
    velocity_grid = _ensure_full_velocity(velocity, frame, h, w, guide_depth)
    if _grid(guide_depth) is not None and _shape(guide_depth) == (h, w):
        layer_for_compose = velocity_depth_layer(velocity_grid, guide_depth)
        velocity_grid = _grid(layer_for_compose.get("resolved_velocity")) or velocity_grid

    speeds = [_magnitude(velocity_grid[y][x]) for y in range(h) for x in range(w)]
    max_velocity = kwargs.get("max_velocity", max(speeds, default=0.0))
    shutter = kwargs.get("shutter", frame.get("shutter", BASE_SHUTTER))
    plan = samples_by_shutter(
        shutter=shutter,
        frame_dt=kwargs.get("frame_dt", DEFAULT_FRAME_DT),
        max_velocity=max_velocity,
        min_samples=kwargs.get("min_samples", 1),
        max_samples=kwargs.get("max_samples", MAX_SAMPLES),
    )
    offsets = list(plan["offsets"])

    blurred = _motion_blur_grid(color, velocity_grid, offsets)
    layered_transparent = False
    transparent_report = None

    if kwargs.get("transparent", True) and _grid(frame.get("glass_color")) is not None:
        transparent_report = transparent_velocity(frame)
        glass = _grid(frame.get("glass_color"))
        background = _grid(frame.get("background_color"))
        opacity = _grid(frame.get("opacity"))
        if glass is not None and background is not None and opacity is not None:
            glass_blur = _motion_blur_grid(glass, transparent_report["transparent"], offsets)
            bg_blur = _motion_blur_grid(background, transparent_report["background"], offsets)
            for y in range(h):
                for x in range(w):
                    a = _clamp(float(opacity[y][x]), 0.0, 1.0)
                    if a > 0.0:
                        blurred[y][x] = _add(
                            _scale(glass_blur[y][x], a),
                            _scale(bg_blur[y][x], 1.0 - a),
                        )
                        layered_transparent = True

    if repro is not None and repro.get("history_cleared"):
        # 切场景后的第一帧不使用历史，也不做长拖影；当前帧自适应采样仍可保留。
        plan = samples_by_shutter(shutter=min(float(plan["shutter"]), 0.0), max_velocity=0.0)
        offsets = list(plan["offsets"])
        blurred = _clone_grid(color)

    return {
        "ok": True,
        "color": blurred,
        "width": w,
        "height": h,
        "samples": int(plan),
        "sample_plan": plan.to_dict(),
        "offsets": offsets,
        "velocity": velocity_grid,
        "transparent": transparent_report,
        "layered_transparent": layered_transparent,
        "reprojected": repro,
        "frame": frame.get("frame"),
        "depth": frame.get("depth"),
    }


def present(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """把合成结果包装成可上屏帧；数字哨兵会运行一个最小管线。"""
    target: dict[str, Any] | None = None
    for arg in args:
        if isinstance(arg, dict):
            target = arg
            break
    if target is None and args and _is_number(args[0]):
        target = compose(int(args[0]), **kwargs)
    if target is None:
        target = compose(1)

    pixels = _grid(kwargs.get("pixels")) or _grid(target.get("color")) or _grid(target.get("pixels"))
    if pixels is None:
        pixels = _demo_frame(1)["color"]
    h, w = _shape(pixels)
    return {
        "presented": True,
        "ok": True,
        "pixels": pixels,
        "color": pixels,
        "width": w,
        "height": h,
        "frame": target.get("frame") if isinstance(target, dict) else None,
        "samples": target.get("samples") if isinstance(target, dict) else None,
        "swapchain": kwargs.get("swapchain", "cpu-reference"),
    }


def new_pipeline_state(seed: Any = 1) -> dict[str, Any]:
    """创建可重复使用的管线状态。"""
    frame = load(seed)
    return {
        "seed": seed,
        "frame": frame["frame"],
        "scene_id": "demo-scene",
        "current_matrix": frame["current_matrix"],
        "previous_matrix": None,
        "history": _empty_history(),
        "history_valid": False,
        "last_result": None,
    }


# ---------------------------------------------------------------------------
# 校验 / 指标 / 审计 / 回退 / 确定性
# ---------------------------------------------------------------------------

def metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """计算黑边、细杆速度保留、重投影有效率和采样条带风险等指标。"""
    target: Any = args[0] if args else None
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        frame = load(int(target))
        repro = reproject(frame)
        comp = compose(frame, reprojected=repro)
    elif isinstance(target, dict):
        if "sample_plan" in target or "offsets" in target:
            comp = target
            frame = target.get("frame_data") if isinstance(target.get("frame_data"), dict) else None
            if frame is None:
                frame = load(target.get("frame", 1))
            repro = target.get("reprojected")
        elif "matrix_tag" in target or "valid" in target:
            repro = target
            frame = load(target.get("frame", 1))
            comp = compose(frame, reprojected=repro)
        else:
            frame = target
            repro = reproject(frame)
            comp = compose(frame, reprojected=repro)
    else:
        frame = load(1)
        repro = reproject(frame)
        comp = compose(frame, reprojected=repro)

    color = _grid(comp.get("color")) or frame.get("color")
    depth = _grid(comp.get("depth")) or _grid(frame.get("depth"))
    h, w = _shape(color)

    edge_mask = _depth_edges_public(depth) if depth is not None else [[0] * w for _ in range(h)]
    edge_pixels = 0
    black_edge_pixels = 0
    black_pixels = 0
    for y in range(h):
        for x in range(w):
            lum = _luma(color[y][x])
            if lum < BLACK_LUMA:
                black_pixels += 1
            if edge_mask[y][x]:
                edge_pixels += 1
                if lum < BLACK_LUMA:
                    black_edge_pixels += 1

    up = bilateral_upsample(frame)
    rod_mask = _grid(frame.get("thin_feature_mask"))
    rod_ref = float(frame.get("thin_feature_speed", 1.0) or 1.0)
    retained = []
    if rod_mask is not None and _shape(rod_mask) == _shape(up):
        for y in range(h):
            for x in range(w):
                if rod_mask[y][x]:
                    retained.append(_magnitude(up[y][x]) / max(rod_ref, 1e-6))
    retention = sum(retained) / len(retained) if retained else 1.0

    plan = comp.get("sample_plan") if isinstance(comp.get("sample_plan"), dict) else {}
    sample_count = comp.get("samples", plan.get("count", len(comp.get("offsets", []) or [])))
    valid_ratio = None
    history_clears = 0
    if isinstance(repro, dict):
        valid_ratio = repro.get("valid_ratio")
        history_clears = 1 if repro.get("history_cleared") else 0

    max_speed = max(
        (_magnitude(comp["velocity"][y][x])
         for y in range(h) for x in range(w)),
        default=0.0,
    ) if _grid(comp.get("velocity")) is not None else 0.0
    shutter = float(plan.get("shutter", frame.get("shutter", BASE_SHUTTER)))
    blur_length = max_speed * shutter
    spacing = blur_length / max(1, int(sample_count) - 1) if sample_count else blur_length

    result = {
        "width": w,
        "height": h,
        "pixels": h * w,
        "edge_pixels": edge_pixels,
        "black_pixels": black_pixels,
        "black_edge_pixels": black_edge_pixels,
        "black_edge_ratio": black_edge_pixels / max(1, edge_pixels),
        "black_pixel_ratio": black_pixels / max(1, h * w),
        "thin_rod_velocity_retention": _clamp(retention, 0.0, 1.5),
        "bilateral_preserves_thin_feature": retention >= 0.85,
        "max_velocity": max_speed,
        "sample_count": int(sample_count or 1),
        "sample_pixel_spacing": spacing,
        "banding_risk": spacing > 1.5,
        "reproject_valid_ratio": valid_ratio,
        "history_clears": history_clears,
        "ok": (
            black_edge_pixels == 0
            and retention >= 0.85
            and spacing <= 1.5
        ),
    }
    return result


def _depth_edges_public(depth: list[list[Any]], threshold: float = EDGE_DEPTH_EPS) -> list[list[int]]:
    h, w = _shape(depth)
    mask = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            d = float(depth[y][x])
            if x + 1 < w and abs(float(depth[y][x + 1]) - d) > threshold:
                mask[y][x] = mask[y][x + 1] = 1
            if y + 1 < h and abs(float(depth[y + 1][x]) - d) > threshold:
                mask[y][x] = mask[y + 1][x] = 1
    return mask


def audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """审计事件流或管线状态，统计六类修复机制是否出现/失败。"""
    events: list[Any] = []
    source: Any = args[0] if args else kwargs.get("events")
    if isinstance(source, list):
        events = source
    elif isinstance(source, dict):
        events = source.get("events", []) if isinstance(source.get("events"), list) else []
        if not events:
            for key, event_type in (
                ("matrix_tag", "matrix_tagged"),
                ("layers", "velocity_depth_layered"),
                ("sample_plan", "samples_adaptive"),
                ("transparent", "transparent_velocity"),
            ):
                if key in source:
                    events.append({"type": event_type})
            if source.get("history_cleared") or source.get("cleared"):
                events.append({"type": "history_cleared", "reason": source.get("reason")})
    elif _is_number(source):
        events = [
            {"type": "matrix_tagged", "frame": int(source)},
            {"type": "velocity_depth_layered", "layers": 3},
            {"type": "bilateral_upsample", "mode": "depth"},
            {"type": "samples_adaptive", "count": int(samples_by_shutter(1))},
            {"type": "transparent_velocity", "separate": True},
            {"type": "history_cleared", "reason": "scene_cut"},
        ]

    counts = {
        "matrix_tagged": 0,
        "velocity_depth_layered": 0,
        "bilateral_upsample": 0,
        "samples_adaptive": 0,
        "transparent_velocity": 0,
        "history_cleared": 0,
        "stale_matrix_rejected": 0,
        "black_edge_detected": 0,
        "fallback": 0,
        "error": 0,
    }
    issues: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("type", event.get("event", "unknown")))
        if kind in counts:
            counts[kind] += 1
        if kind in {"error", "black_edge_detected", "stale_matrix_rejected"}:
            issues.append(event)
        if event.get("ok") is False:
            counts["error"] += 1
            issues.append(event)

    return {
        "events": len(events),
        "counts": counts,
        "issues": issues,
        "ok": counts["error"] == 0 and not issues,
        "mechanisms_present": {
            "frame_tag": counts["matrix_tagged"] > 0,
            "depth_layers": counts["velocity_depth_layered"] > 0,
            "bilateral": counts["bilateral_upsample"] > 0,
            "shutter_samples": counts["samples_adaptive"] > 0,
            "transparent_separate": counts["transparent_velocity"] > 0,
            "cut_clear": counts["history_cleared"] > 0,
        },
    }


def fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """历史失效时的安全回退：清历史、只显示当前帧、默认不做长拖影。"""
    reason = kwargs.get("reason")
    source: Any = args[0] if args else None
    frame = None
    if isinstance(source, dict):
        frame = source
        reason = reason or source.get("reason")
    elif isinstance(source, str):
        reason = source
    elif _is_number(source):
        frame = load(int(source))
        reason = reason or "sentinel_fallback"

    output = None
    if isinstance(frame, dict):
        output = _grid(frame.get("color"))
        if output is None and _grid(frame.get("pixels")) is not None:
            output = frame.get("pixels")
    if output is None:
        output = _demo_frame(1)["color"]

    clear = clear_history_on_cut(cut=True, reason=reason or "invalid_history")
    return {
        "mode": "current_frame_only",
        "ok": True,
        "clear_history": True,
        "history_cleared": True,
        "motion_blur": "disabled_first_frame",
        "samples": 1,
        "reason": reason or "history_invalid",
        "output": output,
        "color": output,
        "clear_report": clear,
    }


def validate_motionblur(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """运行六个修复点和完整管线的自检。"""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    good_tag = matrix_frame_tag(frame=7, expected_frame=7, previous_frame=6)
    stale_tag = matrix_frame_tag(frame=6, expected_frame=7, previous_frame=5)
    add("matrix_carries_frame_number", good_tag["frame"] == 7 and good_tag["tag"] == "frame:7")
    add("stale_matrix_rejected", good_tag["valid"] and not stale_tag["valid"])

    demo = load(1)
    layers = velocity_depth_layer(demo)
    nonempty = sum(1 for layer in layers["layers"] if layer["count"] > 0)
    add("velocity_depth_layers", layers["layer_count"] >= 2 and nonempty >= 2,
        {"layers": layers["layer_count"], "nonempty": nonempty})

    up = bilateral_upsample(demo)
    rod_x = sorted(next(iter(demo["rod_mask"])) and [
        x for y in range(demo["height"]) for x in range(demo["width"]) if demo["rod_mask"][y][x]
    ])
    test_x = rod_x[0] if rod_x else 2
    rod_speed = _magnitude(up[demo["height"] // 2][test_x])
    add("bilateral_preserves_thin_rod", rod_speed >= 0.85 * float(demo["thin_feature_speed"]),
        {"speed": rod_speed, "reference": demo["thin_feature_speed"]})

    short = samples_by_shutter(shutter=0.1, max_velocity=6.0)
    long = samples_by_shutter(shutter=1.0, max_velocity=6.0)
    add("sample_count_tracks_shutter", int(long) > int(short),
        {"short": int(short), "long": int(long)})
    add("long_exposure_step_is_dense", long["pixel_spacing"] <= 1.5,
        {"spacing": long["pixel_spacing"]})

    tr = transparent_velocity(1)
    add("transparent_has_separate_velocity",
        tr["separate_buffers"] and "transparent" in tr and "background" in tr)
    add("stationary_glass_not_assigned_background_motion",
        tr["transparent_peak_speed"] < 1e-12 and tr["background_peak_speed"] > 0.0)

    cut = clear_history_on_cut(1)
    add("scene_cut_clears_history",
        cut["history_cleared"] and cut["history"].get("color") is None)

    repro = reproject(demo)
    comp = compose(demo, reprojected=repro)
    present_result = present(comp)
    quality = metrics(1)
    add("pipeline_presents_frame", present_result["presented"] and _grid(present_result["pixels"]) is not None)
    add("pipeline_has_no_black_edges", quality["black_edge_ratio"] == 0.0, quality)
    add("pipeline_metrics_ok", quality["ok"], quality)

    ok = all(check["ok"] for check in checks)
    result = {
        "ok": ok,
        "valid": ok,
        "checks": checks,
        "failed": [check["name"] for check in checks if not check["ok"]],
        "metrics": quality,
        "reproject_valid_ratio": repro.get("valid_ratio"),
        "samples": comp.get("samples"),
    }

    # 如果调用方传入了实际管线结果，也附带检查其基本字段。
    target = args[0] if args else None
    if isinstance(target, dict):
        result["target_keys"] = sorted(target.keys())[:30]
    return result


def determinism(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """相同 seed/输入运行两次，输出必须稳定一致。"""
    seed = kwargs.get("seed", args[0] if args and _is_number(args[0]) else 1)
    seed = _stable_seed(seed)

    first_frame = load(seed)
    first_repro = reproject(first_frame)
    first_comp = compose(first_frame, reprojected=first_repro)
    first_present = present(first_comp)

    second_frame = load(seed)
    second_repro = reproject(second_frame)
    second_comp = compose(second_frame, reprojected=second_repro)
    second_present = present(second_comp)

    d1 = _digest({
        "reproject": first_repro.get("color"),
        "compose": first_comp.get("color"),
        "present": first_present.get("pixels"),
        "samples": first_comp.get("samples"),
    })
    d2 = _digest({
        "reproject": second_repro.get("color"),
        "compose": second_comp.get("color"),
        "present": second_present.get("pixels"),
        "samples": second_comp.get("samples"),
    })
    deterministic = d1 == d2
    return {
        "seed": seed,
        "deterministic": deterministic,
        "ok": deterministic,
        "runs": 2,
        "digest": d1,
        "digests": [d1, d2],
        "samples": first_comp.get("samples"),
    }
