# pairwise-motion-blur-reproject

## 领域

游戏、图形与多媒体

## 主题

运动模糊重投影

## 说明

矩阵帧号、双边上采样与快门采样。

初始状态下 `graphics/motion_blur_reproject.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
