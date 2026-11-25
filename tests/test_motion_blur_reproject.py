import pytest
import graphics.motion_blur_reproject as mod

CASES = [
    ("matrix_frame_tag", (1,)),
    ("velocity_depth_layer", (1,)),
    ("bilateral_upsample", (1,)),
    ("samples_by_shutter", (1,)),
    ("transparent_velocity", (1,)),
    ("clear_history_on_cut", (1,)),
    ("load", (1,)),
    ("reproject", (1,)),
    ("compose", (1,)),
    ("present", (1,)),
    ("validate_motionblur", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
