import pytest
import torch

from ricedx.models import count_flops, count_params, create_model

CFGS = [
    {"name": "inception_v4", "timm": "inception_v4.tf_in1k", "img_size": 299},
    {"name": "mobilenetv3_large", "timm": "mobilenetv3_large_100.ra_in1k", "img_size": 224},
    {"name": "deit_small", "timm": "deit_small_patch16_224.fb_in1k", "img_size": 224},
]


@pytest.mark.parametrize("cfg", CFGS, ids=lambda c: c["name"])
def test_forward_shape(cfg):
    m = create_model(cfg, 4, pretrained=False, img_size=128).eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 3, 128, 128))
    assert out.shape == (2, 4)
    assert count_params(m) > 0 and count_flops(m, 128) > 0


def test_gradcam_on_inception():
    from ricedx.explain import gradcam

    m = create_model(CFGS[0], 4, pretrained=False, img_size=128)
    cam, cls, p = gradcam(m, torch.rand(1, 3, 128, 128))
    assert cam.shape == (128, 128) and 0 <= cls < 4 and 0 <= p <= 1
    assert cam.min() >= 0 and cam.max() <= 1
