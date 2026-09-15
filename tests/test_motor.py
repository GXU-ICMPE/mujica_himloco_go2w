import math
import torch
from mujica.motor import DCMotorLimiter


def test_speed_envelope_and_calf_peak_position():
    motor = DCMotorLimiter(["FL_thigh_joint", "FL_calf_joint"], [24., 36.], [30., 20.])
    q = torch.tensor([[0., -math.pi/2], [0., -2.72], [0., -math.pi/2]])
    qd = torch.tensor([[0., 0.], [0., 0.], [30., 20.]])
    limit = motor.limits(q, qd)
    assert torch.allclose(limit[0], torch.tensor([24., 36.]))
    assert limit[1, 1] < limit[0, 1]
    assert torch.equal(limit[2], torch.zeros(2))
    assert torch.allclose(motor.limits(q[:1], torch.tensor([[22.5, 15.]])),
                          torch.tensor([[12., 18.]]))


def test_calibrated_override_clip_and_disabled_static_limits():
    config = {"enabled": False, "joints": {"wheel": {"peak_torque": 40., "no_load_speed": 60.}}}
    motor = DCMotorLimiter(["wheel"], [24.], [30.], config)
    q = torch.zeros(2, 1)
    applied = motor.clip(torch.tensor([[50.], [-50.]]), q, torch.ones_like(q)*100)
    assert torch.equal(applied, torch.tensor([[40.], [-40.]]))


def test_invalid_motor_name_and_speed_rejected():
    import pytest
    with pytest.raises(ValueError):
        DCMotorLimiter(["wheel"], [24.], [0.])
    with pytest.raises(ValueError):
        DCMotorLimiter(["wheel"], [24.], [30.], {"joints": {"missing": {}}})
