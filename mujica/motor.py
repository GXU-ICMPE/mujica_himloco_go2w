"""Configurable DC-motor envelope (MUJICA IV-C).

The paper does not publish fitted coefficients. The default derives peak torque
and speed from the supplied URDF; knee/position factors are UNCALIBRATED
engineering assumptions, not a reconstruction of Unitree's motor manual.
Torque clipping is an actuator model, NOT a P3O cost constraint.
"""
import torch


DEFAULT_MOTOR_CONFIG = {
    "enabled": True,
    "calibrated": False,
    "knee_fraction": 0.5,
    "calf_min_factor": 0.3,
    "calf_cos_offset": 0.0,
    "calf_cos_amplitude": 1.0,
    "calf_cos_phase": -1.5707963267948966,
}


class DCMotorLimiter:
    def __init__(self, dof_names, torque_limits, velocity_limits, config=None):
        self.config = dict(DEFAULT_MOTOR_CONFIG)
        if config is not None:
            if not isinstance(config, dict):
                config = {k: getattr(config, k) for k in DEFAULT_MOTOR_CONFIG
                          if hasattr(config, k)}
            self.config.update(config)
        self.names = list(dof_names)
        self.peak = torch.as_tensor(torque_limits).detach().clone().float()
        self.speed = torch.as_tensor(velocity_limits, device=self.peak.device).detach().clone().float()
        self.calf = torch.tensor(["calf" in name for name in self.names], device=self.peak.device)
        if self.peak.numel() != len(self.names) or self.speed.numel() != len(self.names):
            raise ValueError("One torque and velocity limit is required per DOF")
        if not bool(torch.all(self.peak > 0)) or not bool(torch.all(self.speed > 0)):
            raise ValueError("Motor torque and speed limits must be positive")
        if not 0 <= self.config["knee_fraction"] < 1:
            raise ValueError("knee_fraction must be in [0, 1)")
        if not 0 <= self.config["calf_min_factor"] <= 1:
            raise ValueError("calf_min_factor must be in [0, 1]")
        # Optional measured limits keyed by the exact simulator DOF names.
        for name, entry in self.config.get("joints", {}).items():
            if name not in self.names:
                raise ValueError("Unknown calibrated motor name: " + name)
            idx = self.names.index(name)
            if "peak_torque" in entry:
                self.peak[idx] = float(entry["peak_torque"])
            if "no_load_speed" in entry:
                self.speed[idx] = float(entry["no_load_speed"])
        if not bool(torch.all(self.peak > 0)) or not bool(torch.all(self.speed > 0)):
            raise ValueError("Measured torque and speed limits must be positive")

    def limits(self, q, qd):
        peak, speed = self.peak.to(qd), self.speed.to(qd)
        if not self.config["enabled"]:
            return peak.expand_as(qd)
        knee = self.config["knee_fraction"] * speed
        speed_factor = ((speed - qd.abs()) / (speed - knee)).clamp(0.0, 1.0)
        cfg = self.config
        calf_factor = (cfg["calf_cos_offset"] + cfg["calf_cos_amplitude"] *
                       torch.cos(q - cfg["calf_cos_phase"]).abs()).clamp(cfg["calf_min_factor"], 1.0)
        pos_factor = torch.where(self.calf.to(qd.device), calf_factor, torch.ones_like(q))
        return peak * speed_factor * pos_factor

    def clip(self, torque, q, qd):
        limit = self.limits(q, qd)
        return torch.maximum(torch.minimum(torque, limit), -limit)

    def violation_count(self, torque, q, qd):
        # This metric is diagnostic only and is not added to the user reward.
        return (torque.abs() >= self.limits(q, qd)).sum(dim=-1)
