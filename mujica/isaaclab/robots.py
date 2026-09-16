"""Robot-specific policy ordering; never depend on PhysX joint enumeration."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joints: tuple
    wheels: tuple
    hips: tuple
    feet: tuple
    collision_groups: tuple

    @property
    def urdf(self):
        return Path(__file__).resolve().parents[2] / f"resources/robots/{self.name}/urdf/{self.name}.urdf"

    @property
    def critic_dim(self):
        return 58 + 3 + len(self.collision_groups) + 4 + 187


GO2W = RobotSpec(
    "go2w",
    tuple(f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf", "foot")),
    tuple(f"{leg}_foot_joint" for leg in ("FL", "FR", "RL", "RR")),
    tuple(f"{leg}_hip_joint" for leg in ("FL", "FR", "RL", "RR")),
    tuple(f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")),
    (("base",), ("Head_upper", "Head_lower")) + tuple(
        (f"{leg}_{part}",) for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf", "foot")),
)
X5 = RobotSpec(
    "x5",
    tuple(f"{leg}_{part}" for leg in ("RF", "LF", "RH", "LH") for part in ("HAA", "HFE", "KFE"))
    + tuple(f"{leg}_WHEEL" for leg in ("RF", "LF", "RH", "LH")),
    tuple(f"{leg}_WHEEL" for leg in ("RF", "LF", "RH", "LH")),
    tuple(f"{leg}_HAA" for leg in ("RF", "LF", "RH", "LH")),
    tuple(f"{leg}_FOOT" for leg in ("RF", "LF", "RH", "LH")),
    (("base",),) + tuple((f"{leg}_{part}",) for leg in ("RF", "LF", "RH", "LH")
                          for part in ("hip", "thigh", "calf", "FOOT")),
)


def robot_spec(name="go2w"):
    try:
        return {"go2w": GO2W, "x5": X5}[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported robot: {name}") from exc
