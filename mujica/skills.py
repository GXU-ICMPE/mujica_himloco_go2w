"""One skill interface shared by environments, training, export and playback."""

TASK_FAMILY = "terrain_locomotion"
TASK_CONTRACT_VERSION = 2
SKILL_NAMES = ("flat_slope", "discrete", "stairs")
SKILL_VALUES = {name: index for index, name in enumerate(SKILL_NAMES)}
TERRAIN_GROUPS = (("flat", "slope_up", "slope_down"), ("discrete",), ("stairs_up", "stairs_down"))


def skill_metadata():
    return dict(task_family=TASK_FAMILY, task_contract_version=TASK_CONTRACT_VERSION,
                skill_names=list(SKILL_NAMES), skill_values=dict(SKILL_VALUES),
                terrain_groups={name: list(group) for name, group in zip(SKILL_NAMES, TERRAIN_GROUPS)})


def validate_skill_metadata(metadata):
    if (metadata.get("task_family") != TASK_FAMILY
            or metadata.get("task_contract_version") != TASK_CONTRACT_VERSION
            or metadata.get("skill_names") != list(SKILL_NAMES)
            or metadata.get("skill_values") != SKILL_VALUES
            or metadata.get("terrain_groups") != skill_metadata()["terrain_groups"]):
        raise ValueError("Incompatible skill contract: expected v2 flat_slope/discrete/stairs. "
                         "Old Moving/Climb/Recovery checkpoints cannot be relabeled or resumed; train a new S1.")
