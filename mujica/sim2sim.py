"""Run an exported recurrent MUJICA policy with MuJoCo (CPU or viewer).

``python -m mujica.sim2sim --policy mujica.pt --metadata mujica.json --headless``

Joint order is an explicit export contract. Never assume that Isaac Gym DOF order,
MuJoCo qpos order, and actuator order are the same. The bundled MJCF calls the
URDF's ``*_foot_joint`` joints ``*_wheel_joint``; this one alias is handled by name.
The default scene is constructed in memory from the bundled robot XML plus a
plane, avoiding the upstream demonstration scene's machine-specific asset paths.
No downloads or robot communication happen in this module.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np


from .skills import SKILL_VALUES, validate_skill_metadata

SKILLS = {"auto": -1, **SKILL_VALUES}
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_joint_names(policy_names, model_names):
    """Resolve exact names first, then the bundled Go2W foot/wheel alias.

    Return names in policy order; reject missing/ambiguous mappings rather than
    silently controlling a different joint.
    """
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("Export metadata contains duplicate joint_names")
    available = set(model_names)
    resolved = []
    for name in policy_names:
        alternate = name[:-len("_foot_joint")] + "_wheel_joint"
        if name in available:
            candidate = name
        elif name.endswith("_foot_joint") and alternate in available:
            candidate = alternate
        else:
            raise ValueError(f"Policy joint {name!r} is absent from the MuJoCo model")
        if candidate in resolved:
            raise ValueError(f"Multiple policy joints map to {candidate!r}")
        resolved.append(candidate)
    return resolved


@dataclass(frozen=True)
class JointMapping:
    """Indices below are arrays in *policy* joint order."""

    joint_names: tuple
    joint_ids: np.ndarray
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    actuator_ids: np.ndarray
    actuator_torque_per_ctrl: np.ndarray

    @classmethod
    def from_model(cls, model, policy_names, mj):
        model_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
                       for j in range(model.njnt)]
        resolved = resolve_joint_names(policy_names, model_names)
        ids = np.array([model_names.index(name) for name in resolved], dtype=int)
        actuator_ids, factors = [], []
        for name, jid in zip(resolved, ids):
            if int(model.jnt_type[jid]) not in (
                    int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
                raise ValueError(f"{name}: policy DOFs must be scalar hinge/slide joints")
            matches = np.flatnonzero(
                (model.actuator_trnid[:, 0] == jid)
                & (model.actuator_trntype == int(mj.mjtTrn.mjTRN_JOINT)))
            if len(matches) != 1:
                raise ValueError(f"{name}: expected exactly one joint motor, found {len(matches)}")
            aid = int(matches[0])
            if (int(model.actuator_dyntype[aid]) != int(mj.mjtDyn.mjDYN_NONE)
                    or int(model.actuator_gaintype[aid]) != int(mj.mjtGain.mjGAIN_FIXED)
                    or int(model.actuator_biastype[aid]) != int(mj.mjtBias.mjBIAS_NONE)):
                raise ValueError(f"{name}: use a plain torque motor, not a position/velocity actuator")
            factor = float(model.actuator_gear[aid, 0] * model.actuator_gainprm[aid, 0])
            if not np.isfinite(factor) or abs(factor) < 1e-12:
                raise ValueError(f"{name}: invalid actuator gear/gain")
            actuator_ids.append(aid)
            factors.append(factor)
        return cls(tuple(resolved), ids, model.jnt_qposadr[ids].copy(),
                   model.jnt_dofadr[ids].copy(), np.array(actuator_ids), np.array(factors))

    def state(self, data):
        return data.qpos[self.qpos_indices].copy(), data.qvel[self.qvel_indices].copy()

    def write_torques(self, data, torque):
        # Unrelated scene actuators must not retain commands from an earlier step.
        data.ctrl[:] = 0.0
        data.ctrl[self.actuator_ids] = np.asarray(torque) / self.actuator_torque_per_ctrl


def _vector(metadata, key, count, fallback=None):
    value = metadata.get(key, fallback)
    if value is None:
        raise ValueError(f"Missing export metadata: {key}")
    if isinstance(value, dict):
        value = [value[name] for name in metadata["joint_names"]]
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"metadata.{key} must contain {count} finite values in policy order")
    return array


def validate_metadata(metadata):
    validate_skill_metadata(metadata)
    names = metadata.get("joint_names", [])
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError("metadata.joint_names must contain 16 unique names from the training environment")
    if metadata.get("frame_dim") != 58 or metadata.get("history_len") != 6:
        raise ValueError("This runner expects the MUJICA Go2W 58-value frame and 6-frame history")
    if int(metadata.get("hidden_dim", 0)) <= 0:
        raise ValueError("metadata.hidden_dim must be positive")
    wheels = metadata.get("wheel_indices", [])
    if len(wheels) != 4 or len(set(wheels)) != 4 or any(
            not isinstance(i, int) or i < 0 or i >= 16 for i in wheels):
        raise ValueError("metadata.wheel_indices must contain four distinct policy DOF indices")
    for key in ("default_dof_pos", "p_gains", "d_gains", "torque_limits", "velocity_limits"):
        value = _vector(metadata, key, 16)
        if key in ("torque_limits", "velocity_limits") and np.any(value <= 0):
            raise ValueError(f"metadata.{key} must be positive")
    for key in ("sim_dt", "control_dt", "clip_actions"):
        value = float(metadata.get(key, 0))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"metadata.{key} must be positive")
    ratio = float(metadata["control_dt"]) / float(metadata["sim_dt"])
    if not math.isclose(ratio, round(ratio), abs_tol=1e-7) or ratio < 1:
        raise ValueError("control_dt must be an integer multiple of sim_dt")


def build_frame(angular_velocity_body, gravity_body, commands, q, qd,
                previous_action, skill, metadata):
    """58-D deployable frame; wheel positions never mutate simulation state."""
    scales = metadata.get("obs_scales", {})
    error = np.asarray(q, dtype=float).copy() - _vector(metadata, "default_dof_pos", 16)
    error[metadata["wheel_indices"]] = 0.0
    command_scale = np.array([scales.get("lin_vel", 2.0),
                              scales.get("lin_vel", 2.0), scales.get("ang_vel", 0.25)])
    frame = np.concatenate((
        np.asarray(angular_velocity_body) * scales.get("ang_vel", 0.25),
        np.asarray(gravity_body), np.asarray(commands) * command_scale,
        error * scales.get("dof_pos", 1.0),
        np.asarray(qd) * scales.get("dof_vel", 0.05),
        np.asarray(previous_action), [float(skill)]))
    if frame.shape != (58,) or not np.all(np.isfinite(frame)):
        raise ValueError("Non-finite or malformed proprioceptive observation")
    limit = float(metadata.get("clip_observations", 100.0))
    return np.clip(frame, -limit, limit).astype(np.float32)


def pd_torque(action, q, qd, metadata):
    """Requested PD torque before the common calibrated motor limiter.

    Do not clip to URDF peaks here: calibrated motor peaks can differ, and the
    training environment passes the requested torque directly to its limiter.
    """
    action = np.clip(np.asarray(action), -float(metadata["clip_actions"]),
                     float(metadata["clip_actions"]))
    wheels = metadata["wheel_indices"]
    position_error = _vector(metadata, "default_dof_pos", 16) - np.asarray(q)
    position_error += float(metadata.get("action_scale", 0.25)) * action
    position_error[wheels] = 0.0
    velocity_reference = np.zeros(16)
    velocity_reference[wheels] = float(metadata.get("vel_scale", 10.0)) * action[wheels]
    torque = (_vector(metadata, "p_gains", 16) * position_error
              + _vector(metadata, "d_gains", 16) * (velocity_reference - np.asarray(qd)))
    return torque


def effective_torque_limits(metadata):
    """Maximum exported actuator peaks, including per-joint calibration."""
    limits = _vector(metadata, "torque_limits", 16).copy()
    names = metadata["joint_names"]
    for name, entry in metadata.get("motor", {}).get("joints", {}).items():
        if name not in names:
            raise ValueError("Unknown calibrated motor name: " + name)
        if "peak_torque" in entry:
            limits[names.index(name)] = float(entry["peak_torque"])
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("Calibrated torque limits must be positive and finite")
    return limits


def update_history(history, frame, reset=False):
    """Newest-first history; match the training environment's reset padding."""
    if reset:
        history[:] = frame
    else:
        history[1:] = history[:-1].copy()
        history[0] = frame


def default_scene_xml(metadata, robot_xml=None):
    """Use the shipped robot without modifying source assets or visual meshes."""
    robot_xml = Path(robot_xml or REPO_ROOT / "resources/robots/go2w/mjcf/go2w.xml")
    tree = ET.parse(robot_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((robot_xml.parent / compiler.get("meshdir", "")).resolve()))
        if compiler.get("texturedir"):
            compiler.set("texturedir", str((robot_xml.parent / compiler.get("texturedir")).resolve()))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(metadata["sim_dt"]))
    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"No worldbody in {robot_xml}")
    ET.SubElement(world, "geom", name="mujica_flat_floor", type="plane", size="0 0 0.1",
                  friction="0.8 0.02 0.01", rgba="0.25 0.29 0.31 1")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true")
    joints = [j.get("name") for j in world.iter("joint")]
    resolved = resolve_joint_names(metadata["joint_names"], joints)
    limits = dict(zip(resolved, effective_torque_limits(metadata)))
    # The upstream MJCF hard-codes limits that differ from its URDF. Here the
    # controller and MuJoCo motor use the training export's explicit limits.
    for motor in root.findall("./actuator/motor"):
        name = motor.get("joint")
        if name in limits:
            gear = float(motor.get("gear", "1").split()[0])
            limit = limits[name] / abs(gear)
            motor.set("ctrlrange", f"{-limit} {limit}")
    return ET.tostring(root, encoding="unicode")


def _base_joint(model, mapping, mj):
    """Locate the root of this robot, independent of other scene free bodies."""
    body = int(model.jnt_bodyid[mapping.joint_ids[0]])
    while body:
        start, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
        for jid in range(start, start + count):
            if int(model.jnt_type[jid]) == int(mj.mjtJoint.mjJNT_FREE):
                return body, jid
        body = int(model.body_parentid[body])
    raise ValueError("Robot must have a floating-base free joint")


def _check_scene_limits(model, mapping, metadata):
    expected = effective_torque_limits(metadata)
    for i, aid in enumerate(mapping.actuator_ids):
        if model.actuator_ctrllimited[aid]:
            actual = np.sort(model.actuator_ctrlrange[aid] * mapping.actuator_torque_per_ctrl[i])
            if actual[0] > -expected[i] + 1e-6 or actual[1] < expected[i] - 1e-6:
                warnings.warn(f"Scene limits {mapping.joint_names[i]} torque to {actual.tolist()}, "
                              f"narrower than training ±{expected[i]:g}", stacklevel=2)


def run(args):
    # Lazy imports make --help and observation/mapping tests independent of both
    # Isaac Gym and MuJoCo. A missing optional dependency is never auto-installed.
    try:
        import mujoco as mj
    except ImportError as exc:
        raise SystemExit("MuJoCo is required for sim2sim. Install the project's sim2sim requirements.") from exc
    import torch
    torch.set_num_threads(args.threads)

    metadata_path = Path(args.metadata) if args.metadata else Path(args.policy).with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata)
    if args.skill == "auto" and not metadata.get("has_selector", False):
        raise SystemExit("This S1 export has no selector. Set --skill " + ", ".join(SKILL_VALUES) + ".")
    policy = torch.jit.load(args.policy, map_location="cpu").eval()
    if args.scene:
        model = mj.MjModel.from_xml_path(str(Path(args.scene).resolve()))
    else:
        model = mj.MjModel.from_xml_string(default_scene_xml(metadata))
    model.opt.timestep = float(metadata["sim_dt"])
    data = mj.MjData(model)
    mapping = JointMapping.from_model(model, metadata["joint_names"], mj)
    base_body, base_joint = _base_joint(model, mapping, mj)
    _check_scene_limits(model, mapping, metadata)

    from mujica.motor import DCMotorLimiter
    # Even with the speed/position envelope disabled, this applies static peaks
    # and optional calibration exactly as in the training environment.
    motor = DCMotorLimiter(metadata["joint_names"],
                           torch.tensor(metadata["torque_limits"], dtype=torch.float32),
                           torch.tensor(metadata["velocity_limits"], dtype=torch.float32),
                           config=metadata.get("motor", {}))

    # Reset all policy state together. We do not reset the GRU on a skill switch.
    mj.mj_resetData(model, data)
    base_qadr = int(model.jnt_qposadr[base_joint])
    data.qpos[base_qadr + 2] = float(metadata.get("init_base_height", 0.45))
    if args.initial_pose == "back":
        data.qpos[base_qadr + 3:base_qadr + 7] = [0, 1, 0, 0]
    elif args.initial_pose == "side":
        data.qpos[base_qadr + 3:base_qadr + 7] = [math.sqrt(0.5), math.sqrt(0.5), 0, 0]
    else:
        data.qpos[base_qadr + 3:base_qadr + 7] = [1, 0, 0, 0]
    data.qpos[mapping.qpos_indices] = _vector(metadata, "default_dof_pos", 16)
    mj.mj_forward(model, data)
    history = np.zeros((6, 58), dtype=np.float32)
    hidden = torch.zeros((1, int(metadata["hidden_dim"])), dtype=torch.float32)
    action = np.zeros(16, dtype=np.float32)
    override = torch.tensor([SKILLS[args.skill]], dtype=torch.long)
    selected_skill = max(0, SKILLS[args.skill])
    commands = np.array([args.vx, args.vy, args.yaw], dtype=float)
    control_steps = round(float(metadata["control_dt"]) / float(metadata["sim_dt"]))
    total_steps = math.ceil(args.seconds / float(metadata["sim_dt"]))
    velocity = np.zeros(6)
    last_skill = None

    viewer_context = nullcontext(None)
    if not args.headless:
        import mujoco.viewer
        viewer_context = mujoco.viewer.launch_passive(model, data)

    print("Policy joint order: " + ", ".join(metadata["joint_names"]))
    print(f"MuJoCo {model.opt.timestep:g}s, policy {metadata['control_dt']:g}s, "
          f"skill={args.skill}, initial_pose={args.initial_pose}")
    with viewer_context as viewer, torch.inference_mode():
        if viewer is not None:
            viewer.cam.lookat[:] = data.xpos[base_body]
            viewer.cam.distance = 2.5
        for step in range(total_steps):
            if viewer is not None and not viewer.is_running():
                break
            started = time.monotonic()
            q, qd = mapping.state(data)
            if step % control_steps == 0:
                mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, base_body, velocity, 1)
                rotation = data.xmat[base_body].reshape(3, 3)
                gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
                frame = build_frame(velocity[:3], gravity_body, commands, q, qd,
                                    action, selected_skill, metadata)
                update_history(history, frame, reset=step == 0)
                output, hidden, chosen = policy(torch.from_numpy(history).unsqueeze(0), hidden, override)
                action = output.detach().cpu().numpy().reshape(-1)
                if action.shape != (16,) or not np.all(np.isfinite(action)):
                    raise RuntimeError("Policy returned invalid actions")
                action = np.clip(action, -float(metadata["clip_actions"]), float(metadata["clip_actions"]))
                if not torch.isfinite(hidden).all():
                    raise RuntimeError("Policy returned non-finite recurrent state")
                selected_skill = int(chosen.item())
                if selected_skill not in (0, 1, 2):
                    raise RuntimeError(f"Policy returned invalid skill: {selected_skill}")
                # The export updates a clone. Preserve the actually used skill
                # when this frame becomes a historical observation next step.
                history[0, -1] = selected_skill
                if selected_skill != last_skill:
                    print(f"t={data.time:.2f}s skill={list(SKILLS)[selected_skill + 1]}")
                    last_skill = selected_skill
            torque = pd_torque(action, q, qd, metadata)
            torque = motor.clip(torch.from_numpy(torque).float().unsqueeze(0),
                                torch.from_numpy(q).float().unsqueeze(0),
                                torch.from_numpy(qd).float().unsqueeze(0)).squeeze(0).cpu().numpy()
            mapping.write_torques(data, torque)
            mj.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                raise RuntimeError("MuJoCo produced a non-finite state")
            if viewer is not None:
                viewer.cam.lookat[:] = data.xpos[base_body]
                viewer.sync()
                remaining = model.opt.timestep - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    print(f"Finished {data.time:.3f}s simulated; base_xyz="
          f"{np.round(data.xpos[base_body], 4).tolist()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="TorchScript policy exported by mujica.export")
    parser.add_argument("--metadata", help="Export JSON; defaults to policy path with .json suffix")
    parser.add_argument("--scene", help="Optional MuJoCo scene XML; default is the bundled Go2W on a plane")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--threads", type=int, default=1, help="CPU Torch inference threads")
    parser.add_argument("--skill", choices=SKILLS, default="auto")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw angular velocity command (rad/s)")
    parser.add_argument("--initial-pose", choices=("standing", "side", "back"), default="standing")
    args = parser.parse_args(argv)
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not all(np.isfinite([args.vx, args.vy, args.yaw])):
        parser.error("velocity commands must be finite")
    run(args)


if __name__ == "__main__":
    main()
