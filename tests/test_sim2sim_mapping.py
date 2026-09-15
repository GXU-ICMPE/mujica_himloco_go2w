"""Deployment contract tests, runnable without Isaac Gym or MuJoCo."""

from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from mujica.skills import skill_metadata

from mujica.sim2sim import (JointMapping, build_frame, default_scene_xml,
                           effective_torque_limits, pd_torque, resolve_joint_names,
                           update_history, validate_metadata)


def metadata():
    # Deliberately grouped by joint type, unlike the MJCF's per-leg traversal.
    names = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf", "foot")
             for leg in ("FL", "RL", "FR", "RR")]
    return {**skill_metadata(), "joint_names": names, "wheel_indices": [12, 13, 14, 15],
            "default_dof_pos": [0.] * 4 + [0.8] * 4 + [-1.5] * 4 + [0.] * 4,
            "p_gains": [40.] * 12 + [0.] * 4,
            "d_gains": [1.] * 12 + [0.5] * 4,
            "torque_limits": [100.] * 16, "velocity_limits": [30.] * 16,
            "action_scale": 0.25, "vel_scale": 10., "clip_actions": 100.,
            "frame_dim": 58, "history_len": 6, "hidden_dim": 128,
            "sim_dt": 0.005, "control_dt": 0.02}


def fake_model():
    # An unrelated free body precedes the robot; actuators have a third order.
    names = [None, None] + [f"{leg}_{joint}_joint" for leg in ("RR", "FL", "FR", "RL")
                            for joint in ("hip", "thigh", "calf", "wheel")]
    actuator_joints = np.roll(np.arange(2, 18), 5)
    gear = np.zeros((16, 6))
    gear[:, 0] = np.where(np.arange(16) % 2, -2., 3.)
    gain = np.zeros((16, 10))
    gain[:, 0] = 1.0
    model = SimpleNamespace(njnt=18,
                            jnt_type=np.array([0, 0] + [3] * 16),
                            jnt_qposadr=np.array([0, 7] + list(range(14, 30))),
                            jnt_dofadr=np.array([0, 6] + list(range(12, 28))),
                            actuator_trnid=np.column_stack((actuator_joints, np.full(16, -1))),
                            actuator_trntype=np.zeros(16, dtype=int),
                            actuator_dyntype=np.zeros(16, dtype=int),
                            actuator_gaintype=np.zeros(16, dtype=int),
                            actuator_biastype=np.zeros(16, dtype=int),
                            actuator_gear=gear, actuator_gainprm=gain)
    mj = SimpleNamespace(
        mj_id2name=lambda model, obj, index: names[index],
        mjtObj=SimpleNamespace(mjOBJ_JOINT=3),
        mjtJoint=SimpleNamespace(mjJNT_HINGE=3, mjJNT_SLIDE=2),
        mjtTrn=SimpleNamespace(mjTRN_JOINT=0),
        mjtDyn=SimpleNamespace(mjDYN_NONE=0),
        mjtGain=SimpleNamespace(mjGAIN_FIXED=0),
        mjtBias=SimpleNamespace(mjBIAS_NONE=0))
    return model, mj


class DeploymentContractTests(unittest.TestCase):
    def test_mapping_handles_three_independent_orders_and_motor_gear(self):
        meta = metadata()
        model, mj = fake_model()
        mapping = JointMapping.from_model(model, meta["joint_names"], mj)
        data = SimpleNamespace(qpos=np.arange(30, dtype=float),
                               qvel=np.arange(28, dtype=float) * 10,
                               ctrl=np.zeros(16))
        q, qd = mapping.state(data)
        for i, name in enumerate(meta["joint_names"]):
            mj_name = name.replace("_foot_joint", "_wheel_joint")
            jid = next(j for j in range(model.njnt) if mj.mj_id2name(model, 3, j) == mj_name)
            self.assertEqual(q[i], data.qpos[model.jnt_qposadr[jid]])
            self.assertEqual(qd[i], data.qvel[model.jnt_dofadr[jid]])
        torque = np.arange(16, dtype=float) + 1
        mapping.write_torques(data, torque)
        for i, jid in enumerate(mapping.joint_ids):
            aid = int(np.flatnonzero(model.actuator_trnid[:, 0] == jid)[0])
            self.assertAlmostEqual(data.ctrl[aid] * model.actuator_gear[aid, 0], torque[i])
        self.assertFalse(np.array_equal(mapping.actuator_ids, np.arange(16)))

    def test_missing_and_colliding_names_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            resolve_joint_names(["FL_foot_joint"], ["FR_wheel_joint"])
        with self.assertRaisesRegex(ValueError, "Multiple"):
            resolve_joint_names(["FL_foot_joint", "FL_wheel_joint"], ["FL_wheel_joint"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            resolve_joint_names(["FL_foot_joint"] * 2, ["FL_wheel_joint"])

    def test_rejects_servo_instead_of_double_applying_pd(self):
        model, mj = fake_model()
        model.actuator_biastype[0] = 1
        with self.assertRaisesRegex(ValueError, "plain torque motor"):
            JointMapping.from_model(model, metadata()["joint_names"], mj)

    def test_frame_uses_training_scales_and_does_not_zero_real_wheel_positions(self):
        meta = metadata()
        q = np.array(meta["default_dof_pos"]) + np.arange(16) / 10
        q[12:] = [10, 20, 30, 40]
        original_q = q.copy()
        previous_action = np.arange(16) / 20
        frame = build_frame([4, 8, 12], [0, 0, -1], [0.5, -1, 2], q,
                            np.ones(16) * 2, previous_action, 2, meta)
        np.testing.assert_array_equal(q, original_q)
        np.testing.assert_allclose(frame[:3], [1, 2, 3])
        np.testing.assert_allclose(frame[3:6], [0, 0, -1])
        np.testing.assert_allclose(frame[6:9], [1, -2, 0.5])
        np.testing.assert_allclose(frame[9:21], np.arange(12) / 10)
        np.testing.assert_array_equal(frame[21:25], np.zeros(4))
        np.testing.assert_allclose(frame[25:41], np.ones(16) * 0.1)
        np.testing.assert_allclose(frame[41:57], previous_action)
        self.assertEqual(frame[57], 2)

    def test_wheel_velocity_does_not_receive_leg_action_scale(self):
        meta = metadata()
        action = np.ones(16)
        q = np.array(meta["default_dof_pos"])
        q[12:] = 2000  # Wheel angle is irrelevant to the velocity controller.
        qd = np.ones(16) * 2
        torque = pd_torque(action, q, qd, meta)
        np.testing.assert_allclose(torque[:12], 40 * 0.25 - 2)
        np.testing.assert_allclose(torque[12:], 0.5 * (10 - 2))
        # The action stored in the next observation is clipped before use.
        meta["clip_actions"] = 1
        np.testing.assert_allclose(pd_torque(action * 200, q, qd, meta), torque)

    def test_metadata_refuses_an_inexact_policy_period(self):
        meta = metadata()
        validate_metadata(meta)
        meta["control_dt"] = 0.023
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            validate_metadata(meta)

    def test_reset_history_repeats_initial_frame_and_preserves_actual_past_skills(self):
        history = np.zeros((6, 58), dtype=np.float32)
        initial = np.arange(58, dtype=np.float32)
        initial[-1] = 2
        update_history(history, initial, reset=True)
        np.testing.assert_array_equal(history, np.tile(initial, (6, 1)))
        next_frame = initial + 3
        next_frame[-1] = 0
        update_history(history, next_frame)
        np.testing.assert_array_equal(history[0], next_frame)
        np.testing.assert_array_equal(history[1:], np.tile(initial, (5, 1)))

    def test_calibrated_peak_can_exceed_urdf_without_premature_clipping(self):
        meta = metadata()
        meta["torque_limits"] = [5.] * 16
        meta["motor"] = {"enabled": False, "joints": {"FL_hip_joint": {"peak_torque": 15.}}}
        torque = pd_torque(np.ones(16), np.array(meta["default_dof_pos"]), np.zeros(16), meta)
        self.assertEqual(torque[0], 10.)
        self.assertEqual(effective_torque_limits(meta)[0], 15.)
        root = ET.fromstring(default_scene_xml(meta))
        actuator = root.find("./actuator/motor[@joint='FL_hip_joint']")
        np.testing.assert_array_equal(np.fromstring(actuator.get("ctrlrange"), sep=" "), [-15, 15])

    def test_default_scene_has_portable_assets_and_exported_torque_limits(self):
        meta = metadata()
        root = ET.fromstring(default_scene_xml(meta))
        meshdir = Path(root.find("compiler").get("meshdir"))
        self.assertTrue(meshdir.is_absolute())
        for asset in root.findall("./asset/mesh"):
            self.assertTrue((meshdir / asset.get("file")).is_file())
        self.assertEqual(len(root.findall(".//hfield")), 0)
        self.assertIsNotNone(root.find("./worldbody/geom[@name='mujica_flat_floor']"))
        for motor in root.findall("./actuator/motor"):
            np.testing.assert_array_equal(np.fromstring(motor.get("ctrlrange"), sep=" "), [-100, 100])


if __name__ == "__main__":
    unittest.main()
