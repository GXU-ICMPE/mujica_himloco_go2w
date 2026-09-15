"""Translate native Gymnasium returns to the existing MUJICA runner contract."""


class MUJICAVecEnv:
    def __init__(self, env):
        self.env = env
        self.num_envs, self.device = env.num_envs, env.device
        self._mujica_config_snapshot = dict(env.settings)

    def reset(self):
        obs, _ = self.env.reset()
        return obs["policy"], obs["critic"]

    def get_observations(self):
        return self.env.history_buf

    def get_privileged_observations(self):
        return self.env.privileged_obs_buf

    def set_skill(self, ids):
        self.env.set_skill(ids)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return (obs["policy"], obs["critic"], reward, terminated | truncated, info,
                info["terminal_ids"], info["terminal_critic"])

    def export_metadata(self):
        return self.env.export_metadata()

    def close(self):
        self.env.close()
