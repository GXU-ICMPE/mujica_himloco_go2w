"""Isaac Gym wrapper for the shared terrain-locomotion heightfield."""
from isaacgym import terrain_utils
from mujica.terrain import HeightField


class MUJICATerrain(HeightField):
    def __init__(self, cfg, num_robots):
        super().__init__(cfg)
        self.num_robots, self.type = num_robots, cfg.mesh_type
        self.env_length, self.env_width = cfg.terrain_length, cfg.terrain_width
        self.length_per_env_pixels, self.width_per_env_pixels = self.nx, self.ny
        self.tot_rows, self.tot_cols = self.height_field_raw.shape
        self.heightsamples = self.height_field_raw
        cfg.num_sub_terrains = cfg.num_rows*cfg.num_cols
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_treshold)
