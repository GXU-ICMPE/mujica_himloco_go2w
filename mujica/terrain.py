"""Flat/slopes, irregular discrete obstacles and regular stairs for both simulators."""
import numpy as np
from .skills import TERRAIN_GROUPS


def assign_terrain_columns(num_envs, task_ids):
    """Equal task allocation despite 3/1/2 terrain variants per task."""
    task_ids = np.asarray(task_ids)
    result = np.empty(num_envs, dtype=np.int64)
    for skill in range(3):
        envs = np.arange(skill, num_envs, 3)
        columns = np.flatnonzero(task_ids == skill)
        if len(columns) == 0:
            raise ValueError("Every task must have terrain columns")
        result[envs] = columns[np.arange(len(envs)) % len(columns)]
    return result


def validate_terrain(cfg):
    if cfg.num_rows < 1 or cfg.num_cols < 6 or cfg.num_cols % 6:
        raise ValueError("Terrain requires positive levels and a column count divisible by six (at least six)")
    if min(cfg.horizontal_scale, cfg.vertical_scale) <= 0 or cfg.border_size < 0:
        raise ValueError("Terrain scales must be positive and border_size non-negative")
    if not 1.5 <= cfg.platform_size < min(cfg.terrain_length, cfg.terrain_width)-1.0:
        raise ValueError("Terrain spawn platform must be at least 1.5m wide with room for terrain outside it")
    if cfg.stair_width < cfg.horizontal_scale:
        raise ValueError("stair_width must span at least one heightfield pixel")
    for name in ("slope_range", "stair_height_range", "discrete_size_range", "discrete_height_range"):
        bounds = getattr(cfg, name)
        if len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 <= bounds[0] <= bounds[1]:
            raise ValueError(name + " must contain two finite non-negative ordered bounds")
    if cfg.discrete_size_range[0] < cfg.horizontal_scale or cfg.discrete_size_range[1] >= min(cfg.terrain_length, cfg.terrain_width):
        raise ValueError("Discrete obstacle sizes must fit inside a terrain tile")
    if cfg.discrete_num_obstacles < 1:
        raise ValueError("At least one discrete obstacle is required")


class HeightField:
    def __init__(self, cfg):
        validate_terrain(cfg)
        self.cfg = cfg
        self.nx = round(cfg.terrain_length / cfg.horizontal_scale)
        self.ny = round(cfg.terrain_width / cfg.horizontal_scale)
        self.border = round(cfg.border_size / cfg.horizontal_scale)
        self.height_field_raw = np.zeros((cfg.num_rows*self.nx + 2*self.border,
                                          cfg.num_cols*self.ny + 2*self.border), dtype=np.int16)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)
        layout = [(skill, name) for skill, group in enumerate(TERRAIN_GROUPS) for name in group]
        self.task_ids = np.array([layout[col % len(layout)][0] for col in range(cfg.num_cols)])
        self.terrain_names = [layout[col % len(layout)][1] for col in range(cfg.num_cols)]
        for col in range(cfg.num_cols):
            for row in range(cfg.num_rows):
                tile, _ = self.tile(row, col)
                x, y = self.border + row*self.nx, self.border + col*self.ny
                self.height_field_raw[x:x+self.nx, y:y+self.ny] = tile
                z = tile[self.nx//2-2:self.nx//2+3, self.ny//2-2:self.ny//2+3].max()*cfg.vertical_scale
                self.env_origins[row, col] = ((row+0.5)*cfg.terrain_length, (col+0.5)*cfg.terrain_width, z)

    def tile(self, row, col):
        cfg = self.cfg
        hs, vs = cfg.horizontal_scale, cfg.vertical_scale
        fraction = min(row + 1, 20) / 20
        name = self.terrain_names[col]
        tile = np.zeros((self.nx, self.ny), dtype=np.int16)
        cx, cy = self.nx//2, self.ny//2
        if name == "flat":
            return tile, name
        if name == "discrete":
            for _ in range(cfg.discrete_num_obstacles):
                sx, sy = np.rint(np.random.uniform(*cfg.discrete_size_range, 2)/hs).astype(int)
                x, y = np.random.randint(self.nx-sx+1), np.random.randint(self.ny-sy+1)
                lo, hi = cfg.discrete_height_range
                # Independent obstacle heights and sizes produce irregular
                # foothold timing; stair tiles keep a uniform tread/riser.
                tile[x:x+sx, y:y+sy] = round(np.random.uniform(lo, lo+(hi-lo)*fraction)/vs)
            safe = round(cfg.platform_size/2/hs)
            tile[cx-safe:cx+safe+1, cy-safe:cy+safe+1] = 0
            return tile, name
        # Square radial surfaces make ascending/descending meaningful for
        # forward, backward and lateral commands from the central spawn patch.
        radius = np.maximum(np.abs(np.arange(self.nx)-cx)[:, None], np.abs(np.arange(self.ny)-cy)[None, :])*hs
        travel = np.maximum(radius-cfg.platform_size/2, 0)
        if name.startswith("slope_"):
            lo, hi = cfg.slope_range
            height = travel * (lo+(hi-lo)*fraction)
        else:
            lo, hi = cfg.stair_height_range
            # Round width in pixels, avoiding int(0.3/0.1) silently becoming 2.
            width = round(cfg.stair_width/hs)*hs
            height = np.ceil(np.maximum(travel/width-1e-8, 0)) * (lo+(hi-lo)*fraction)
        if name.endswith("down"):
            height = -height
        return np.rint(height/vs).astype(np.int16), name

    def mesh(self):
        import trimesh
        from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
        vertices, triangles = convert_height_field_to_mesh(self.height_field_raw,
            self.cfg.horizontal_scale, self.cfg.vertical_scale, self.cfg.slope_treshold)
        vertices[:, :2] -= self.cfg.border_size
        return trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
