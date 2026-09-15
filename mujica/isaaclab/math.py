"""Small torch-only quaternion helpers; all quaternions are wxyz."""
import torch


def torch_rand_float(lo, hi, shape, device):
    return torch.rand(shape, device=device) * (hi - lo) + lo


def quat_mul(a, b):
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    xyz = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    return torch.cat((w, xyz), -1)


def quat_apply(q, v):
    # Broadcast the vector part explicitly for [N, 1, 4] x [N, K, 3].
    xyz, vector = torch.broadcast_tensors(q[..., 1:], v)
    t = 2 * torch.cross(xyz, vector, dim=-1)
    return vector + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse(q, v):
    return quat_apply(torch.cat((q[..., :1], -q[..., 1:]), -1), v)


def quat_from_angle_axis(angle, axis):
    half = angle.unsqueeze(-1) * 0.5
    return torch.cat((half.cos(), half.sin() * axis), -1)


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = (roll / 2).cos(), (roll / 2).sin()
    cp, sp = (pitch / 2).cos(), (pitch / 2).sin()
    cy, sy = (yaw / 2).cos(), (yaw / 2).sin()
    return torch.stack((cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                        cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy), -1)


def yaw_rotate(q, points):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    px, py, pz = points.unbind(-1)
    return torch.stack((c*px - s*py, s*px + c*py, pz.expand_as(c*px)), -1)
