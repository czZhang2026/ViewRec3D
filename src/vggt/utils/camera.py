import math

import torch

from src.vggt.utils.rotation import quat_to_mat


def build_intrinsic(fov_x, fov_y, w: int, h: int):
    """
    Args:
        fov_x: [B, 1] 水平视场角（弧度）
        fov_y: [B, 1] 垂直视场角（弧度）
        w: int
        h: int

    Returns:
        intrinsic: [B, 3, 3]
    """
    # reshape 成 [B]
    fov_x = torch.as_tensor(fov_x).reshape(-1)
    fov_y = torch.as_tensor(fov_y).reshape(-1)

    B = fov_x.shape[0]
    device = fov_x.device
    dtype = fov_x.dtype

    cx = w / 2.0
    cy = h / 2.0

    fx = cx / torch.tan(fov_x / 2.0)   # [B]
    fy = cy / torch.tan(fov_y / 2.0)   # [B]

    f = torch.max(fx, fy) # padding导致intrinsic改变

    intrinsic = torch.zeros((B, 3, 3), device=device, dtype=dtype)

    intrinsic[:, 0, 0] = f
    intrinsic[:, 1, 1] = f
    intrinsic[:, 0, 2] = cx
    intrinsic[:, 1, 2] = cy
    intrinsic[:, 2, 2] = 1.0

    return intrinsic

def build_extrinsic(trans, quat):
    """
    Args:
        trans: [B, 3]
        quat:  [B, 4]  (x, y, z, w)

    Returns:
        c2w: [B, 4, 4]
    """
    B = trans.shape[0]
    device = trans.device
    dtype = trans.dtype

    t = trans                  # [B, 3]
    R = quat_to_mat(quat)      # [B, 3, 3]

    c2w = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    c2w[:, :3, :3] = R
    c2w[:, :3, 3] = t

    return c2w


