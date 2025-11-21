import numpy as np
import math
import torch
from scene.cameras import Camera
from scipy.spatial.transform import Rotation


def generate_new_cam(r_d, tx, resolution=180):

    
    # rot = Rotation.from_rotvec(r_d).as_matrix()
    rot = r_d
    
    trans = tx


    fovx = np.deg2rad(180)
    fovy = np.deg2rad(180)

    cam = Camera(R=rot,colmap_id=None, T=trans, FoVx=fovx, FoVy=fovy, image=None, image_name=None, uid=None,invdepthmap=None,depth_params=None )
    cam.image_width=resolution*2
    cam.image_height=90
    
    return cam

def generate_new_cam_torch(r_d, tx, resolution=180):
    if not isinstance(r_d, torch.Tensor):
        rot = torch.as_tensor(r_d, dtype=torch.float32, device=tx.device if isinstance(tx, torch.Tensor) else 'cuda')
    else:
        rot = r_d.to(dtype=torch.float32)

    if not isinstance(tx, torch.Tensor):
        trans_vec = torch.as_tensor(tx, dtype=torch.float32, device=rot.device)
    else:
        trans_vec = tx.to(dtype=torch.float32)

    fovx = float(torch.deg2rad(torch.tensor(180.0)))
    fovy = float(torch.deg2rad(torch.tensor(180.0)))

    cam = Camera(R=rot, colmap_id=None, T=trans_vec, FoVx=fovx, FoVy=fovy,
                 image=None, image_name=None, uid=None, invdepthmap=None, depth_params=None)
    cam.image_width  = int(resolution * 2)
    cam.image_height = 90
    return cam

def quat_xyzw_to_rotmat_torch(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    assert q.shape[-1] == 4
    x, y, z, w = q.unbind(-1)
    norm = torch.clamp(x*x + y*y + z*z + w*w, min=eps).sqrt()
    x, y, z, w = x/norm, y/norm, z/norm, w/norm

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    m00 = 1 - 2*(yy + zz)
    m01 = 2*(xy - wz)
    m02 = 2*(xz + wy)

    m10 = 2*(xy + wz)
    m11 = 1 - 2*(xx + zz)
    m12 = 2*(yz - wx)

    m20 = 2*(xz - wy)
    m21 = 2*(yz + wx)
    m22 = 1 - 2*(xx + yy)

    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
