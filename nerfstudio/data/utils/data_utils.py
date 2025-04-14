# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions to allow easy re-use of common operations across dataloaders"""

from pathlib import Path
from typing import IO, List, Tuple, Union

import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.Image import Image as PILImage
from jaxtyping import Float


def pil_to_numpy(im: PILImage) -> np.ndarray:
    """Converts a PIL Image object to a NumPy array.

    Args:
        im (PIL.Image.Image): The input PIL Image object.

    Returns:
        numpy.ndarray representing the image data.
    """
    # Load in image completely (PIL defaults to lazy loading)
    im.load()

    # Unpack data
    e = Image._getencoder(im.mode, "raw", im.mode)
    e.setimage(im.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        _, s, d = e.encode(bufsize)
        mem[offset : offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)

    return data


def get_image_mask_tensor_from_path(filepath: Union[Path, IO[bytes]], scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    # load mask
    if isinstance(filepath, io.BytesIO):
        pil_mask = Image.open(filepath).convert('L')
    elif filepath.suffix == ".npy":
        mask = np.load(filepath)  # (H, W)
        pil_mask = Image.fromarray(mask, 'L')
    else:
        pil_mask = Image.open(filepath).convert('L')

    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.Resampling.NEAREST)
    mask_tensor = torch.from_numpy(pil_to_numpy(pil_mask)).unsqueeze(-1).bool()
    if len(mask_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype=torch.int64).view(1, 1, -1)
    if filepath.suffix == ".npz":
        segmask = np.load(filepath, allow_pickle=True)["arr_0"]
        if scale_factor != 1.0:
            width, height = segmask.shape
            newsize = (int(width * scale_factor), int(height * scale_factor))
            segmask = cv2.resize(segmask, newsize, interpolation=cv2.INTER_NEAREST)
        semantics = torch.from_numpy(segmask.astype(np.int64))[..., None]
    else:
        pil_image = Image.open(filepath)
        if scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * scale_factor), int(height * scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.Resampling.NEAREST)
        semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath).astype(np.float32) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float32) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)

    # if confidence available
    if filepath.with_suffix(".conf.npy").exists():
        conf = np.load(filepath.with_suffix(".conf.npy"))
        conf = np.clip(conf, 0, 1)
        return torch.from_numpy(np.stack([image, conf], axis=-1))  # [H, W, 2]
    return torch.from_numpy(image[:, :, np.newaxis])

def get_normal_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    camera_to_world: Float[torch.Tensor, "3 4"],
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a numpy array `*.npy` of floats in the range of [0., 1.].

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        camera_to_world: Camera to world transformation matrix.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [width, height, 1].
    """

    normal = np.load(filepath)
    c, h, w = normal.shape
    if (h, w) != (height, width):
        normal = cv2.resize(normal, (width, height), interpolation=interpolation)
    #
    # # important as the output of omnidata is normalized
    normal = normal * 2.0 - 1.0
    normal = torch.from_numpy(normal).float()
    normal_tr = torch.tensor([[1, 0, 0],
                              [0, -1, 0],
                              [0, 0, -1]], device=normal.device).float()
    #
    # # transform normal to world coordinate system
    # rot = camera_to_world[:3, :3].clone()
    #
    normal_map = normal.reshape(3, -1)
    normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)
    #
    # normal_map = normal_tr @ normal_map
    # normal_map = rot @ normal_tr @ normal_map
    normal_map = normal_map.permute(1, 0).reshape(h, w, 3)

    # if confidence available
    if filepath.with_suffix(".conf.npy").exists():
        conf = np.load(filepath.with_suffix(".conf.npy"))
        conf = np.clip(conf, 0, 1)
        return torch.from_numpy(np.concatenate([normal_map, conf[..., None]], axis=-1))  # [H, W, 3+1]
    return normal_map

"""Code from DRGSplat: Depth-Regularized 3D Gaussian Splatting"""
def gaussian_1d(kernel_size, sigma, derivative=False):
    """
    Create a 1D Gaussian or Gaussian-derivative kernel vector.

    Args:
        kernel_size (int): Size of the kernel (odd preferred).
        sigma (float): Standard deviation of the Gaussian.
        derivative (bool): If True, create derivative-of-Gaussian.
                        If False, create standard Gaussian.

    Returns:
        torch.Tensor of shape (kernel_size,) containing the kernel.
    """
    # Coordinate grid centered at 0
    center = (kernel_size - 1) / 2
    x = torch.arange(kernel_size) - center

    # Compute standard Gaussian
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss = gauss / gauss.sum()  # normalize

    if not derivative:
        return gauss

    # Gaussian derivative (in 1D)
    # d/dx of gaussian = -x/sigma^2 * gauss
    # We'll normalize so it sums to 0, but no final normalization on amplitude
    gauss_deriv = -x / (sigma ** 2) * gauss
    # Typically we ensure sum of positive side = -sum of negative side
    # so the integral is 0. The above formula already does that inherently.
    return gauss_deriv


def make_gaussian_deriv_kernels(kernel_size=5, sigma=1.0):
    """
    Create 2D kernels for derivative in X and derivative in Y,
    with optional Gaussian weighting in the orthogonal direction.

    Returns:
        kx, ky (each shape (1,1,kernel_size,kernel_size))
        - kx: derivative in X, smooth in Y
        - ky: derivative in Y, smooth in X
    """
    # 1D kernels
    g = gaussian_1d(kernel_size, sigma, derivative=False)  # e.g. [g(x)]
    dg = gaussian_1d(kernel_size, sigma, derivative=True)  # e.g. [g'(x)]

    # Convert to 2D outer products:
    # kx(x,y) = dg(x) *  g(y)
    # ky(x,y) =  g(x) * dg(y)
    g = g.view(1, -1)  # shape (1, kernel_size)
    dg = dg.view(1, -1)  # shape (1, kernel_size)

    kx_2d = dg.t() @ g  # (kernel_size, kernel_size)
    ky_2d = g.t() @ dg  # (kernel_size, kernel_size)

    # Reshape to (1,1,kH,kW) => so we can conv2d with 'groups=1'
    kx_2d = kx_2d.unsqueeze(0).unsqueeze(0)
    ky_2d = ky_2d.unsqueeze(0).unsqueeze(0)

    return kx_2d, ky_2d

def compute_normals_finite_diff(
        depths,
        Ks,
        kernel_size=5,
        sigma=1.0
):
    """
    Computes normals and a validity mask.  Normals at and around zero-depth
    values are set to (0,0,0), and the mask indicates valid normals.

    Args:
        depths: (B, H, W) tensor of depth values.
        Ks:     (B, 3, 3) intrinsics.
        kernel_size: Kernel size for derivative calculation.
        sigma: Standard deviation for the Gaussian derivative.

    Returns:
        normals: (B, H, W, 3) unit normals.
        valid_mask: (B, H, W) boolean mask, True where normals are valid.
    """
    device = depths.device
    B, H, W = depths.shape

    # --- 1. Create Initial Validity Mask ---
    valid_mask = depths != 0

    # --- 2. Build Meshgrid ---
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid_y = yy.unsqueeze(0).expand(B, -1, -1)
    grid_x = xx.unsqueeze(0).expand(B, -1, -1)

    # --- 3. Intrinsics ---
    fx = Ks[:, 0, 0].view(B, 1, 1)
    fy = Ks[:, 1, 1].view(B, 1, 1)
    cx = Ks[:, 0, 2].view(B, 1, 1)
    cy = Ks[:, 1, 2].view(B, 1, 1)

    # --- 4. 3D Points ---
    X = (grid_x - cx) * depths / fx
    Y = (grid_y - cy) * depths / fy
    Z = depths
    points_3d = torch.stack([X, Y, Z], dim=1)  # (B, 3, H, W)

    # --- 5. Derivative Kernels ---
    kx_2d, ky_2d = make_gaussian_deriv_kernels(kernel_size, sigma)
    kx_2d = kx_2d.to(device)
    ky_2d = ky_2d.to(device)
    kx_2d_3ch = kx_2d.repeat(3, 1, 1, 1)
    ky_2d_3ch = ky_2d.repeat(3, 1, 1, 1)

    # --- 6. Convolution (Derivatives) ---
    p_x = F.conv2d(points_3d, kx_2d_3ch, padding=kernel_size // 2, groups=3)
    p_y = F.conv2d(points_3d, ky_2d_3ch, padding=kernel_size // 2, groups=3)

    # --- 7. Masking AFTER convolution, BEFORE cross product ---
    expanded_mask = valid_mask.unsqueeze(1)  # (B, 1, H, W)
    p_x = torch.where(expanded_mask, p_x, torch.zeros_like(p_x))
    p_y = torch.where(expanded_mask, p_y, torch.zeros_like(p_y))

    # --- 8. Cross Product ---
    p_x_bhw3 = p_x.permute(0, 2, 3, 1)  # (B, 3, H, W) -> (B, H, W, 3)
    p_y_bhw3 = p_y.permute(0, 2, 3, 1)
    normals_bhw3 = torch.cross(p_x_bhw3, p_y_bhw3, dim=-1)

    # --- 9. Normalize ---
    norm_mag = torch.linalg.norm(normals_bhw3, dim=-1, keepdim=True)
    normals_bhw3 = normals_bhw3 / (norm_mag + 1e-8)

    # --- 10. Dilate the invalid region and create final mask---
    # Use a binary dilation to expand the invalid region. The kernel size
    # should match the derivative kernel size.
    dilation_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
    expanded_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W) for conv2d
    dilated_mask = (F.conv2d(expanded_mask, dilation_kernel, padding='same', groups=1) > 0).squeeze(
        1)  # Back to (B, H, W)

    # --- 11. Apply final mask ---
    normals_bhw3 = torch.where(dilated_mask.unsqueeze(-1), normals_bhw3, torch.zeros_like(normals_bhw3))

    return normals_bhw3, dilated_mask  # Return normals AND mask

def identity_collate(x):
    """This function does nothing but serves to help our dataloaders have a pickleable function, as lambdas are not pickleable"""
    return x
