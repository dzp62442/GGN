import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch


def _replace_segment(path: str, old: str, new: str) -> str:
    if old not in path:
        return path
    return path.replace(old, new)


def HWC3(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 1:
        return np.concatenate([image] * 3, axis=2)
    if image.shape[2] == 4:
        color = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        blended = color * alpha + 255.0 * (1.0 - alpha)
        return blended.clip(0, 255).astype(np.uint8)
    raise ValueError("Unsupported channel count for HWC3 conversion")


def load_info(info: dict, dataset_prefix: str, data_root: str) -> tuple[str, torch.Tensor, torch.Tensor]:
    img_path = info["data_path"].replace(dataset_prefix, data_root)
    c2w = torch.tensor(info["sensor2lidar_transform"], dtype=torch.float32)

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t
    w2c = torch.tensor(w2c, dtype=torch.float32)

    return img_path, c2w, w2c


def load_conditions(
    img_paths: Sequence[str],
    resolution: tuple[int, int],
    *,
    is_input: bool,
    load_rel_depth: bool = False,
    highres: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del load_rel_depth
    h_target, w_target = resolution

    def maybe_resize(pil_img: Image.Image, intrinsics: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        need_resize = pil_img.height != h_target or pil_img.width != w_target
        if not need_resize:
            return pil_img, intrinsics
        scale_h = h_target / pil_img.height
        scale_w = w_target / pil_img.width
        intrinsics = intrinsics.copy()
        intrinsics[0, 0] *= scale_w
        intrinsics[1, 1] *= scale_h
        intrinsics[0, 2] *= scale_w
        intrinsics[1, 2] *= scale_h
        pil_img = pil_img.resize((w_target, h_target), Image.BILINEAR)
        return pil_img, intrinsics

    images: list[torch.Tensor] = []
    intrinsics: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for path in img_paths:
        param_path = _replace_segment(path, "samples", "samples_param_small")
        param_path = _replace_segment(param_path, "sweeps", "sweeps_param_small")
        if highres:
            param_path = param_path.replace("_small", "")
        with Path(param_path).open("r") as f:
            intrinsic = np.array(json.load(f)["camera_intrinsic"], dtype=np.float32)

        img_path = _replace_segment(path, "samples", "samples_small")
        img_path = _replace_segment(img_path, "sweeps", "sweeps_small")
        if highres:
            img_path = img_path.replace("_small", "")
        image = Image.open(img_path)
        image, intrinsic = maybe_resize(image, intrinsic)

        intrinsic = intrinsic.copy()
        intrinsic[0, :] /= w_target
        intrinsic[1, :] /= h_target

        image = HWC3(np.array(image))
        images.append(torch.from_numpy(image).permute(2, 0, 1).float() / 255.0)
        intrinsics.append(torch.from_numpy(intrinsic))

        if is_input:
            mask = torch.ones((h_target, w_target), dtype=torch.bool)
        else:
            mask_path = _replace_segment(path, "samples", "samples_mask_small")
            mask_path = _replace_segment(mask_path, "sweeps", "sweeps_mask_small")
            if highres:
                mask_path = mask_path.replace("_small", "")
            mask_img = Image.open(mask_path).convert("L")
            if mask_img.height != h_target or mask_img.width != w_target:
                mask_img = mask_img.resize((w_target, h_target), Image.BILINEAR)
            mask = torch.from_numpy(np.array(mask_img) / 255.0 >= 0.5)
        masks.append(mask)

    images_tensor = torch.stack(images, dim=0)
    intrinsics_tensor = torch.stack(intrinsics, dim=0)
    masks_tensor = torch.stack(masks, dim=0)
    return images_tensor, masks_tensor, intrinsics_tensor
