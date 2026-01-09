import json
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from einops import repeat
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .utils_omniscene import load_conditions, load_info
from .view_sampler import ViewSampler


CAMERA_TYPES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


@dataclass
class DatasetOmniSceneCfg(DatasetCfgCommon):
    name: Literal["omniscene"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = False
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False


class DatasetOmniScene(Dataset):
    data_version: str = "interp_12Hz_trainval"
    dataset_prefix: str = "/datasets/nuScenes"

    def __init__(
        self,
        cfg: DatasetOmniSceneCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        load_rel_depth: bool | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.load_rel_depth = (
            stage == "test" if load_rel_depth is None else load_rel_depth
        )
        if stage != "test":
            self.load_rel_depth = False
        self.data_root = str(cfg.roots[0])
        self.resolution = (cfg.image_shape[0], cfg.image_shape[1])
        self.near = cfg.near if cfg.near != -1 else 0.5
        self.far = cfg.far if cfg.far != -1 else 100.0

        self.bin_tokens = self._load_bin_tokens()
        if stage == "train" and cfg.train_times_per_scene > 1:
            self.bin_tokens = self.bin_tokens * cfg.train_times_per_scene
        if cfg.test_len > 0 and stage == "test":
            self.bin_tokens = self.bin_tokens[: cfg.test_len]

    def _load_bin_tokens(self) -> list[str]:
        root = Path(self.data_root) / self.data_version
        if self.stage == "train":
            path = root / "bins_train_3.2m.json"
            tokens = json.load(path.open("r"))["bins"]
        else:
            path = root / "bins_val_3.2m.json"
            tokens = json.load(path.open("r"))["bins"]
            if self.stage == "val":
                tokens = tokens[:30000:3000][:10]
            elif self.stage == "test":
                tokens = tokens[0::14][:2048]
        return tokens

    def __len__(self) -> int:
        return len(self.bin_tokens)

    def __getitem__(self, index: int):
        bin_token = self.bin_tokens[index]
        bin_path = (
            Path(self.data_root)
            / self.data_version
            / "bin_infos_3.2m"
            / f"{bin_token}.pkl"
        )
        with bin_path.open("rb") as f:
            bin_info = pkl.load(f)

        sensor_info_center = {
            sensor: bin_info["sensor_info"][sensor][0] for sensor in CAMERA_TYPES + ["LIDAR_TOP"]
        }

        input_img_paths, input_c2ws = [], []
        for cam in CAMERA_TYPES:
            info = sensor_info_center[cam]
            img_path, c2w, _ = load_info(info, self.dataset_prefix, self.data_root)
            input_img_paths.append(img_path)
            input_c2ws.append(c2w)
        input_c2ws = torch.stack(input_c2ws)

        input_imgs, input_masks, input_ks, input_rel_depths = load_conditions(
            input_img_paths,
            self.resolution,
            is_input=True,
            load_rel_depth=self.load_rel_depth,
            highres=self.cfg.highres,
        )

        output_img_paths, output_c2ws = [], []
        frame_num = len(bin_info["sensor_info"]["LIDAR_TOP"])
        if frame_num < 3:
            raise ValueError(f"bin {bin_token} has insufficient frames ({frame_num})")
        for cam in CAMERA_TYPES:
            indices = [1, 2]
            for idx in indices:
                info = bin_info["sensor_info"][cam][idx]
                img_path, c2w, _ = load_info(info, self.dataset_prefix, self.data_root)
                output_img_paths.append(img_path)
                output_c2ws.append(c2w)
        output_c2ws = torch.stack(output_c2ws)

        output_imgs, output_masks, output_ks, output_rel_depths = load_conditions(
            output_img_paths,
            self.resolution,
            is_input=False,
            load_rel_depth=self.load_rel_depth,
            highres=self.cfg.highres,
        )

        output_imgs = torch.cat([output_imgs, input_imgs], dim=0)
        output_masks = torch.cat([output_masks, input_masks], dim=0)
        output_c2ws = torch.cat([output_c2ws, input_c2ws], dim=0)
        output_ks = torch.cat([output_ks, input_ks], dim=0)
        if output_rel_depths is not None and input_rel_depths is not None:
            output_rel_depths = torch.cat(
                [output_rel_depths, input_rel_depths], dim=0
            )

        context = {
            "extrinsics": input_c2ws,
            "intrinsics": input_ks,
            "image": input_imgs,
            "near": repeat(torch.tensor(self.near, dtype=torch.float32), "-> v", v=input_c2ws.shape[0]),
            "far": repeat(torch.tensor(self.far, dtype=torch.float32), "-> v", v=input_c2ws.shape[0]),
            "index": torch.arange(input_c2ws.shape[0], dtype=torch.int64),
        }
        target = {
            "extrinsics": output_c2ws,
            "intrinsics": output_ks,
            "image": output_imgs,
            "near": repeat(torch.tensor(self.near, dtype=torch.float32), "-> v", v=output_c2ws.shape[0]),
            "far": repeat(torch.tensor(self.far, dtype=torch.float32), "-> v", v=output_c2ws.shape[0]),
            "index": torch.arange(output_c2ws.shape[0], dtype=torch.int64),
            "masks": output_masks,
        }
        if output_rel_depths is not None:
            target["rel_depth"] = output_rel_depths
        return {
            "context": context,
            "target": target,
            "scene": bin_token,
        }
