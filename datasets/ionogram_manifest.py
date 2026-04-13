import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class IonogramManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        image_mode: str = "L",   # "L" or "RGB"
        image_size: int = 448,
        normalize_to_01: bool = True,
        local_crop: tuple[int, int] | None = (186, 410),
    ):
        self.manifest_path = Path(manifest_path)
        self.image_mode = image_mode
        self.image_size = image_size
        self.normalize_to_01 = normalize_to_01
        self.local_crop = local_crop

        self.samples: List[Dict] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                item = json.loads(line)

                # 优先使用显式的 input_paths / target_paths
                if "input_paths" in item and "target_paths" in item:
                    input_paths = item["input_paths"]
                    target_paths = item["target_paths"]
                # 兼容只有 image_paths 的情况：前8张输入，后2张目标
                elif "image_paths" in item:
                    assert len(item["image_paths"]) == 10, \
                        f"{self.manifest_path} 第 {line_idx} 行 image_paths 不是10张"
                    input_paths = item["image_paths"][:8]
                    target_paths = item["image_paths"][8:10]
                else:
                    raise ValueError(
                        f"{self.manifest_path} 第 {line_idx} 行缺少 input_paths/target_paths 或 image_paths"
                    )

                assert len(input_paths) == 8, \
                    f"{self.manifest_path} 第 {line_idx} 行 input_paths 数量不是8"
                assert len(target_paths) == 2, \
                    f"{self.manifest_path} 第 {line_idx} 行 target_paths 数量不是2"

                item["input_paths"] = input_paths
                item["target_paths"] = target_paths
                self.samples.append(item)

        self.channels = 1 if self.image_mode == "L" else 3

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert(self.image_mode)

        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32)

        if self.image_mode == "L":
            arr = arr[..., None]  # H,W -> H,W,1

        if self.normalize_to_01:
            arr = arr / 255.0

        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(arr).float()

    def __getitem__(self, idx):
        item = self.samples[idx]

        x = torch.stack([self._load_image(p) for p in item["input_paths"]], dim=0)   # [8, C, H, W]
        y = torch.stack([self._load_image(p) for p in item["target_paths"]], dim=0)  # [2, C, H, W]

        sample = {
            "x": x,
            "y": y,
            "label": item.get("label", None),
            "sequence_id": item.get("sequence_id", str(idx)),
            "year": item.get("year", None),
            "source": item.get("source", None),
            "timestamps": item.get("timestamps", None),
            "split": item.get("split", None),
        }

        if self.local_crop is not None:
            sample["x_local"] = self.crop_f_region(x)
            sample["y_local"] = self.crop_f_region(y)

        return sample
    
    def crop_f_region(self, seq: torch.Tensor) -> torch.Tensor:
        # Fixed F-region crop boundary in the 448x448 image coordinate system.
        # local_crop=(186, 410) produces a 224x448 local sequence.
        top, bottom = self.local_crop
        return seq[:, :, top:bottom, :]
