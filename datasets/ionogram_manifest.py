import json
import os
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


@contextmanager
def _image_load_timeout(timeout_sec: float):
    if timeout_sec is None or float(timeout_sec) <= 0 or os.name == "nt" or not hasattr(signal, "SIGALRM"):
        yield
        return

    timeout_sec = float(timeout_sec)

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"Image load exceeded timeout of {timeout_sec:.1f}s.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


class IonogramManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        image_mode: str = "L",   # "L" or "RGB"
        image_size: int = 448,
        normalize_to_01: bool = True,
        local_crop: Optional[Tuple[int, int]] = (186, 410),
        image_load_timeout_sec: float = 0.0,
        skip_bad_samples: bool = False,
        max_decode_retries: int = 3,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_mode = image_mode
        self.image_size = image_size
        self.normalize_to_01 = normalize_to_01
        self.local_crop = local_crop
        self.image_load_timeout_sec = float(image_load_timeout_sec)
        self.skip_bad_samples = bool(skip_bad_samples)
        self.max_decode_retries = max(1, int(max_decode_retries))

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
        with _image_load_timeout(self.image_load_timeout_sec):
            with Image.open(path) as img:
                img = img.convert(self.image_mode)

                if self.image_size is not None:
                    img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

                arr = np.asarray(img, dtype=np.float32)

        if self.image_mode == "L":
            arr = arr[..., None]  # H,W -> H,W,1

        if self.normalize_to_01:
            arr = arr / 255.0

        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(arr).float()

    def _build_sample(self, idx: int):
        item = self.samples[idx]

        x = torch.stack([self._load_image(p) for p in item["input_paths"]], dim=0)   # [8, C, H, W]
        y = torch.stack([self._load_image(p) for p in item["target_paths"]], dim=0)  # [2, C, H, W]

        sample = {
            "x": x,
            "y": y,
            "dataset_idx": item.get("dataset_idx", idx),
            "sample_id": item.get("sample_id", item.get("sequence_id", str(idx))),
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

    def __getitem__(self, idx):
        last_error = None
        sample_idx = int(idx)
        for retry_idx in range(self.max_decode_retries):
            try:
                return self._build_sample(sample_idx)
            except Exception as exc:
                last_error = exc
                if not self.skip_bad_samples:
                    raise
                item = self.samples[sample_idx]
                sample_id = item.get("sample_id", item.get("sequence_id", str(sample_idx)))
                print(
                    f"[dataset] skipping bad sample idx={sample_idx} sample_id={sample_id} "
                    f"retry={retry_idx + 1}/{self.max_decode_retries} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                sample_idx = (sample_idx + 1) % len(self.samples)
        raise RuntimeError(
            f"Failed to decode sample starting from idx={idx} after {self.max_decode_retries} retries."
        ) from last_error
    
    def crop_f_region(self, seq: torch.Tensor) -> torch.Tensor:
        # Fixed F-region crop boundary in the 448x448 image coordinate system.
        # local_crop=(186, 410) produces a 224x448 local sequence.
        top, bottom = self.local_crop
        return seq[:, :, top:bottom, :]
