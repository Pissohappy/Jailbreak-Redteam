# attacks/visual_adv/attack.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple, Optional, Union
import os
import json
import csv
import random
from pathlib import Path

import torch
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from core.base_classes import BaseAttack
from core.data_formats import TestCase
from dataclasses import dataclass


@dataclass
class VisualAdvConfig:
    epsilon: float = 8.0
    alpha: float = 2.0
    n_iters: int = 500
    constrained: bool = True
    batch_size: int = 8
    image_path: str = "attacks/data/clean.jpeg"
    target_path: str = "attacks/data/derogatory_corpus.csv"
    target_model: str = "minigpt4.minigpt4_model.MiniGPT4"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class VisualAdvAttack(BaseAttack):
    CONFIG_CLASS = VisualAdvConfig

    def __init__(self, config: Dict[str, Any] = None, output_image_dir: str = None):
        super().__init__(config, output_image_dir)
        # If epsilon/alpha looks like pixel values (>1), convert to [0,1] ratio
        self.epsilon = getattr(self.cfg, "epsilon", 8.0)
        self.alpha = getattr(self.cfg, "alpha", 2.0)
        # if epsilon looks like pixel ( > 1 ) convert to fraction over 255
        if self.epsilon > 1.0:
            self.epsilon = float(self.epsilon) / 255.0
        if self.alpha > 1.0:
            self.alpha = float(self.alpha) / 255.0

        self.n_iters = int(getattr(self.cfg, "n_iters", 500))
        self.constrained = bool(getattr(self.cfg, "constrained", True))
        self.batch_size = int(getattr(self.cfg, "batch_size", 8))
        self.image_path = str(getattr(self.cfg, "image_path", ""))
        self.target_path = str(getattr(self.cfg, "target_path", ""))
        self.target_model_name = str(getattr(self.cfg, "target_model", ""))
        self.device = torch.device(
            getattr(self.cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load target list (first column of CSV is the target)
        self.targets = []
        if self.target_path and os.path.exists(self.target_path):
            try:
                with open(self.target_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 1 and row[0].strip():  # Ensure not empty string
                            self.targets.append(row[0].strip())
            except Exception as e:
                print(
                    f"[VisualAdv] Warning: Failed to load targets from {self.target_path}: {e}"
                )

        # If targets is empty, provide default value or raise error
        if not self.targets:
            raise ValueError(
                f"No targets loaded from {self.target_path}. "
                "Please ensure the target_path points to a valid CSV file with at least one target."
            )

        # Load model (assume multimodalmodels module is already in the project)
        # Use lazy import to avoid import-time issues
        # self.model = getattr(multimodalmodels, self.target_model_name)()
        import importlib

        module_path, cls_name = self.target_model_name.rsplit(".", 1)
        mod = importlib.import_module(f"multimodalmodels.{module_path}")
        self.model = getattr(mod, cls_name)()
        self.save_path = None

    # -----------------------
    # PGD related functions (keep original logic, but safer)
    # -----------------------
    def _pgd_unconstrained(self, image_path: str):
        # open image -> model.vis_processor -> tensor (B,C,H,W) or (C,H,W)
        img = Image.open(image_path).convert("RGB")
        # assume model has vis_processor that returns a tensor or processed object
        # try a few possibilities
        proc = self.model.vis_processor(img)
        if isinstance(proc, dict):
            # sometimes processors return dict with "pixel_values"
            img_t = proc.get("pixel_values")
        else:
            img_t = proc

        # ensure tensor shape [1,C,H,W] and on device
        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)
        img_t = img_t.to(self.device)

        # adv_noise initialized in [0,1]
        adv_noise = torch.rand_like(img_t, device=self.device)
        adv_noise.requires_grad_(True)

        adv_img_prompt = None
        for t in tqdm(range(self.n_iters + 1), desc="PGD_unconstrained"):
            # sample targets
            if not self.targets:
                raise ValueError("Cannot sample targets: targets list is empty")
            if len(self.targets) >= self.batch_size:
                batch_targets = random.sample(self.targets, self.batch_size)
            else:
                batch_targets = random.choices(self.targets, k=max(1, self.batch_size))

            x_adv = self._normalize(
                adv_noise
            )  # normalizing model input as original implementation
            # model.compute_loss_batch expects x_adv to be in a format acceptable to the model
            target_loss = self.model.compute_loss_batch(x_adv, batch_targets)
            # Backward pass to compute gradients
            if target_loss is None:
                raise RuntimeError("model.compute_loss_batch returned None")
            target_loss.backward()

            with torch.no_grad():
                adv_noise.data = (
                    adv_noise.data - self.alpha * adv_noise.grad.detach().sign()
                ).clamp(0.0, 1.0)
            adv_noise.grad.zero_()
            # Clear model gradients
            try:
                self.model.model.zero_grad()
            except Exception:
                pass

            if t % 100 == 0 or t == self.n_iters:
                if (
                    verbose := False
                ):  # Only keep hook, remove this line if verbose is needed
                    print(f"target_loss: {target_loss.item():.6f} at iter {t}")

                x_adv = self._normalize(adv_noise)
                adv_img = self._denormalize(x_adv).detach().cpu()
                # adv_img: [1,C,H,W] -> squeeze
                adv_img_prompt = adv_img.squeeze(0)
                # Don't save temporary images here each time to avoid generating many files

        if adv_img_prompt is None:
            raise RuntimeError(
                "Failed to produce adversarial image in PGD_unconstrained"
            )
        return adv_img_prompt

    def _pgd_constrained(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        proc = self.model.vis_processor(img)
        # try common key names
        img_t = None
        if isinstance(proc, dict):
            for k in ("pixel_values", "pixel_value", "input_pixels"):
                if k in proc:
                    img_t = proc[k]
                    break
        if img_t is None:
            # fallback: processor returned tensor directly
            if hasattr(proc, "unsqueeze"):
                img_t = proc
            else:
                raise RuntimeError(
                    "Unable to obtain tensor from model.vis_processor for constrained PGD"
                )

        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)
        img_t = img_t.to(self.device)

        # initialize adv_noise in [-epsilon, +epsilon]
        adv_noise = (
            torch.rand_like(img_t, device=self.device) * 2.0 * self.epsilon
            - self.epsilon
        ).detach()
        x = self._denormalize(img_t).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0.0, 1.0) - x.data
        adv_noise.requires_grad_(True)

        # freeze model params if possible
        try:
            self.model.model.requires_grad_(False)
        except Exception:
            pass

        adv_img_prompt = None
        for t in tqdm(range(self.n_iters + 1), desc="PGD_constrained"):
            if not self.targets:
                raise ValueError("Cannot sample targets: targets list is empty")
            if len(self.targets) >= self.batch_size:
                batch_targets = random.sample(self.targets, self.batch_size)
            else:
                batch_targets = random.choices(self.targets, k=max(1, self.batch_size))

            x_adv = x + adv_noise
            x_adv = self._normalize(x_adv)

            target_loss = self.model.compute_loss_batch(x_adv, batch_targets)
            if target_loss is None:
                raise RuntimeError("model.compute_loss_batch returned None")

            target_loss.backward()
            with torch.no_grad():
                adv_noise.data = (
                    adv_noise.data - self.alpha * adv_noise.grad.detach().sign()
                ).clamp(-self.epsilon, self.epsilon)
                adv_noise.data = (adv_noise.data + x.data).clamp(0.0, 1.0) - x.data
            adv_noise.grad.zero_()

            if t % 100 == 0 or t == self.n_iters:
                if verbose := False:
                    print(f"target_loss: {target_loss.item():.6f} at iter {t}")
                x_adv = x + adv_noise
                x_adv = self._normalize(x_adv)
                adv_img = self._denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img.squeeze(0)

        if adv_img_prompt is None:
            raise RuntimeError("Failed to produce adversarial image in PGD_constrained")
        return adv_img_prompt

    # normalize/denormalize: Note to keep consistent with mean/variance used by the model
    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=images.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=images.device
        ).view(1, 3, 1, 1)
        return (images - mean) / std

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=images.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=images.device
        ).view(1, 3, 1, 1)
        return images * std + mean

    # Override generate: expects prompts to be dict (consistent with what's passed in main.py)
    def generate_test_case(
        self,
        original_prompt: str,
        image_path: str,
        case_id: str,
        **kwargs,
    ) -> TestCase:
        if self.save_path is None:
            ## Only need to generate image once:
            if self.constrained:
                adv_img_tensor = self._pgd_constrained(self.image_path)
            else:
                adv_img_tensor = self._pgd_unconstrained(self.image_path)
            if adv_img_tensor.dim() == 3:
                adv_t_save = adv_img_tensor.unsqueeze(0)
            else:
                adv_t_save = adv_img_tensor

            self.save_path = self.output_image_dir / "advimg.jpg"
            save_image(adv_t_save, str(self.save_path))

        return self.create_test_case(
            case_id=case_id,
            jailbreak_prompt=original_prompt,
            jailbreak_image_path=str(self.save_path),
            original_prompt=original_prompt,
            original_image_path=str(image_path),
        )
