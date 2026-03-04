# attacks/umk/attack.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple, Optional, Union
import os
import json
import csv
import random
from pathlib import Path

import torch

from .utils.torchattacks.attacks.pixle import *
from .utils.torchattacks.attacks.bim import *
from .utils.torchattacks.attacks.pgd_uap_v1 import *
from .utils.torchattacks.attacks.pgdl2 import *

import numpy as np

from core.data_formats import TestCase
from core.base_classes import BaseAttack
from dataclasses import dataclass


def denorm(image, device):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

    image_denorm = image * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

    return image_denorm


# -----------------------
# Default configuration (can be overridden in config.json)
# -----------------------
@dataclass
class ImgJPConfig:
    device: str = "cuda"
    eps: float = 32
    alpha: float = 1
    steps: int = 3000
    target_model: str = "minigpt4.minigpt4_model.MiniGPT4"


class ImgJPAttack(BaseAttack):
    CONFIG_CLASS = ImgJPConfig

    def __init__(self, config: Dict[str, Any] = None, output_image_dir: str = None):
        super().__init__(config, output_image_dir)
        # Parameter settings
        self.eps = self.cfg.eps
        self.alpha = self.cfg.alpha
        self.steps = self.cfg.steps
        self.device = self.cfg.device

        import importlib

        self.target_model_name = str(getattr(self.cfg, "target_model", ""))
        module_path, cls_name = self.target_model_name.rsplit(".", 1)
        mod = importlib.import_module(f"multimodalmodels.{module_path}")
        self.model = getattr(mod, cls_name)()
        self.adv_image = None

    def generate_test_case(
        self, original_prompt: str, image_path: str, case_id: str, **kwargs
    ) -> TestCase:
        if self.adv_image is None:

            random_number = random.randint(1, 2000)
            random.seed(random_number)
            np.random.seed(random_number)
            torch.manual_seed(random_number)

            attack = PGD(
                self.model,
                eps=self.eps / 255,
                alpha=self.alpha / 255,
                steps=self.steps,
                # nprompt=self.model.train_num,
                random_start=False,
            )

            attack.set_mode_targeted_by_label()
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
            attack.set_normalization_used(mean, std)

            image = torch.zeros(1, 3, 224, 224).to(self.device)
            images = []
            images.append(image)
            adv_img = attack(images, self.model.shift_labels_1)
            self.adv_image = denorm(adv_img[0])

            self.adv_image_path = self.output_image_dir / f"imgjp_adv.jpg"
            self.adv_image.save(self.adv_image_path)

        return self.create_test_case(
            case_id=case_id,
            jailbreak_prompt=original_prompt,
            jailbreak_image_path=str(self.adv_image_path),
            original_prompt=original_prompt,
            original_image_path=str(image_path),
        )
