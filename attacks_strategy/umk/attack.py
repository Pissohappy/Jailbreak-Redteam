# attacks/umk/attack.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple, Optional, Union

import csv
import random
from torchvision.utils import save_image
from PIL import Image

from .minigpt_utils import vlm_attacker, prompt_wrapper
from dataclasses import dataclass
from core.base_classes import BaseAttack, TestCase


@dataclass
class UMKConfig:
    n_iters: int = 5000
    eps: int = 32
    alpha: float = 1.0
    corpus: str = "attacks/data/harmful_behaviors.csv"
    prior_image: str = ""
    batch_size: int = 8
    target_model: str = None


class UMKAttack(BaseAttack):
    CONFIG_CLASS = UMKConfig

    def __init__(self, config: Dict[str, Any] = None, output_image_dir: str = None):
        super().__init__(config, output_image_dir)
        import importlib

        self.target_model_name = str(getattr(self.cfg, "target_model", ""))
        module_path, cls_name = self.target_model_name.rsplit(".", 1)
        mod = importlib.import_module(f"multimodalmodels.{module_path}")
        self.model = getattr(mod, cls_name)()

    def generate_test_case(
        self,
        original_prompt: str,
        image_path: str,
        case_id: str,
        **kwargs,
    ) -> TestCase:

        images_dir = self.output_image_dir

        file = open(self.cfg.corpus, "r")

        data = list(csv.reader(file))
        file.close()
        queries = []
        targets = []
        num = len(data)

        for i in range(1, num):
            queries.append(data[i][0])
            targets.append(data[i][1])

        all_numbers = list(range(len(queries)))

        selected_numbers = random.sample(all_numbers, 66)

        queries = [queries[index] for index in selected_numbers]
        targets = [targets[index] + ":" for index in selected_numbers]

        my_attacker = vlm_attacker.Attacker(
            self.cfg,
            self.model,
            queries,
            targets,
            device=self.model.device,
            is_rtp=False,
        )

        img_path = getattr(self.cfg, "prior_image", "")
        img = Image.open(img_path).convert("RGB")
        img = self.model.vis_processor(img).unsqueeze(0).to(self.model.device)
        text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt
        adv_img_tensor, text_suffix = my_attacker.attack_unconstrained(
            text_prompt_template,
            img=img,
            batch_size=self.cfg.batch_size,
            num_iter=self.cfg.n_iters,
            alpha=self.cfg.alpha / 255,
        )
        if adv_img_tensor.dim() == 3:
            adv_t_save = adv_img_tensor.unsqueeze(0)
        else:
            adv_t_save = adv_img_tensor
        save_path = images_dir / f"umk_{case_id}.jpg"
        save_image(adv_t_save, str(save_path))

        jailbreak_prompt = original_prompt + text_suffix
        return self.create_test_case(
            case_id=case_id,
            jailbreak_prompt=jailbreak_prompt,
            jailbreak_image_path=str(save_path),
            original_prompt=original_prompt,
            original_image_path=str(image_path),
        )
