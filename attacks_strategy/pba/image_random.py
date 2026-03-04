import random
from typing import List, Tuple, Optional

import torch

from .minigpt_utils import prompt_wrapper, generator
from .utils import get_perspective_score


def _normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    return images * std + mean


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


class Image_Attacker:
    def __init__(
            self,
            args,
            model,
            targets: List[str],
            image: torch.Tensor,
            device: int = 0,
            batch_size: int = 1,
            text_prompt: str = "",
            constrained: bool = False,
    ):
        self.args = args
        self.model = model
        self.device = device if isinstance(device, str) else f"cuda:{device}"
        self.batch_size = batch_size
        self.text_prompt = text_prompt
        self.constrained = constrained

        self.generator = generator.Generator(model=self.model)

        if image.dim() == 3:
            image = image.unsqueeze(0)
        assert image.dim() == 4, "image must be BCHW"
        image = image.to(self.device)

        self.x_adv = _clamp01(_denormalize(image).clone())

        self.epoch = int(getattr(self.args, "epoch", getattr(self.args, "epoch_img", 400)))
        self.K = int(getattr(self.args, "K", 50))
        self.Q = int(getattr(self.args, "queries_per_eval", 10))
        self.tox_threshold = float(getattr(self.args, "tox_threshold", 1.5))

        self.noise_scale = float(getattr(self.args, "noise_scale", 0.10))
        self.noise_decay = float(getattr(self.args, "noise_decay", 0.999))

        self.best_noise: Optional[torch.Tensor] = None
        self.best_tox: float = -1.0

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _make_prompt(self, x_adv_01: torch.Tensor, text: str) -> prompt_wrapper.Prompt:
        if x_adv_01.dim() == 4:
            x_chw = x_adv_01[0]
        else:
            x_chw = x_adv_01

        x_adv_norm = _normalize(x_chw.unsqueeze(0))
        x_adv_norm_chw = x_adv_norm[0]  # CHW

        prompt = prompt_wrapper.Prompt(
            model=self.model,
            text_prompts=[text],
            img_prompts=[[x_adv_norm_chw]],
        )

        prompt.img_embs = prompt.img_embs * self.batch_size
        prompt.update_context_embs()
        return prompt

    @torch.no_grad()
    def _get_response(self, x_adv_01: torch.Tensor, text: str) -> str:
        prompt = self._make_prompt(x_adv_01, text)
        out = self.generator.generate(prompt)
        if isinstance(out, (list, tuple)) and len(out) >= 1:
            response = out[0]
        else:
            response = out
        if not isinstance(response, str) or response == "":
            response = "No response"
        return response

    def _query_once(self, x_adv_01: torch.Tensor, text: str) -> float:
        resp = self._get_response(x_adv_01, text)
        return float(get_perspective_score(resp))

    def _mean_toxicity(self, x_adv_01: torch.Tensor, text: str) -> float:
        scores = [self._query_once(x_adv_01, text) for _ in range(self.Q)]
        return float(sum(scores) / max(1, len(scores)))

    def _sample_noise(self, like: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.randn_like(like, device=like.device) * scale

    def _apply_noise(self, base_img_01: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return _clamp01(base_img_01 + noise)

    def train(self) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        x_adv = self.x_adv.clone()

        current_tox = self._mean_toxicity(x_adv, self.text_prompt)
        if current_tox >= self.tox_threshold:
            self.best_noise = torch.zeros_like(x_adv)
            self.best_tox = current_tox
            return self.best_noise.squeeze(0), _normalize(x_adv).squeeze(0)

        scale = self.noise_scale

        for _ in range(1, self.epoch + 1):
            best_local_tox = -1.0
            best_local_noise = None
            best_local_img = None

            for _k in range(self.K):
                noise = self._sample_noise(x_adv, scale)
                cand = self._apply_noise(x_adv, noise)
                tox = self._mean_toxicity(cand, self.text_prompt)
                if tox > best_local_tox:
                    best_local_tox = tox
                    best_local_noise = noise
                    best_local_img = cand

            if best_local_img is not None and best_local_tox >= current_tox:
                x_adv = best_local_img.detach()
                current_tox = best_local_tox
                self.best_noise = best_local_noise.detach()
                self.best_tox = current_tox

            if current_tox >= self.tox_threshold:
                break

            scale *= self.noise_decay

        return (self.best_noise.squeeze(0) if self.best_noise is not None else None,
                _normalize(x_adv).squeeze(0))
