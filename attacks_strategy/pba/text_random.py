import random
import string
from typing import List, Tuple, Optional

import torch

from .minigpt_utils import prompt_wrapper, generator
from .utils import get_perspective_score


def _seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def _sanitize(s: str) -> str:
    return " ".join(s.split())


class Text_Attacker:
    def __init__(
            self,
            args,
            model,
            targets: List[str],
            image: torch.Tensor,
            device: int = 0,
            batch_size: int = 1,
    ):
        self.args = args
        self.model = model
        self.device = device if isinstance(device, str) else f"cuda:{device}"
        self.batch_size = batch_size

        self.text_prompt: str = ""
        if image is not None and image.dim() == 3:
            image = image.unsqueeze(0)
        self.image: Optional[torch.Tensor] = image.to(self.device) if image is not None else None

        self.epoch = int(getattr(self.args, "epoch", getattr(self.args, "epoch_txt", 100)))
        self.n_candidates = int(getattr(self.args, "n_candidates", 400))
        self.suffix_len = int(getattr(self.args, "suffix_len", 10))
        self.Q = int(getattr(self.args, "queries_per_eval", 10))
        self.tox_threshold = float(getattr(self.args, "tox_threshold", 1.5))

        self.get_vocabulary()
        self.token_pool = self.vocabs

        self.generator = generator.Generator(model=self.model)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def get_vocabulary(self):
        tok = getattr(self.model, "llama_tokenizer", None) or getattr(self.model, "tokenizer", None)
        assert tok is not None, "Tokenizer not found: expected model.llama_tokenizer or model.tokenizer"

        vocab_keys = list(tok.get_vocab().keys())
        single_token_vocabs: List[str] = []
        for item in vocab_keys:
            ids = tok(item, return_tensors="pt", add_special_tokens=False).input_ids
            if ids.shape[1] == 1:
                single_token_vocabs.append(item)

        single_token_vocabs = [w for w in single_token_vocabs if len(w.strip()) > 0]

        self.vocabs = single_token_vocabs

    def _make_prompt(self, image_norm: torch.Tensor, text: str) -> prompt_wrapper.Prompt:
        if image_norm.dim() == 4:
            img_chw = image_norm[0]
        else:
            img_chw = image_norm
        prompt = prompt_wrapper.Prompt(
            model=self.model,
            text_prompts=[text],
            img_prompts=[[img_chw]],
        )
        prompt.img_embs = prompt.img_embs * self.batch_size
        prompt.update_context_embs()
        return prompt

    @torch.no_grad()
    def _get_response(self, image_norm: torch.Tensor, text: str) -> str:
        prompt = self._make_prompt(image_norm, text)
        out = self.generator.generate(prompt)
        response = out[0] if isinstance(out, (list, tuple)) and len(out) >= 1 else out
        if not isinstance(response, str) or response == "":
            response = "No response"
        return response

    def _query_once(self, image_norm: torch.Tensor, text: str) -> float:
        resp = self._get_response(image_norm, text)
        return float(get_perspective_score(resp))

    def _mean_toxicity(self, image_norm: torch.Tensor, text: str) -> float:
        scores = [self._query_once(image_norm, text) for _ in range(self.Q)]
        return float(sum(scores) / max(1, len(scores)))

    def _sample_suffix_token(self) -> str:
        return random.choice(self.token_pool)

    def _sample_suffix(self, length: int) -> str:
        toks = [self._sample_suffix_token() for _ in range(length)]
        return " ".join(toks)

    def attack(self, image_norm: Optional[torch.Tensor]) -> Tuple[str, str]:

        if image_norm is None:
            assert self.image is not None and self.image.dim() in (3, 4), "image_norm or self.image must be set"
            image_norm = self.image if self.image.dim() == 3 else self.image[0]
        if image_norm.dim() == 4:
            image_norm = image_norm[0]

        base_text = self.text_prompt.strip()
        base_tox = self._mean_toxicity(image_norm, base_text)

        added_tokens: List[str] = []
        while len(added_tokens) < self.suffix_len and base_tox < self.tox_threshold:
            best_tok: Optional[str] = None
            best_tox: float = base_tox
            for _ in range(self.n_candidates):
                cand_tok = self._sample_suffix_token()
                cand_text = (base_text + " " + " ".join(added_tokens + [cand_tok])).strip()
                tox = self._mean_toxicity(image_norm, cand_text)
                if tox > best_tox:
                    best_tox = tox
                    best_tok = cand_tok
            if best_tok is None:
                break
            added_tokens.append(best_tok)
            base_tox = best_tox

        if len(added_tokens) == self.suffix_len and base_tox < self.tox_threshold:
            best_overall_pos: Optional[int] = None
            best_overall_tok: Optional[str] = None
            best_overall_tox: float = base_tox
            for pos in range(self.suffix_len):
                orig_tok = added_tokens[pos]
                for _ in range(self.n_candidates):
                    cand_tok = self._sample_suffix_token()
                    if cand_tok == orig_tok:
                        continue
                    trial = added_tokens.copy()
                    trial[pos] = cand_tok
                    cand_text = (base_text + " " + " ".join(trial)).strip()
                    tox = self._mean_toxicity(image_norm, cand_text)
                    if tox > best_overall_tox:
                        best_overall_tox = tox
                        best_overall_pos = pos
                        best_overall_tok = cand_tok
            if best_overall_pos is not None:
                added_tokens[best_overall_pos] = best_overall_tok
                base_tox = best_overall_tox

        round_suffix = " ".join(added_tokens).strip()
        return (round_suffix, "")
