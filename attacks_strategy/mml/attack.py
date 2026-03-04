#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from pathlib import Path
import os
import json
import random
import re
from PIL import Image, ImageFont, ImageDraw
import textwrap

from core.data_formats import TestCase
from core.base_classes import BaseAttack


from . import utils

# Import prompt templates from const
from .const import (
    wr_game_prompt,
    mirror_game_prompt,
    base64_game_prompt,
    rotate_game_prompt,
)

# Build mapping
DATAFORMAT2PROMPT = {
    "images_wr": wr_game_prompt,
    "images_mirror": mirror_game_prompt,
    "images_base64": base64_game_prompt,
    "images_rotate": rotate_game_prompt,
}


# ===================== Utility Functions =====================
def _get_font(font_path: Optional[str], size: int):
    return ImageFont.truetype(font_path, size)


def text_step_by_step(text: str, steps=3, wrap=False, wrap_width=15):
    text = text.removesuffix("\n")
    if wrap:
        text = textwrap.fill(text, width=wrap_width)
    for idx in range(1, steps + 1):
        text += f"\n{idx}. "
    return text


def text_to_image(
    text: str,
    font_path: Optional[str] = None,
    font_size: int = 80,
    image_size: Tuple[int, int] = (760, 760),
    margin_xy: Tuple[int, int] = (20, 10),
    spacing: int = 11,
    bg="#FFFFFF",
    fg="#000000",
) -> Image.Image:
    font = _get_font(font_path, font_size)
    im = Image.new("RGB", image_size, bg)
    dr = ImageDraw.Draw(im)
    dr.text(margin_xy, text, fill=fg, font=font, spacing=spacing)
    return im


def gen_image_from_text(
    text: str,
    *,
    font_path: Optional[str] = None,
    font_size: int = 80,
    image_size=(760, 760),
    margin_xy=(20, 10),
    wrap_width=15,
    steps=3,
) -> Image.Image:
    text2 = text_step_by_step(text, steps=steps, wrap=True, wrap_width=wrap_width)
    image = text_to_image(
        text2,
        font_path=font_path,
        font_size=font_size,
        margin_xy=margin_xy,
        image_size=image_size,
    )
    return image


def scramble_str(title: str):
    """
    Extract words from title and randomly shuffle order
    """
    import re

    words = re.findall(r"\b\w+\b", title)
    scrambled = words[:]
    random.shuffle(scrambled)
    return scrambled


# ===================== Configuration =====================
@dataclass
class MMLConfig:
    """MML attack configuration - new architecture independent configuration class"""

    # Rendering parameters
    font_path: Optional[str] = None
    font_size: int = 80
    img_size: Tuple[int, int] = (760, 760)
    wrap_width: int = 15
    steps: int = 3
    margin_xy: Tuple[int, int] = (20, 20)
    aug_type: str = "wr"


# ===================== MML Attack Implementation =====================
class MMLAttack(BaseAttack):
    """
    MML attack method - new architecture implementation
    Generates four types of attack images: word-replace, base64, mirror, rotation
    """

    # Configuration class
    CONFIG_CLASS = MMLConfig

    def __init__(self, config: Dict[str, Any] = None, output_image_dir: str = None):
        """Initialize MML attack method

        Args:
            config: Configuration dictionary, will be loaded from config file and passed to parent class
            output_image_dir: Output image directory path
        """
        # Call parent class initialization, automatically handle configuration conversion
        super().__init__(config, output_image_dir)

        # Mapping for prompt field in original script
        self.dataformat2prompt = DATAFORMAT2PROMPT

    def generate_test_case(
        self,
        original_prompt: str,
        image_path: str,
        case_id: str,
        **kwargs,
    ) -> TestCase:
        """
        Generate test case - MML attack implementation

        Args:
            original_prompt: Original prompt string (harmful behavior text)
            image_path: Original image path (MML generates its own image, doesn't use provided image)
            case_id: Test case ID
            **kwargs: Additional parameters

        Returns:
            Single test case object
        """

        # MML attack generates four types of images, we need to generate a test case for each type
        # Here we generate word-replace type as an example, actual implementation may need to generate all four types
        # prepare subdirs
        subdirs = {
            "wr": self.output_image_dir / "word_replace",
            "base64": self.output_image_dir / "base64",
            "mirror": self.output_image_dir / "mirror",
            "rotation": self.output_image_dir / "rotation",
        }

        subdirs[self.cfg.aug_type].mkdir(parents=True, exist_ok=True)

        if self.cfg.aug_type == "wr":
            res = utils.text_transformers(original_prompt)

            wr_img = gen_image_from_text(
                res.get("replaced_prompt", original_prompt),
                font_path=self.cfg.font_path,
                font_size=self.cfg.font_size,
                image_size=self.cfg.img_size,
                wrap_width=self.cfg.wrap_width,
                margin_xy=self.cfg.margin_xy,
                steps=self.cfg.steps,
            )
            wr_name = f"{case_id}-wr.png"
            img_path = str(subdirs["wr"] / wr_name)
            wr_img.save(img_path)
            adv_prompt = self.dataformat2prompt.get("images_wr", "images_wr").format(
                res.get("replaced_map", ""), res.get("replaced_prompt", "")
            )
        elif self.cfg.aug_type == "base64":
            # Generate base64 image
            base64_src = res.get("base64_prompt", original_prompt)
            base64_img = gen_image_from_text(
                base64_src,
                font_path=self.cfg.font_path,
                font_size=self.cfg.font_size,
                image_size=self.cfg.img_size,
                wrap_width=self.cfg.wrap_width,
                steps=self.cfg.steps,
            )
            base64_name = f"{case_id}-base64.png"
            img_path = str(subdirs["base64"] / base64_name)
            base64_img.save(img_path)
            adv_prompt = self.dataformat2prompt.get(
                "images_base64", "images_base64"
            ).format(res.get("base64_prompt", ""))
        else:
            # Generate normal image for mirror and rotation transformation
            normal_img = gen_image_from_text(
                original_prompt,
                font_path=self.cfg.font_path,
                font_size=self.cfg.font_size,
                image_size=self.cfg.img_size,
                wrap_width=self.cfg.wrap_width,
                steps=self.cfg.steps,
            )
            scrambled_prompt = scramble_str(original_prompt)
            img_res = utils.image_transformer(normal_img, transform_type="all")
            # Apply image transformation
            if self.cfg.aug_type == "mirror":
                mirror_img = img_res.get("mirror", None)

                mirror_name = f"{case_id}-mirror.png"
                img_path = str(subdirs["mirror"] / mirror_name)
                mirror_img.save(img_path)

                adv_prompt = self.dataformat2prompt.get(
                    "images_mirror", "images_mirror"
                )
            elif self.cfg.aug_type == "rotation":
                rotation_img = img_res.get("rotation", None)
                rotation_name = f"{case_id}-rotation.png"
                img_path = str(subdirs["rotation"] / rotation_name)
                rotation_img.save(img_path)
                adv_prompt = self.dataformat2prompt.get(
                    "images_rotate", "images_rotate"
                ).format(scrambled_prompt)

        return self.create_test_case(
            case_id=case_id,
            jailbreak_prompt=adv_prompt,
            jailbreak_image_path=str(img_path),
            original_prompt=original_prompt,
            original_image_path=str(image_path),
            metadata={"aug_type": self.cfg.aug_type},
        )
