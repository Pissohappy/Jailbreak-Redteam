#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for visual perturbation attacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageFont


def get_font(
    font_path: Optional[str], size: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load font from path or return a default system font."""
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, size)

    # Try common system fonts
    common_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for fp in common_fonts:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)

    # Fallback to default font
    return ImageFont.load_default()


def ensure_output_dir(output_dir: Optional[Path]) -> Path:
    """Ensure output directory exists and return the path."""
    if output_dir is None:
        raise ValueError("output_image_dir is not set")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
