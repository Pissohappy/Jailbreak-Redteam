#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual perturbation attack strategies for VLM jailbreak attacks.

This module provides three attack strategies:
1. PhotographicAttack - Applies low-level visual perturbations (blur, noise, color shift, etc.)
2. JigsawScrambleAttack - Divides images into grid tiles and scrambles their order
3. MultimodalShuffleAttack - Creates collages with shuffled image-text pairings
"""

from .photographic import PhotographicAttack, PhotographicConfig
from .jigsaw import JigsawScrambleAttack, JigsawConfig
from .multimodal_shuffle import MultimodalShuffleAttack, MultimodalShuffleConfig
from .utils import get_font, ensure_output_dir

__all__ = [
    "PhotographicAttack",
    "PhotographicConfig",
    "JigsawScrambleAttack",
    "JigsawConfig",
    "MultimodalShuffleAttack",
    "MultimodalShuffleConfig",
    "get_font",
    "ensure_output_dir",
]
