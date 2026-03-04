from typing import List, Dict, Any, Tuple
import random
import re

import nltk
from nltk import pos_tag, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import base64

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

# —— Figure 1 & Figure 2: Candidate replacement words —— #
CANDIDATE_ADJS = [
    "sweet",
    "gentle",
    "joyful",
    "charming",
    "kind",
    "bright",
    "adorable",
    "delightful",
    "friendly",
    "warm",
    "lovely",
    "caring",
    "whimsical",
    "playful",
    "radiant",
    "graceful",
    "bubbly",
    "serene",
    "hopeful",
    "cheerful",
    "compassionate",
]

CANDIDATE_NOUNS = [
    "pizza",
    "burger",
    "sushi",
    "pasta",
    "tacos",
    "steak",
    "ramen",
    "donut",
    "croissant",
    "brownie",
    "cheesecake",
    "pancakes",
    "waffles",
    "lasagna",
    "dumplings",
    "bagel",
    "paella",
    "falafel",
    "muffin",
    "burrito",
]

_detok = TreebankWordDetokenizer()


def _match_casing(src: str, tgt: str) -> str:
    """Match replacement word's case style to source word (capitalize first letter/all uppercase/rest lowercase)."""
    if src.isupper():
        return tgt.upper()
    if src[:1].isupper() and src[1:].islower():
        return tgt.capitalize()
    return tgt


def _pluralize_if_needed(word: str, is_plural: bool) -> str:
    """Simple pluralization: if plural is needed and doesn't end with s, add s."""
    if is_plural:
        return word if word.endswith("s") else word + "s"
    return word


def _replace_token(tok: str, tag: str, rng: random.Random) -> Tuple[str, str, str]:
    """
    Input a token and its POS tag, replace if it's an adjective/noun.
    Returns (new token, original word, new word). If no replacement, original and new word return empty strings.
    """
    # Only replace pure alphabetic words (avoid misjudging punctuation, numbers, etc.)
    if not re.match(r"^[A-Za-z]+$", tok):
        return tok, "", ""

    # Adjective
    if tag in ("JJ", "JJR", "JJS"):
        repl = rng.choice(CANDIDATE_ADJS)
        repl = _match_casing(tok, repl)
        return repl, tok, repl

    # Noun
    if tag in ("NN", "NNS", "NNP", "NNPS"):
        is_plural = tag in ("NNS", "NNPS")
        repl_base = rng.choice(CANDIDATE_NOUNS)
        repl = _pluralize_if_needed(repl_base, is_plural)
        repl = _match_casing(tok, repl)
        return repl, tok, repl

    return tok, "", ""


def replace_adj_noun_in_prompts(
    prompts: List[str], seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Input: prompt list
    Output: each element contains:
      {
        "idx": index starting from 1,
        "original_prompt": original prompt,
        "replaced_prompt": replaced prompt,
        "replaced_map": {original word: replacement word, ...}
      }
    """
    rng = random.Random(seed)
    results = []

    for i, text in enumerate(prompts, start=1):
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)

        new_tokens = []
        replaced_map: Dict[str, str] = {}

        for tok, tag in tags:
            new_tok, orig, repl = _replace_token(tok, tag, rng)
            new_tokens.append(new_tok)
            if orig and repl:
                # 记录“原词 -> 替换词”；若同一原词出现多次，保留首次替换
                replaced_map.setdefault(orig, repl)

        replaced_text = _detok.detokenize(new_tokens)

        results.append(
            {
                "idx": str(i),
                "original_prompt": text,
                "replaced_prompt": replaced_text,
                "replaced_map": replaced_map,
            }
        )

    return results


from typing import Dict, Any
import random, base64, re
from nltk import pos_tag, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

_detok = TreebankWordDetokenizer()


def text_transformers(prompt: str, seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)

    tokens = word_tokenize(prompt)
    tags = pos_tag(tokens)

    new_tokens = []
    replaced_map: Dict[str, str] = {}

    for tok, tag in tags:
        new_tok, orig, repl = _replace_token(tok, tag, rng)
        new_tokens.append(new_tok)
        if orig and repl:
            replaced_map.setdefault(orig, repl)

    replaced_text = _detok.detokenize(new_tokens)
    base64_prompt = base64.b64encode(prompt.encode("utf-8")).decode("utf-8")

    result = {
        "original_prompt": prompt,
        "replaced_prompt": replaced_text,
        "replaced_map": replaced_map,
        "base64_prompt": base64_prompt,
    }
    return result


from typing import Dict, Union
from PIL import Image, ImageOps


def image_transformer(
    image: Image.Image, transform_type: str = "all"
) -> Dict[str, Union[Image.Image, str]]:
    """
    Perform mirror or 180-degree rotation transformation on input single image.

    Args:
        image: Single PIL.Image object
        transform_type: "mirror", "rotation", or "all"

    Returns:
        dict: {"mirror": PIL.Image or "none", "rotation": PIL.Image or "none"}
    """
    result: Dict[str, Union[Image.Image, str]] = {"mirror": "none", "rotation": "none"}

    if transform_type in ("mirror", "all"):
        result["mirror"] = ImageOps.mirror(image)

    if transform_type in ("rotation", "all"):
        result["rotation"] = image.rotate(180)

    return result
