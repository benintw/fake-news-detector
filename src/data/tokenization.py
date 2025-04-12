from typing import Dict

from transformers import AutoTokenizer


def sliding_window_tokenize(
    text: str, tokenizer: "AutoTokenizer", max_length: int = 512, stride: int = 128
) -> Dict:
    """Tokenize text using a sliding window approach with overlap.

    This approach handles long texts by breaking them into overlapping chunks,
    which helps maintain context across chunk boundaries.

    Args:
        text: The text to tokenize
        tokenizer: The tokenizer to use
        max_length: Maximum token length for each window
        stride: Number of overlapping tokens between adjacent windows

    Returns:
        Dictionary containing tokenized outputs with input_ids and attention_masks
    """
    # Use the tokenizer's built-in sliding window functionality
    return tokenizer(
        text,
        return_overflowing_tokens=True,
        stride=stride,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
