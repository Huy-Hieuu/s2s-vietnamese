"""LLM data preparation — tokenization and dataset utilities."""

from __future__ import annotations

from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    import json

    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info("loaded_jsonl", path=str(path), count=len(items))
    return items


def format_chat_llama3(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Format messages using the Llama 3 chat template.

    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": "..."}.
        add_generation_prompt: Append assistant prefix at the end.

    Returns:
        Formatted prompt string.
    """
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
    if add_generation_prompt:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def verify_tokenizer_diacritics(tokenizer: object, sample_texts: list[str] | None = None) -> bool:
    """Verify that the tokenizer handles Vietnamese diacritics without character splitting.

    Returns True if all texts round-trip correctly.
    """
    if sample_texts is None:
        sample_texts = [
            "Xin chào, tôi là trợ lý AI.",
            "Thời tiết hôm nay rất đẹp.",
            "Bạn có thể giúp tôi không?",
            "Học máy và trí tuệ nhân tạo.",
        ]

    all_ok = True
    for text in sample_texts:
        token_ids = tokenizer.encode(text)  # type: ignore[attr-defined]
        decoded = tokenizer.decode(token_ids)  # type: ignore[attr-defined]
        # Strip special tokens for comparison
        decoded_clean = decoded.strip()
        if decoded_clean != text:
            logger.warning(
                "diacritics_roundtrip_failed",
                original=text,
                decoded=decoded_clean,
            )
            all_ok = False
    return all_ok
