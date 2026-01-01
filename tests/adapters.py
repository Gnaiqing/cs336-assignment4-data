from __future__ import annotations

import os
from typing import Any


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from cs336_data.preprocess import extract_text_from_bytestr
    return extract_text_from_bytestr(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    from cs336_data.preprocess import identify_language
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    from cs336_data.preprocess import mask_emails
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    from cs336_data.preprocess import mask_phone_numbers
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    from cs336_data.preprocess import mask_ip_addresses
    return mask_ip_addresses(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    from cs336_data.preprocess import identify_nsfw
    return identify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    from cs336_data.preprocess import identify_toxic_speech
    return identify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    from cs336_data.preprocess import identify_data_quality
    return identify_data_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    from cs336_data.preprocess import gopher_quality_filter
    label, msg = gopher_quality_filter(text)
    return label


def run_exact_line_deduplication(
        input_files: list[os.PathLike], output_directory: os.PathLike
):
    from cs336_data.preprocess import exact_deduplication
    exact_deduplication(input_files, output_directory)
    return None


def run_minhash_deduplication(
        input_files: list[os.PathLike],
        num_hashes: int,
        num_bands: int,
        ngrams: int,
        jaccard_threshold: float,
        output_directory: os.PathLike,
):
    from cs336_data.preprocess import minhash_deduplication
    minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)
