from fastwarc.warc import ArchiveIterator, WarcRecordType
from fastwarc.stream_io import *
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding, bytes_to_str
import fasttext
import os
from pathlib import Path
import re
import unicodedata
import random
from typing import Iterable, List, Tuple
from collections import defaultdict
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from cs336_data.utils import DisjointSet
import shutil

language_cls = fasttext.FastText.load_model("lid.176.bin")
nsfw_cls = fasttext.FastText.load_model("jigsaw_fasttext_bigrams_nsfw_final.bin")
toxic_cls = fasttext.FastText.load_model("jigsaw_fasttext_bigrams_hatespeech_final.bin")
quality_cls = fasttext.FastText.load_model("model_dataquality.bin")

EMAIL_PATTERN = re.compile(
    r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
)

PHONE_PATTERN = re.compile(
    r'''
    (?<!\w)                # left boundary (not letter/number)
    (?:\+1[\s.-]?)?        # optional country code
    (?:                    # area code
        \(\d{3}\)          # (415)
        |                  # or
        \d{3}              # 415
    )
    [\s.-]?                # optional separator
    \d{3}                  # exchange
    [\s.-]?                # optional separator
    \d{4}                  # line number
    (?!\w)                 # right boundary
    ''',
    re.VERBOSE
)

IPV4_PATTERN = re.compile(
    r'''
    (?<!\d)                              # not preceded by a digit
    (?:                                 # first 3 octets
        (?:25[0-5]                      # 250–255
        | 2[0-4]\d                      # 200–249
        | 1\d{2}                        # 100–199
        | [1-9]?\d)                     # 0–99
        \.
    ){3}
    (?:25[0-5]                          # last octet
    | 2[0-4]\d
    | 1\d{2}
    | [1-9]?\d)
    (?!\d)                              # not followed by a digit
    ''',
    re.VERBOSE
)


def extract_text_from_bytestr(bytestr: bytes):
    decoded = bytes_to_str(bytestr, detect_encoding(bytestr))
    html_text = extract_plain_text(decoded)
    return html_text


def identify_language(text: str):
    cleaned_text = text.replace("\n", "")
    label, prob = language_cls.predict(cleaned_text)
    return label[0].removeprefix("__label__"), prob[0]


def identify_nsfw(text: str):
    cleaned_text = text.replace("\n", "")
    label, prob = nsfw_cls.predict(cleaned_text)
    return label[0].removeprefix("__label__"), prob[0]


def identify_toxic_speech(text: str):
    cleaned_text = text.replace("\n", "")
    label, prob = toxic_cls.predict(cleaned_text)
    return label[0].removeprefix("__label__"), prob[0]


def identify_data_quality(text: str):
    cleaned_text = text.replace("\n", "")
    label, prob = quality_cls.predict(cleaned_text)
    return label[0].removeprefix("__label__"), prob[0]


def mask_emails(text: str):
    return EMAIL_PATTERN.subn("|||EMAIL_ADDRESS|||", text)


def mask_phone_numbers(text: str) -> Tuple[str, int]:
    """
    Replace US phone numbers with '|||PHONE_NUMBER|||'
    and return (masked_text, num_masks).
    """
    masked_text, num_masks = PHONE_PATTERN.subn(
        "|||PHONE_NUMBER|||",
        text
    )
    return masked_text, num_masks


def mask_ip_addresses(text: str) -> Tuple[str, int]:
    """
    Replace all IPv4 addresses with '|||IP_ADDRESS|||'
    and return (masked_text, num_masks).
    """
    masked_text, num_masks = IPV4_PATTERN.subn(
        "|||IP_ADDRESS|||",
        text
    )
    return masked_text, num_masks


def mask_pii(text: str) -> Tuple[str, int]:
    masked_text, email_masks = mask_emails(text)
    masked_text, phone_masks = mask_phone_numbers(masked_text)
    masked_text, ip_masks = mask_ip_addresses(masked_text)
    num_masks = email_masks + phone_masks + ip_masks
    return masked_text, num_masks


def check_language_distribution(stream, sample_size=100):
    total_cnt = 0
    cnt = 0
    for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
        html_bytes = record.reader.read()
        html_text = extract_text_from_bytestr(html_bytes)
        lan, prob = identify_language(html_text)
        total_cnt += 1
        if lan == "en":
            cnt += 1
            print(html_text)
            print(f"======Language: {lan}, Prob: {prob:.2f}=======")
            masked_text, num_masks = mask_pii(html_text)
            print(masked_text)
            print(f"======Num Masks: {num_masks}=======")

        if total_cnt == sample_size:
            break

    print(f"English fraction: {(cnt / sample_size):.2f}%")


def check_harmful_content(stream, sample_size=20):
    """
    Select a number of records and check whether they have harmful content
    Args:
        stream:
        n_check:

    Returns:

    """
    cnt = 0
    for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
        html_bytes = record.reader.read()
        html_text = extract_text_from_bytestr(html_bytes)
        lan, prob = identify_language(html_text)
        if lan == "en" and prob >= 0.7:
            cnt += 1
            label1, prob1 = identify_nsfw(html_text)
            label2, prob2 = identify_toxic_speech(html_text)
            print(html_text)
            print(f"======NSFW  label: {label1}, Prob: {prob1:.2f}=======")
            print(f"======Toxic label: {label2}, Prob: {prob2:.2f}=======")

        if cnt >= sample_size:
            break


def check_gopher_quality(stream, sample_size=20):
    """
    Select a number of records and check whether they have harmful content
    Args:
        stream:
        n_check:

    Returns:

    """
    cnt = 0
    for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
        html_bytes = record.reader.read()
        html_text = extract_text_from_bytestr(html_bytes)
        lan, prob = identify_language(html_text)
        if lan == "en" and prob >= 0.7:
            cnt += 1
            quality, msg = gopher_quality_filter(html_text)
            print(html_text)
            print(f"======Quality Check: {quality}, MSG: {msg}=======")

            if cnt >= sample_size:
                break


def contains_alpha(input_string):
    """
    Checks if a string contains at least one alphabetic character.
    """
    return any(c.isalpha() for c in input_string)


def gopher_quality_filter(text: str) -> Tuple[bool, str]:
    from nltk.tokenize import word_tokenize
    import numpy as np
    words = word_tokenize(text)
    if len(words) < 50 or len(words) > 100000:
        return False, "word count exceed range"
    avg_len = np.mean([len(token) for token in words])
    if avg_len < 3 or avg_len > 10:
        return False, "average word length exceed range"
    lines = text.split("\n")
    ellipsis_postfix = [line.endswith("...") for line in lines]
    ellipsis_ratio = np.mean(ellipsis_postfix)
    if ellipsis_ratio >= 0.3:
        return False, "too many lines ending with ..."
    alpha_character = [contains_alpha(word) for word in words]
    alpha_character_ratio = np.mean(alpha_character)
    if alpha_character_ratio < 0.8:
        return False, "no enough words containing alphabetic character"

    return True, "all check passed"


def exact_deduplication(input_paths: List, output_dir: str):
    """
    Deduplicate files specified in input_paths, store them in output_dir
    Args:
        input_paths:
        output_dir:

    Returns:

    """
    all_lines = set()
    redundant_lines = set()
    for input_path in input_paths:
        with open(input_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line_val = hash(line) % (1e9 + 7)
                if line_val in all_lines:
                    redundant_lines.add(line_val)
                else:
                    all_lines.add(line_val)

    for input_path in input_paths:
        filename = os.path.basename(input_path)
        with open(input_path, "r") as f:
            lines = f.readlines()
            unique_lines = []
            for line in lines:
                line_val = hash(line) % (1e9 + 7)
                if line_val not in redundant_lines:
                    unique_lines.append(line)

        with open(Path(output_dir) / filename, "w") as f:
            f.write("".join(unique_lines))

P = (1 << 61) - 1  # a common large prime (2^61 - 1)

def make_ab(k: int, rng_seed: int = 12345) -> List[Tuple[int, int]]:
    rng = random.Random(rng_seed)
    params = []
    for _ in range(k):
        a = rng.randrange(1, P)  # non-zero
        b = rng.randrange(0, P)
        params.append((a, b))
    return params


def minhash_signature_affine(ngrams: Iterable[str], k: int, rng_seed: int = 12345) -> List[int]:
    import hashlib
    params = make_ab(k, rng_seed)
    sig = [P] * k
    for ng in ngrams:
        # stable base hash -> integer in [0, P)
        x = int.from_bytes(hashlib.blake2b(ng.encode(), digest_size=8).digest(), "little") % P
        for i, (a, b) in enumerate(params):
            hv = (a * x + b) % P
            if hv < sig[i]:
                sig[i] = hv
    return sig


# Unicode-aware "punctuation / symbols" removal via category checks.
# Categories starting with:
#   P = punctuation, S = symbols
_PUNCT_OR_SYMBOL = {"P", "S"}

_WHITESPACE_RE = re.compile(r"\s+")

def normalize_for_minhash(text: str) -> str:
    """
    Normalization following the description:
    - apply NFD unicode normalization
    - remove accents (combining marks)
    - lowercase
    - remove punctuation
    - normalize whitespace

    Returns a normalized string suitable for n-gram/Jaccard/MinHash.
    """
    if not text:
        return ""

    # 1) NFD normalization (decompose: accents become combining marks)
    text = unicodedata.normalize("NFD", text)

    # 2) Remove accents: drop combining marks (category "Mn")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # 3) Lowercase
    text = text.lower()

    # 4) Remove punctuation (and symbols, which often behave like punctuation)
    #    Keep letters/numbers/whitespace. Replace removed chars with space to avoid concatenating words.
    out_chars = []
    for ch in text:
        cat0 = unicodedata.category(ch)[0]
        if cat0 in _PUNCT_OR_SYMBOL:
            out_chars.append(" ")
        else:
            out_chars.append(ch)
    text = "".join(out_chars)

    # 5) Normalize whitespace: collapse to single spaces and strip ends
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text

def jaccard_similarity(file1, file2, ngrams_length):
    with open(file1, "r") as f:
        text = f.read()
        cleaned_text = normalize_for_minhash(text)
        # Tokenize the text into words
        tokens_1 = word_tokenize(cleaned_text)
        n_grams_1 = ngrams(tokens_1, ngrams_length)
        n_grams_1 = set([" ".join(t) for t in n_grams_1])

    with open(file2, "r") as f:
        text = f.read()
        cleaned_text = normalize_for_minhash(text)
        # Tokenize the text into words
        tokens_2 = word_tokenize(cleaned_text)
        n_grams_2 = ngrams(tokens_2, ngrams_length)
        n_grams_2 = set([" ".join(t) for t in n_grams_2])

    sim = len(n_grams_1 & n_grams_2) / len(n_grams_1 | n_grams_2)
    return sim


def minhash_deduplication(
        input_files: list[os.PathLike],
        num_hashes: int,
        num_bands: int,
        ngrams_length: int,
        jaccard_threshold: float,
        output_directory: os.PathLike,
):
    """
    Run fuzzy minhash deduplication on document level
    Args:
        input_files:
        num_hashes: number of hash functions to use
        num_bands: number of bands in LSH
        ngrams_length: ngrams used to compute jaccard similarity
        jaccard_threshold: threshold for jaccard similarity for two document to match
        output_directory: directory to store output files

    Returns:
    """
    buckets = [defaultdict(list) for _ in range(num_bands)]
    for idx, file in enumerate(input_files):
        with open(file, "r") as f:
            text = f.read()
            cleaned_text = normalize_for_minhash(text)
            # Tokenize the text into words
            tokens = word_tokenize(cleaned_text)
            # Generate n-grams
            n_grams = ngrams(tokens, ngrams_length)
            n_grams = set([" ".join(t) for t in n_grams])
            # print(f"n_grams in file {file} ({idx}): ", n_grams)
            sig = minhash_signature_affine(n_grams, k=num_hashes)
            # assign file to buckets
            band_len = num_hashes // num_bands
            for i in range(num_bands):
                band_sig = tuple(sig[i * band_len: (i+1) * band_len])
                buckets[i][band_sig].append(idx)

    # Use disjoint set to merge duplicates
    ds = DisjointSet(range(len(input_files)))
    for i in range(num_bands):
        for key in buckets[i]:
            if len(buckets[i][key]) > 1:
                for j1 in range(len(buckets[i][key])):
                    for j2 in range(j1+1, len(buckets[i][key])):
                        idx1 = buckets[i][key][j1]
                        idx2 = buckets[i][key][j2]
                        file1 = input_files[idx1]
                        file2 = input_files[idx2]
                        if (ds.find(idx1) != ds.find(idx2) and
                                jaccard_similarity(file1, file2, ngrams_length) > jaccard_threshold):
                            ds.union(idx1, idx2)

    for idx, file in enumerate(input_files):
        if ds.find(idx) == idx:
            shutil.copy(file, output_directory)


if __name__ == "__main__":
    stream = GZipStream(FileStream("CC-MAIN-20250417135010-20250417165010-00065.warc.gz", "rb"))
    cnt = 0
    total_cnt = 0
    for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
        total_cnt += 1
        html_bytes = record.reader.read()
        html_text = extract_text_from_bytestr(html_bytes)
        lan, prob = identify_language(html_text)
        if lan == "en" and prob >= 0.7:
            cnt += 1

    print("Total webpages:", total_cnt)
    print("Filtered webpages in English: ", cnt)

