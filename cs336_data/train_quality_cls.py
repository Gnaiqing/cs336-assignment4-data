import requests
from resiliparse.extract.html2text import extract_plain_text
from typing import Tuple, List
from tqdm import tqdm
from cs336_data.preprocess import gopher_quality_filter, identify_toxic_speech, identify_nsfw, identify_language
import os
import random
import argparse
from fastwarc.warc import ArchiveIterator, WarcRecordType
from fastwarc.stream_io import *
from resiliparse.extract.html2text import extract_plain_text
from cs336_data.preprocess import extract_text_from_bytestr
from pathlib import Path
import fasttext


def train_valid_split(
    positive_file: str,
    negative_file: str,
    train_file: str = "dataquality.train",
    valid_file: str = "dataquality.valid",
    train_ratio: float = 0.7,
    seed: int = 42,
    shuffle_final: bool = True,
):
    random.seed(seed)

    def load_lines(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]

    def split(lines: List[str]) -> Tuple[List[str], List[str]]:
        random.shuffle(lines)
        k = int(len(lines) * train_ratio)
        return lines[:k], lines[k:]

    # Load data
    pos = load_lines(positive_file)
    neg = load_lines(negative_file)

    # Split per class (important!)
    pos_train, pos_valid = split(pos)
    neg_train, neg_valid = split(neg)

    # Combine
    train = pos_train + neg_train
    valid = pos_valid + neg_valid

    if shuffle_final:
        random.shuffle(train)
        random.shuffle(valid)

    # Write output
    Path(train_file).write_text("\n".join(train) + "\n", encoding="utf-8")
    Path(valid_file).write_text("\n".join(valid) + "\n", encoding="utf-8")

    print("Split summary:")
    print(f"  Positive: {len(pos)} → train {len(pos_train)}, valid {len(pos_valid)}")
    print(f"  Negative: {len(neg)} → train {len(neg_train)}, valid {len(neg_valid)}")
    print(f"  Train total: {len(train)}")
    print(f"  Valid total: {len(valid)}")


def extract_text_from_url(url) -> Tuple[bool, str]:
    try:
        response = requests.get(url, timeout=1)
    except Exception as e:
        return False, str(e)

    # Check if the request was successful
    if response.status_code == 200:
        html_content = response.text  # HTML as a string
        plain_text = extract_plain_text(html_content)
        return True, plain_text
    else:
        return False, f"Failed to retrieve content. Status code: {response.status_code}"


def heuristic_check_quality(text) -> Tuple[bool, str]:
    """
    Check the quality of text based on a series of heuristic (language, nsfw, toxic, gopher)
    Args:
        text: HTML text

    Returns:
        passed: if the text pass all heuristic checks
        msg: message containing reason for not passing the checks
    """
    lan, prob = identify_language(text)
    if lan != "en" or prob < 0.7:
        return False, "language identification not in english"

    label, prob = identify_nsfw(text)
    if not (label == "non-nsfw" and prob > 0.99):
        return False, "NSFW check failed"

    label, prob = identify_toxic_speech(text)
    if not (label == "non-toxic" and prob > 0.99):
        return False, "toxic speech check failed"

    passed, msg = gopher_quality_filter(text)
    if passed:
        return True, "all check passed"
    else:
        return False, msg


def sample_high_quality_urls(urls, sample_size=10000):
    candidate_urls = random.sample(urls, sample_size)
    selected_urls = []
    for url in tqdm(candidate_urls):
        url = url.strip()
        success, text = extract_text_from_url(url)
        if success:
            check_passed, msg = heuristic_check_quality(text)
            if check_passed:
                selected_urls.append(url)
                # print(f"url {url} added to list. ")
        #     else:
        #         print(f"url {url} quality check failed: {msg}")
        # else:
        #     print(f"url {url} extraction failed: {text}")

    return selected_urls


def save_high_quality_urls(url_input, url_output, sample_size):
    with open(url_input, "r") as f:
        urls = f.readlines()

    selected_urls = sample_high_quality_urls(urls, sample_size=sample_size)
    print(f"Selected {len(selected_urls)} positive examples")
    with open(url_output, "w") as f:
        f.write("\n".join(selected_urls) + "\n")


def write_negative_samples(
    input_file="CC-MAIN-20250417135010-20250417165010-00065.warc.gz",
    output_file="negative_samples.txt",
    max_records=1500,
    min_prob=0.7,
):
    in_stream = GZipStream(FileStream(input_file, "rb"))

    selected = 0
    total = 0
    negative_samples = []

    for record in ArchiveIterator(in_stream, record_types=WarcRecordType.response):
        total += 1

        # Read payload ONCE
        payload = record.reader.read()

        # Convert to text for language detection
        try:
            text = extract_text_from_bytestr(payload)
        except Exception:
            continue

        if not text.strip():
            continue

        lang, prob = identify_language(text)

        if lang == "en" and prob >= min_prob:
            cleaned_text = text.replace("\n", "")
            negative_samples.append(f"__label__negative {cleaned_text}")
            selected += 1

    in_stream.close()

    if selected > max_records:
        negative_samples = random.sample(negative_samples, k=max_records)
        selected = max_records

    with open(output_file, "w") as f:
        f.write("\n".join(negative_samples) + "\n")

    print(f"Scanned {total} records")
    print(f"Selected {selected} Negative pages")


def write_positive_samples(
        input_file="subsampled_positive_urls.warc.gz",
        output_file="positive_samples.txt"):
    in_stream = GZipStream(FileStream(input_file, "rb"))
    selected = 0
    positive_samples = []
    for record in ArchiveIterator(in_stream, record_types=WarcRecordType.response):
        payload = record.reader.read()
        try:
            text = extract_text_from_bytestr(payload)
        except Exception:
            continue

        cleaned_text = text.replace("\n", "")
        positive_samples.append(f"__label__positive {cleaned_text}")
        selected += 1

    with open(output_file, "w") as f:
        f.write("\n".join(positive_samples) + "\n")

    print(f"Selected {selected} Positive pages")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-input", type=str, default="enwiki-20240420-extracted_urls.txt")
    parser.add_argument("--url-output", type=str, default="subsampled_positive_urls.txt")
    parser.add_argument("--url-sample-size", type=int, default=10000)
    parser.add_argument("--positive-samples", type=str, default="positive_samples.txt")
    parser.add_argument("--negative-samples", type=str, default="negative_samples.txt")

    args = parser.parse_args()
    ## Step 1: select positive webpages
    # if not os.path.exists(args.url_output):
    #     with open(args.url_input, "r") as f:
    #         urls = f.readlines()
    #
    #     selected_urls = sample_high_quality_urls(urls, sample_size=args.url_sample_size)
    #     print(f"Selected {len(selected_urls)} positive examples")
    #     with open(args.url_output, "w") as f:
    #         f.write("\n".join(selected_urls) + "\n")

    # step 2: write positive and negative samples to txt files
    # write_negative_samples(
    #     input_file="CC-MAIN-20250417135010-20250417165010-00065.warc.gz",
    #     output_file="negative_samples.txt",
    #     max_records=2200,
    #     min_prob=0.7)

    # write_positive_samples(
    #     input_file="subsampled_positive_urls.warc.gz",
    #     output_file="positive_samples.txt")

    # train_valid_split(positive_file="positive_samples.txt", negative_file="negative_samples.txt", train_ratio=0.8)

    # step 3: train the fasttext model
    # model = fasttext.train_supervised(input="dataquality.train", lr=1.0, epoch=50, wordNgrams=3)
    # print(model.test("dataquality.valid"))
    # model.save_model("model_dataquality.bin")

    # step 4: test the model
    model = fasttext.FastText.load_model("model_dataquality.bin")
    content = """
Y′UV, also written YUV, is the color model found in the PAL analogue color TV standard. A color is described as a Y′ component (luma) and two chroma components U and V. The prime symbol (') denotes that the luma is calculated from gamma-corrected RGB input and that it is different from true luminance.[1] Today, the term YUV is commonly used in the computer industry to describe colorspaces that are encoded using YCbCr.[2]

In TV formats, color information (U and V) was added separately via a subcarrier so that a black-and-white receiver would still be able to receive and display a color picture transmission in the receiver's native black-and-white format, with no need for extra transmission bandwidth.

As for etymology, Y, Y′, U, and V are not abbreviations. The use of the letter Y for luminance can be traced back to the choice of XYZ primaries. This lends itself naturally to the usage of the same letter in luma (Y′), which approximates a perceptually uniform correlate of luminance. Likewise, U and V were chosen to differentiate the U and V axes from those in other spaces, such as the x and y chromaticity space. See the equations below or compare the historical development of the math.[3][4][5]experience.

Y′UV was invented when engineers wanted color television in a black-and-white infrastructure.[6] They needed a signal transmission method that was compatible with black-and-white (B&W) TV while being able to add color. The luma component already existed as the black and white signal; they added the UV signal to this as a solution.

The UV representation of chrominance was chosen over straight R and B signals because U and V are color difference signals. In other words, the U and V signals tell the television to shift the color of a certain spot without altering its brightness, or to make one color brighter at the cost of the other and by how much it should be shifted. The higher (or the lower when negative) the U and V values are, the more saturated (colorful) the spot gets. The closer the U and V values get to zero, the lesser it shifts the color meaning that the red, green and blue lights will be more equally bright, producing a grayer spot. This is the benefit of using color difference signals, i.e. instead of telling how much red there is to a color, it tells by how much it is more red than green or blue.

In turn this meant that when the U and V signals would be zero or absent, it would just display a grayscale image. If R and B were to have been used, these would have non-zero values even in a B&W scene, requiring all three data-carrying signals. This was important in the early days of color television, because old black and white TV signals had no U and V signals present, meaning the color TV would just display it as B&W TV out of the box. In addition, black and white receivers could take the Y′ signal and ignore the U- and V-color signals, making Y′UV backward-compatible with all existing black-and-white equipment, input and output. If the color-TV standard wouldn't have used color difference signals, it could mean a color TV would make funny colors out of a B&W broadcast or it would need additional circuitry to translate the B&W signal to color.


"""
    content = content.replace("\n", "")
    print(model.predict(content))







