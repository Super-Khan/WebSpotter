"""
LLM-based Malicious Payload Localization

Replaces the TextCNN + post-hoc explanation pipeline with a direct LLM prompt
that asks the model to label each MSU as malicious (1) or benign (0).

Reuses the existing repo infrastructure for:
- Dataset loading (core.inputter.HTTPDataset)
- MSU splitting (core.preprocess.tokenize_with_alignment)
- Ground truth extraction & metrics (localization.post_explain.metrics)

Usage:
    python localization/llm_based/run_llm_explain.py \
        --dataset fpad \
        --test_path datasets/FPAD/test.jsonl \
        --outputdir explain_result/FPAD/llm \
        --model claude-sonnet-4-20250514
"""

import sys
import os
import json
import time
import argparse
import re
from urllib.parse import unquote_plus
from tqdm import tqdm

sys.path.append(".")

from core.inputter import HTTPDataset, RequestInfo
from core.preprocess.tokenize_with_alignment import (
    get_http_level_split,
    get_http_level_split_furl,
    get_http_level_split_furl_header,
)
from localization.post_explain.metrics import (
    check_payload_accuracy_fpad,
    check_payload_accuracy_pkdd,
    check_payload_accuracy_csic,
    check_payload_accuracy_poc,
    evaluate_dataset_performance,
    get_ground_truth_fpad,
    get_ground_truth_pkdd,
    get_ground_truth_csic,
    get_ground_truth_poc,
)

# ---------------------------------------------------------------------------
# Prompt template (from the paper)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Your task is to locate malicious payloads in an HTTP request. "
    "The HTTP request will be divided into minimal semantic units (MSUs), "
    "and the input is an array of strings, where each string represents an MSU "
    "of the HTTP request. The output should be a dictionary in JSON format, "
    "where the key is a string from the input array, and the value is 0 or 1. "
    "A value of 0 indicates that the unit does not contain malicious payloads, "
    "and a value of 1 means otherwise."
)

FEW_SHOT_EXAMPLES = [
    {
        "input": [
            "POST", "/tienda1", "/<marquee loop=1 width=0 onfinish=alert(1)>",
            "/anadir.jsp", "id=2", "nombre=Iber", "precio=5003",
            "cantidad=64", "B1=Entrar",
        ],
        "output": {
            "POST": 0, "/tienda1": 0,
            "/<marquee loop=1 width=0 onfinish=alert(1)>": 1,
            "/anadir.jsp": 0, "id=2": 0, "nombre=Iber": 0,
            "precio=5003": 0, "cantidad=64": 0, "B1=Entrar": 0,
        },
    },
]


def build_prompt(msu_list: list[str]) -> list[dict]:
    """Build the messages list for the LLM API call."""
    examples_text = ""
    for idx, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += f"\nExample-{idx}:\n"
        examples_text += f"Input: {json.dumps(ex['input'])}\n"
        examples_text += f"Output: {json.dumps(ex['output'])}\n"

    user_content = (
        f"{SYSTEM_PROMPT}\n"
        f"{examples_text}\n"
        f"User Input: {json.dumps(msu_list)}\n"
    )

    return [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# MSU splitting (reuses repo functions, selects by dataset)
# ---------------------------------------------------------------------------
def get_msu_splitter(dataset: str):
    """Return the appropriate MSU splitting function for the dataset.

    The raw split functions return URL-encoded MSUs.  The normal pipeline
    decodes them via ``unquote_plus`` inside the tokenizer (e.g.
    ``char_tokenizer_with_http_level_alignment_furl_header``).  All
    downstream metric functions expect decoded text, so we wrap the
    splitter to apply the same decoding step.
    """
    if dataset == "pkdd":
        raw_splitter = get_http_level_split_furl_header
    else:
        raw_splitter = get_http_level_split

    def _split_and_decode(req):
        return [unquote_plus(msu, encoding="utf-8", errors="replace")
                for msu in raw_splitter(req)]

    return _split_and_decode


# ---------------------------------------------------------------------------
# Ground truth + metrics (reuses repo functions)
# ---------------------------------------------------------------------------
def analyze_attacks_accuracy(dataset_type, data_json, suspected_attacks, total_num):
    """
    Wrapper around the repo's per-dataset check_payload_accuracy_* functions.
    suspected_attacks: list of (msu_text, score) tuples for MSUs predicted as malicious.
    """
    if dataset_type in ("fpad", "cve"):
        if "attacks" in data_json:
            attacks = data_json["attacks"]
            metrics, ulocation = check_payload_accuracy_poc(attacks, suspected_attacks, total_num)
        else:
            attacks = data_json.get("location", [])
            metrics, ulocation = check_payload_accuracy_fpad(attacks, suspected_attacks, total_num)
    elif dataset_type == "pkdd":
        attacks = data_json.get("attacks", [])
        metrics, ulocation = check_payload_accuracy_pkdd(attacks, suspected_attacks, total_num)
    elif dataset_type == "csic":
        attacks = data_json.get("attacks", [])
        metrics, ulocation = check_payload_accuracy_csic(attacks, suspected_attacks, total_num)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    precision, recall, f1_score, accuracy, jaccard_index = metrics
    return precision, recall, f1_score, accuracy, jaccard_index, ulocation, attacks


def get_location_ground_truth(dataset_type, data_json, msu_list):
    """Wrapper around the repo's per-dataset get_ground_truth_* functions."""
    if dataset_type in ("fpad", "cve"):
        if "attacks" in data_json:
            return get_ground_truth_poc(data_json["attacks"], msu_list)
        else:
            return get_ground_truth_fpad(data_json.get("location", []), msu_list)
    elif dataset_type == "pkdd":
        return get_ground_truth_pkdd(data_json.get("attacks", []), msu_list)
    elif dataset_type == "csic":
        return get_ground_truth_csic(data_json.get("attacks", []), msu_list)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def call_llm(messages, model, max_retries=5):
    """Call the Anthropic API and return the text response."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=messages,
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            print(f"  Rate limited, retrying in {wait}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            wait = 2 ** (attempt + 1)
            print(f"  API error ({e}), retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError("Max retries exceeded for LLM API call")


def parse_llm_response(response_text: str, msu_list: list[str]) -> dict:
    """
    Parse the LLM JSON response into a dict mapping MSU -> 0/1.
    Falls back to extracting the first JSON object if the response contains
    extra text around it.
    """
    # Try direct parse first
    try:
        result = json.loads(response_text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from the response text
    match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Try extracting a potentially large/nested JSON block
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(response_text[start : end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Last resort: return all MSUs as 0
    print(f"  WARNING: Could not parse LLM response, defaulting all to 0")
    return {msu: 0 for msu in msu_list}


# ---------------------------------------------------------------------------
# Result writing (mirrors the format in run_explain.py)
# ---------------------------------------------------------------------------
def write_result(file, original_text, msu_list, predictions, ground_truth_labels,
                 precision, recall, f1_score, accuracy, jaccard_index, attacks, ulocation):
    file.write(f"Original text: {original_text}\n")
    file.write("MSU Predictions:\n")
    for msu, pred, gt in zip(msu_list, predictions, ground_truth_labels):
        file.write(f"  MSU: {msu}, Predicted: {pred}, GroundTruth: {gt}\n")
    file.write(f"Attacks: {attacks}\n")
    file.write(f"Used Locations: {ulocation}\n")
    file.write(
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
        f"F1 Score: {f1_score:.4f}, Accuracy: {accuracy:.4f}, "
        f"Jaccard Index: {jaccard_index:.4f}\n"
    )
    file.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLM-based malicious payload localization"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["csic", "pkdd", "fpad", "cve"])
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--outputdir", type=str, required=True)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Anthropic model ID")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of malicious samples to evaluate (for cost control)")
    args = parser.parse_args()

    dataset_type = args.dataset
    test_path = args.test_path
    outputdir = args.outputdir

    os.makedirs(outputdir, exist_ok=True)

    # ---- Load dataset (reuses repo infrastructure) ----
    with open(test_path, "r") as f:
        test_data_json = [json.loads(line) for line in f]

    test_dataset = HTTPDataset.load_from(test_path)
    test_labels = [req.label for req in test_dataset]

    # ---- Select MSU splitter ----
    msu_splitter = get_msu_splitter(dataset_type)

    # ---- Evaluation accumulators ----
    all_precision = []
    all_recall = []
    all_f1_scores = []
    all_accuracy = []
    all_jaccard_index = []
    all_exact_match = []

    total_count = 0
    total_time_start = time.time()

    result_file = os.path.join(outputdir, "results.txt")
    jsonl_output_file = os.path.join(outputdir, "predictions.jsonl")

    with open(result_file, "w") as rfile, open(jsonl_output_file, "w") as jfile:
        for i in tqdm(range(len(test_dataset)), desc="LLM Localization"):
            # ---- Split into MSUs (reuses repo functions) ----
            req = test_dataset[i]
            msu_list = msu_splitter(req)

            # ---- Build prompt & call LLM ----
            messages = build_prompt(msu_list)
            response_text = call_llm(messages, args.model)

            # ---- Parse LLM response ----
            prediction_dict = parse_llm_response(response_text, msu_list)

            # Build prediction list aligned with msu_list
            predictions = []
            for msu in msu_list:
                pred = prediction_dict.get(msu, 0)
                # Normalise to int 0/1
                try:
                    pred = int(pred)
                except (ValueError, TypeError):
                    pred = 0
                predictions.append(1 if pred else 0)

            label = test_labels[i]

            # ---- Get ground truth for every sample ----
            # Benign samples: all MSUs are 0. Malicious: use dataset annotations.
            if label == 0:
                ground_truth_labels = [0] * len(msu_list)
            else:
                ground_truth_labels = get_location_ground_truth(
                    dataset_type, test_data_json[i], msu_list
                )

            # ---- Exact match: correct only if every MSU prediction matches GT ----
            exact_match = int(predictions == ground_truth_labels)
            all_exact_match.append(exact_match)

            # ---- Write JSONL output for every sample ----
            jfile.write(json.dumps({
                "index": i,
                "label": label,
                "msu_list": msu_list,
                "predictions": predictions,
                "ground_truth": ground_truth_labels,
                "exact_match": exact_match,
            }) + "\n")

            # ---- Per-sample localization metrics only for malicious samples ----
            if label == 0:
                continue

            # Convert predictions to suspected_attacks format
            # The repo's check_payload_accuracy_* expects list of (msu_text, score)
            suspected_attacks = [(msu, 1.0) for msu, pred in zip(msu_list, predictions) if pred == 1]

            precision, recall, f1_score, accuracy, jaccard_index, ulocation, attacks = (
                analyze_attacks_accuracy(dataset_type, test_data_json[i],
                                        suspected_attacks, len(msu_list))
            )

            all_precision.append(precision)
            all_recall.append(recall)
            all_f1_scores.append(f1_score)
            all_accuracy.append(accuracy)
            all_jaccard_index.append(jaccard_index)

            # ---- Write detailed result for malicious samples ----
            original_text = f"Method:{req.method} URL:{req.url} Body:{req.body}".strip()
            write_result(rfile, original_text, msu_list, predictions,
                         ground_truth_labels, precision, recall, f1_score,
                         accuracy, jaccard_index, attacks, ulocation)

            total_count += 1

            if args.max_samples and total_count >= args.max_samples:
                print(f"\nReached max_samples limit ({args.max_samples}), stopping.")
                break

    # ---- Report aggregate metrics ----
    total_time = time.time() - total_time_start

    total_samples = len(all_exact_match)
    overall_exact_match = sum(all_exact_match) / total_samples if total_samples > 0 else 0

    print(f"\n{'='*60}")
    print(f"LLM-based Localization Results")
    print(f"{'='*60}")
    print(f"Dataset:          {dataset_type.upper()}")
    print(f"Model:            {args.model}")
    print(f"Total samples:    {total_samples} (benign + malicious)")
    print(f"Malicious samples:{total_count}")
    print(f"Total time:       {total_time:.2f}s")
    if total_samples > 0:
        print(f"Avg time/sample:  {total_time / total_samples:.2f}s")

    # Overall exact match accuracy (all samples: correct iff every MSU matches GT)
    print(f"\nOverall Exact Match Accuracy: {overall_exact_match:.4f} ({sum(all_exact_match)}/{total_samples})")

    if total_count > 0:
        avg_precision, avg_recall, avg_f1, avg_acc, avg_jaccard = (
            evaluate_dataset_performance(
                all_precision, all_recall, all_f1_scores,
                all_accuracy, all_jaccard_index
            )
        )
        print(f"\nMalicious-only localization metrics:")
        print(f"  Avg Precision:    {avg_precision:.4f}")
        print(f"  Avg Recall:       {avg_recall:.4f}")
        print(f"  Avg F1 Score:     {avg_f1:.4f}")
        print(f"  Avg Accuracy:     {avg_acc:.4f}")
        print(f"  Avg Jaccard:      {avg_jaccard:.4f}")

        # Write summary
        summary_file = os.path.join(outputdir, "summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"Dataset: {dataset_type.upper()}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Malicious samples: {total_count}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Overall Exact Match Accuracy: {overall_exact_match:.4f} ({sum(all_exact_match)}/{total_samples})\n")
            f.write(f"Avg Precision: {avg_precision:.4f}\n")
            f.write(f"Avg Recall: {avg_recall:.4f}\n")
            f.write(f"Avg F1 Score: {avg_f1:.4f}\n")
            f.write(f"Avg Accuracy: {avg_acc:.4f}\n")
            f.write(f"Avg Jaccard: {avg_jaccard:.4f}\n")
        print(f"\nResults saved to: {outputdir}")
    else:
        print("No malicious samples found in the dataset.")


if __name__ == "__main__":
    main()
