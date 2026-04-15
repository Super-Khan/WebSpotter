"""
Evaluate a trained TextCNN detection model on a test set and output
per-sample predictions.

Reuses the existing repo infrastructure for model loading and data
preparation.

Usage:
    python classification/eval_detection.py \
        --model_path tmp_model/textcnn-700-FPAD-512-None-0.pth \
        --test_path datasets/FPAD/test.jsonl \
        --outputdir explain_result/FPAD/detection
"""

import sys
import os
import json
import argparse

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.append(".")

from classification.utils import load_model, get_test_dataset_dataloader
from core.inputter import HTTPDataset


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained TextCNN detection model on test set"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model checkpoint (.pth)")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--outputdir", type=str, required=True,
                        help="Directory to save per-sample results")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.makedirs(args.outputdir, exist_ok=True)

    # ---- Load model ----
    net, word2id, id2word, model_args, n_class = load_model(args.model_path)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # ---- Load test data ----
    testloader = get_test_dataset_dataloader(args.test_path, word2id, model_args)

    # Also load raw JSON for original request text
    test_dataset = HTTPDataset.load_from(args.test_path)

    # ---- Run inference ----
    all_ground_truth = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for data in testloader:
            text, labels = data[0].to(device), data[1].to(device)
            outputs = net(text)
            probs = F.softmax(outputs, dim=1)
            predicted = outputs.argmax(dim=1)

            all_ground_truth.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    # ---- Write per-sample JSONL ----
    jsonl_path = os.path.join(args.outputdir, "detection_results.jsonl")
    with open(jsonl_path, "w") as f:
        for i in range(len(all_predictions)):
            req = test_dataset[i]
            f.write(json.dumps({
                "index": i,
                "method": req.method,
                "url": req.url,
                "body": req.body,
                "ground_truth": all_ground_truth[i],
                "prediction": all_predictions[i],
                "correct": int(all_predictions[i] == all_ground_truth[i]),
                "probabilities": all_probs[i],
            }) + "\n")

    # ---- Print aggregate report ----
    # Per-class report
    target_names = [f"Class {i}" for i in range(n_class)]
    report_str = classification_report(
        all_ground_truth, all_predictions,
        zero_division=0, labels=list(range(n_class)),
        target_names=target_names, digits=4,
    )
    print(report_str)

    # Binary accuracy (benign=0 vs any malicious>0)
    bin_gt = [1 if label > 0 else 0 for label in all_ground_truth]
    bin_pred = [1 if label > 0 else 0 for label in all_predictions]
    bin_acc = accuracy_score(bin_gt, bin_pred)
    bin_f1 = f1_score(bin_gt, bin_pred, average="binary")
    print(f"Binary Accuracy: {bin_acc:.4f}")
    print(f"Binary F1:       {bin_f1:.4f}")

    # Overall
    total = len(all_predictions)
    correct = sum(1 for p, g in zip(all_predictions, all_predictions) if p == g)
    correct = sum(1 for p, g in zip(all_predictions, all_ground_truth) if p == g)
    print(f"\nTotal: {total}, Correct: {correct}, Accuracy: {correct/total:.4f}")
    print(f"\nPer-sample results saved to: {jsonl_path}")

    # ---- Write summary ----
    summary_path = os.path.join(args.outputdir, "detection_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test set: {args.test_path}\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {correct/total:.4f}\n")
        f.write(f"Binary Accuracy: {bin_acc:.4f}\n")
        f.write(f"Binary F1: {bin_f1:.4f}\n")
        f.write(f"\n{report_str}")


if __name__ == "__main__":
    main()
