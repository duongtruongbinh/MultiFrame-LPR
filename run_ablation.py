#!/usr/bin/env python3
"""Automation script to run ablation experiments for ResTran/CRNN models."""

import os
import subprocess
import sys
from typing import Any, Dict, List, Optional


def build_command(exp: Dict[str, Any], output_dir: str = "experiments") -> List[str]:
    """Build the python3 train.py command from an experiment spec."""
    cmd: List[str] = [sys.executable or "python3", "train.py"]

    if "experiment_name" in exp:
        cmd += ["-n", str(exp["experiment_name"])]
    if "model" in exp:
        cmd += ["-m", str(exp["model"])]
    if "resnet_layers" in exp:
        cmd += ["--resnet-layers", str(exp["resnet_layers"])]
    if exp.get("no_stn", False):
        cmd += ["--no-stn"]
    if "aug_level" in exp:
        cmd += ["--aug-level", str(exp["aug_level"])]
    
    cmd += ["--output-dir", output_dir]
    
    for flag in exp.get("extra_flags", []):
        cmd.append(str(flag))

    return cmd


def _parse_best_accuracy(log_path: str) -> Optional[float]:
    """Parse best validation accuracy from log file."""
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                for pattern in ["Best Val Acc:", "Best accuracy:"]:
                    if pattern in line:
                        try:
                            token = line.split(pattern)[1].strip().split("%")[0]
                            return float(token)
                        except (ValueError, IndexError):
                            continue
    except FileNotFoundError:
        pass
    return None


def main() -> None:
    experiments_dir = "experiments"
    os.makedirs(experiments_dir, exist_ok=True)

    experiments: List[Dict[str, Any]] = [
        {
            "name": "restran_base",
            "experiment_name": "restran_base",
            "model": "restran",
            "resnet_layers": 18,
            "no_stn": False,
            "aug_level": "full",
        },
        {
            "name": "restran_no_stn",
            "experiment_name": "restran_no_stn",
            "model": "restran",
            "resnet_layers": 18,
            "no_stn": True,
            "aug_level": "full",
        },
        {
            "name": "restran_r34",
            "experiment_name": "restran_r34",
            "model": "restran",
            "resnet_layers": 34,
            "no_stn": False,
            "aug_level": "full",
        },
        {
            "name": "restran_light_aug",
            "experiment_name": "restran_light_aug",
            "model": "restran",
            "resnet_layers": 18,
            "no_stn": False,
            "aug_level": "light",
        },
        {
            "name": "crnn_base",
            "experiment_name": "crnn_base",
            "model": "crnn",
            "aug_level": "full",
        },
    ]

    results_summary: List[Dict[str, Any]] = []

    for exp in experiments:
        exp_name = exp["name"]
        log_path = os.path.join(experiments_dir, f"{exp_name}.log")
        cmd = build_command(exp, experiments_dir)

        print(f"\n=== Running experiment: {exp_name} ===")
        print("Command:", " ".join(cmd))

        try:
            with open(log_path, "w") as log_file:
                process = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )

            if process.returncode != 0:
                print(f"[{exp_name}] FAILED with return code {process.returncode}. See {log_path}")
                results_summary.append(
                    {"name": exp_name, "best_acc": None}
                )
                continue

            print(f"[{exp_name}] COMPLETED successfully. Log: {log_path}")

            # Parse best accuracy from log
            best_acc = _parse_best_accuracy(log_path)

            results_summary.append(
                {"name": exp_name, "best_acc": best_acc}
            )

        except Exception as e:
            print(f"[{exp_name}] ERROR while running experiment: {e}")
            results_summary.append(
                {"name": exp_name, "best_acc": None}
            )

    # Print and save summary table
    if results_summary:
        summary_lines = []
        summary_lines.append("=== Ablation Summary ===")
        header = f"{'Experiment':20s} | {'Best Acc (%)':12s}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        for row in results_summary:
            name = str(row["name"])
            best_acc = (
                f"{row['best_acc']:.2f}"
                if isinstance(row.get("best_acc"), (int, float))
                else "N/A"
            )
            summary_lines.append(f"{name:20s} | {best_acc:12s}")
        
        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)
        
        # Save to file
        summary_file = os.path.join(experiments_dir, "ablation_summary.txt")
        with open(summary_file, "w") as f:
            f.write(summary_text + "\n")
        print(f"\n📝 Summary saved to: {summary_file}")
        print(
            f"\nLogs for each experiment are stored under '{experiments_dir}/'. "
            "You can inspect them for detailed training curves and metrics."
        )


if __name__ == "__main__":
    main()

