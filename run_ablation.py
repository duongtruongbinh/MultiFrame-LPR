#!/usr/bin/env python3
"""Automation script to run ablation experiments for STN impact analysis (CRNN vs ResTran34)."""

import os
import subprocess
import sys
from typing import Any, Dict, List, Optional


def build_command(experiment_config: Dict[str, Any], output_dir: str = "experiments") -> List[str]:
    """Build the python3 train.py command from an experiment configuration.
    
    Args:
        experiment_config: Dictionary containing experiment parameters.
        output_dir: Directory to save experiment outputs.
    
    Returns:
        List of command line arguments.
    """
    cmd: List[str] = [sys.executable or "python3", "train.py"]

    # Basic arguments
    if "experiment_name" in experiment_config:
        cmd += ["-n", str(experiment_config["experiment_name"])]
    if "model" in experiment_config:
        cmd += ["-m", str(experiment_config["model"])]
    if "aug_level" in experiment_config:
        cmd += ["--aug-level", str(experiment_config["aug_level"])]
    
    # Always set output directory
    cmd += ["--output-dir", output_dir]
    
    # Handle extra flags (like --no-stn)
    for flag in experiment_config.get("extra_flags", []):
        cmd.append(str(flag))

    return cmd


def _parse_best_accuracy(log_path: str) -> Optional[float]:
    """Parse best validation accuracy from log file.
    
    Args:
        log_path: Path to the log file.
    
    Returns:
        Best validation accuracy as float, or None if not found.
    """
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                for pattern in ["Best Val Acc:", "Best accuracy:", "Best Val Acc:"]:
                    if pattern in line:
                        try:
                            token = line.split(pattern)[1].strip().split("%")[0]
                            return float(token)
                        except (ValueError, IndexError):
                            continue

                if "Training complete! Best Val Acc:" in line:
                    try:
                        token = line.split("Best Val Acc:")[1].strip().split("%")[0]
                        return float(token)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        pass
    return None


def main() -> None:
    experiments_dir = "experiments"
    os.makedirs(experiments_dir, exist_ok=True)

    # Define the 4 experiments for STN ablation study
    # Default behavior uses STN; pass "--no-stn" to disable it.
    experiments: List[Dict[str, Any]] = [
        # 1. CRNN without STN
        {
            "name": "crnn_no_stn",
            "experiment_name": "crnn_no_stn",
            "model": "crnn",
            "aug_level": "full",
            "extra_flags": ["--no-stn"]  # Flag to disable STN
        },
        # 2. CRNN with STN (Baseline)
        {
            "name": "crnn_with_stn",
            "experiment_name": "crnn_with_stn",
            "model": "crnn",
            "aug_level": "full",
            # No extra flags -> Default uses STN
        },
        # 3. ResTran34 without STN
        {
            "name": "restran34_no_stn",
            "experiment_name": "restran34_no_stn",
            "model": "restran",
            "aug_level": "full",
            "extra_flags": ["--no-stn"]  # Flag to disable STN
        },
        # 4. ResTran34 with STN
        {
            "name": "restran34_with_stn",
            "experiment_name": "restran34_with_stn",
            "model": "restran",
            "aug_level": "full",
            # No extra flags -> Default uses STN
        },
    ]

    results_summary: List[Dict[str, Any]] = []

    for experiment_config in experiments:
        experiment_name = experiment_config["name"]
        log_path = os.path.join(experiments_dir, f"{experiment_name}.log")
        cmd = build_command(experiment_config, experiments_dir)

        print(f"\n=== Running experiment: {experiment_name} ===")
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
                print(f"[{experiment_name}] FAILED with return code {process.returncode}. See {log_path}")
                results_summary.append(
                    {"name": experiment_name, "best_acc": None}
                )
                continue

            print(f"[{experiment_name}] COMPLETED successfully. Log: {log_path}")

            # Parse best accuracy from log
            best_accuracy = _parse_best_accuracy(log_path)

            results_summary.append(
                {"name": experiment_name, "best_acc": best_accuracy}
            )

        except Exception as e:
            print(f"[{experiment_name}] ERROR while running experiment: {e}")
            results_summary.append(
                {"name": experiment_name, "best_acc": None}
            )

    # Print and save summary table
    if results_summary:
        summary_lines = []
        summary_lines.append("=== Ablation Summary (STN Impact) ===")
        header = f"{'Experiment':25s} | {'Best Acc (%)':12s}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        for row in results_summary:
            name = str(row["name"])
            best_acc = (
                f"{row['best_acc']:.2f}"
                if isinstance(row.get("best_acc"), (int, float))
                else "N/A"
            )
            summary_lines.append(f"{name:25s} | {best_acc:12s}")
        
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