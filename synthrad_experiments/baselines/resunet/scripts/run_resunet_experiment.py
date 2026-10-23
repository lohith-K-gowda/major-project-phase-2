# -*- coding: utf-8 -*-
"""
run_resunet_experiment.py - Master Orchestrator for ResUNet Baseline
Executes:
1. Environment and configuration verification
2. Supervised training across verified train patients
3. Frozen evaluation on isolated test patients
4. CSV metric export and figure generation
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline for ResUNet Baseline")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to synthrad_png dataset")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--skip_train", action="store_true", help="Skip training and run evaluation only")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.abspath(os.path.join(base_dir, ".."))
    
    print("=" * 80)
    print(" SYNTHRAD2023 RESUNET BASELINE EXPERIMENT RUNNER ")
    print("=" * 80)
    print(f"Experiment Directory: {exp_dir}")
    
    # Step 1: Training
    if not args.skip_train:
        print("\n>>> STEP 1: Running ResUNet Training...")
        train_cmd = [
            sys.executable,
            os.path.join(base_dir, "train.py"),
            "--exp_dir", exp_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr)
        ]
        if args.data_dir:
            train_cmd += ["--data_dir", args.data_dir]
            
        ret = subprocess.run(train_cmd)
        if ret.returncode != 0:
            print("[ERROR] Training terminated with errors.")
            return ret.returncode

    # Step 2: Frozen Test Evaluation
    print("\n>>> STEP 2: Running Frozen Test Evaluation on Held-Out Patients...")
    eval_cmd = [
        sys.executable,
        os.path.join(base_dir, "evaluate_test.py"),
        "--exp_dir", exp_dir,
        "--checkpoint", os.path.join(exp_dir, "checkpoints", "best_model.pth")
    ]
    if args.data_dir:
        eval_cmd += ["--data_dir", args.data_dir]
        
    ret = subprocess.run(eval_cmd)
    if ret.returncode != 0:
        print("[ERROR] Evaluation terminated with errors.")
        return ret.returncode

    print("\n" + "=" * 80)
    print(" RESUNET EXPERIMENT COMPLETED SUCCESSFULLY ")
    print("=" * 80)

if __name__ == "__main__":
    main()
