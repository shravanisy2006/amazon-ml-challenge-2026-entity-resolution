"""
run_all.py — run the whole pipeline, data -> output/, in order.

    python run_all.py            # everything (steps 1-8)
    python run_all.py --from 4   # resume from step 4 (earlier outputs are kept in work/)

Stops at the first failing step and says which one, so you can fix it and
resume with --from.
"""
import argparse, subprocess, sys, time

STEPS = [
    ("clean training data",          ["preprocess.py", "--split", "train"]),
    ("clean test data",              ["preprocess.py", "--split", "test"]),
    ("training short lists",         ["blocking.py", "--split", "train", "--sample-partitions", "0.3"]),
    ("training features",            ["features.py", "--split", "train"]),
    ("train model + tune threshold", ["train.py"]),
    ("test short lists",             ["blocking.py", "--split", "test"]),
    ("test features",                ["features.py", "--split", "test"]),
    ("final answers",                ["predict.py"]),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1, help="first step to run (1-8)")
    ap.add_argument("--to", dest="end", type=int, default=len(STEPS), help="last step to run")
    a = ap.parse_args()
    t_all = time.time()
    for n, (name, cmd) in enumerate(STEPS, 1):
        if n < a.start or n > a.end:
            continue
        print(f"\n===== STEP {n}/{len(STEPS)}: {name} =====", flush=True)
        t0 = time.time()
        if subprocess.run([sys.executable] + cmd).returncode != 0:
            print(f"\n!!!!! STEP {n} ({name}) FAILED. Fix the error above, then run:"
                  f"  python run_all.py --from {n}")
            sys.exit(1)
        print(f"===== step {n} finished in {(time.time()-t0)/60:.1f} min =====", flush=True)
    print(f"\nALL DONE in {(time.time()-t_all)/60:.1f} min. Answer files are in the output folder.")

if __name__ == "__main__":
    main()
