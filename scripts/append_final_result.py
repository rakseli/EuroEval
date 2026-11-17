"""Append the result of one finished final training run to the master results file.

launch_final_training_jobs.py runs this at the end of every job, after euroeval has succeeded.
Each job writes its result to euroeval_benchmark_results_spesific_lrs.jsonl in its own run
directory. This script copies the result for --model / --dataset / --lr from there to the master
file. Several jobs can finish at the same time, so the master file is updated while holding an
exclusive flock on a separate lock file. The result is not appended if the master file already
has a result for the (model, dataset) pair. If that result is the same one, the script exits 0, so
it is safe to run again; if it differs, the script exits 1 so the run directory is kept.

Uses only the standard library, so it runs with any python3 (e.g. inside the euroeval container).

Example (backfilling a run that finished before this script existed):
    python3 scripts/append_final_result.py \
        --run_dir scripts/TurkuNLP_finnish-modernbert-tiny-lang-fi-dataset-scandisent-fi-lr-4e-05-final-training \
        --master_file data/updated_files/euroeval_benchmark_results_spesific_lrs.jsonl \
        --model TurkuNLP/finnish-modernbert-tiny --dataset scandisent-fi --lr 4e-05
"""
import argparse
import fcntl
import json
import os
import sys

RESULTS_FILE_NAME = "euroeval_benchmark_results_spesific_lrs.jsonl"


def read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for i, l in enumerate(f, start=1):
            if not l.strip():
                continue
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError as e:
                raise AssertionError(f"{path} line {i} is not valid JSON: {e}")
    return rows


def get_run_result(run_dir, model, dataset, lr):
    path = os.path.join(run_dir, RESULTS_FILE_NAME)
    assert os.path.isfile(path), f"Run results file {path} not found"
    matches = [r for r in read_jsonl(path)
               if r["model"] == model and r["dataset"] == dataset and lr in r["results"]]
    assert matches, f"No result for {model} / {dataset} / lr {lr} in {path}"
    # a rerun in the same run directory appends a new line, use the latest one
    r = matches[-1]
    assert r["validation_split"] is False, f"Result in {path} was evaluated on the validation split"
    assert list(r["results"]) == [lr], f"Result in {path} has learning rates {list(r['results'])}, expected [{lr}]"
    total = r["results"][lr].get("total")
    assert total, f"Result in {path} has no total scores"
    return r


def append_result(master_file, result):
    key = (result["model"], result["dataset"])
    line = json.dumps(result, ensure_ascii=False) + "\n"
    lock_path = master_file + ".lock"
    with open(lock_path, "a") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            existing = read_jsonl(master_file) if os.path.exists(master_file) else []
            matching = [r for r in existing if (r["model"], r["dataset"]) == key]
            if matching:
                if result in matching:
                    print(f"{master_file} already has this result for {key[0]} / {key[1]}, nothing to append")
                    return True
                print(f"{master_file} already has a different result for {key[0]} / {key[1]}, not appending",
                      file=sys.stderr)
                return False
            with open(master_file, "a") as f:
                # add a newline if the file does not end with one, so the new line stays valid JSON
                if f.tell() > 0:
                    with open(master_file, "rb") as rf:
                        rf.seek(-1, os.SEEK_END)
                        if rf.read(1) != b"\n":
                            f.write("\n")
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    print(f"Appended result for {key[0]} / {key[1]} to {master_file}")
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True, help=f"run directory containing {RESULTS_FILE_NAME}")
    ap.add_argument('--master_file', required=True, help="master final results file to append to")
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--lr', required=True, help="learning rate key, e.g. 4e-05")
    args = ap.parse_args()

    assert os.path.abspath(os.path.join(args.run_dir, RESULTS_FILE_NAME)) != os.path.abspath(args.master_file), \
        "Run results file and master file must differ"
    result = get_run_result(args.run_dir, args.model, args.dataset, args.lr)
    # a non-zero exit stops the job script before it removes the run directory
    sys.exit(0 if append_result(args.master_file, result) else 1)
