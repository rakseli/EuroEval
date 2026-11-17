"""Create best_lrs.jsonl for launch_final_training_jobs.py from a filtered hyperparameter search file.

launch_final_training_jobs.py skips every (model, dataset) pair found in
euroeval_benchmark_results_spesific_lrs.jsonl, regardless of the learning rate used. Therefore a
copy of that file is written without the results whose learning rate is no longer the best one
(e.g. runs done with the excluded 1e-05), so those pairs get relaunched.

Only models reported in the final result tables are kept: models that have results in
--final_results_file, excluding failed runs (see get_final_models).

Outputs into --output_dir:
    best_lrs.jsonl                                  best lr for every (model, dataset) pair
    euroeval_benchmark_results_spesific_lrs.jsonl   final results that are still valid

Example:
    python scripts/make_final_training_list.py \
        --hp_results_file data/updated_files/euroeval_benchmark_results_hyperparameter_search_no_1e-05.jsonl \
        --final_results_file data/euroeval/euroeval_benchmark_results_spesific_lrs.jsonl \
        --old_best_lrs_file data/euroeval/best_lrs.jsonl \
        --output_dir data/updated_files
"""
import json
import argparse
import math
import os

from get_best_lrs import get_best_learning_rate
from drop_lr_from_results import read_jsonl, get_final_models


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--hp_results_file', required=True, help="hyperparameter search results without excluded lr")
    ap.add_argument('--final_results_file', required=True, help="original euroeval_benchmark_results_spesific_lrs.jsonl")
    ap.add_argument('--old_best_lrs_file', default=None, help="original best_lrs.jsonl, used for sanity checks")
    ap.add_argument('--excluded_lr', default="1e-05", help="learning rate that must not be selected")
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_best = os.path.join(args.output_dir, "best_lrs.jsonl")
    out_final = os.path.join(args.output_dir, "euroeval_benchmark_results_spesific_lrs.jsonl")
    inputs = {os.path.abspath(p) for p in [args.hp_results_file, args.final_results_file, args.old_best_lrs_file] if p}
    for p in [out_best, out_final]:
        assert os.path.abspath(p) not in inputs, f"Refusing to overwrite input file {p}"
        assert args.overwrite or not os.path.exists(p), f"{p} already exists, use --overwrite"

    # only models reported in the final result tables are kept in the outputs
    final_models = get_final_models(args.final_results_file)
    hp_rows = [r for r in read_jsonl(args.hp_results_file) if r["model"] in final_models]
    final_rows = [r for r in read_jsonl(args.final_results_file) if r["model"] in final_models]
    assert hp_rows, "Empty hyperparameter search file"
    missing = final_models - {r["model"] for r in hp_rows}
    assert not missing, f"Final models missing from hp search file: {sorted(missing)}"

    hp = {}
    candidate_lr_sets = set()
    for r in hp_rows:
        key = (r["model"], r["dataset"])
        assert key not in hp, f"Duplicate hp search result {key}"
        assert r["validation_split"] is True, f"hp search result {key} was not evaluated on validation split"
        assert args.excluded_lr not in r["results"], f"{key} still contains lr {args.excluded_lr}"
        assert len(r["dataset_languages"]) == 1, f"{key} has several languages"
        candidate_lr_sets.add(tuple(sorted(r["results"])))
        hp[key] = r
    assert len(candidate_lr_sets) == 1, f"Rows have differing LR sets: {candidate_lr_sets}"

    final = {}
    for r in final_rows:
        key = (r["model"], r["dataset"])
        assert key not in final, f"Duplicate final result {key}"
        assert r["validation_split"] is False, f"final result {key} was evaluated on validation split"
        assert len(r["results"]) == 1, f"final result {key} has {len(r['results'])} learning rates"
        assert key in hp, f"final result {key} has no hp search result"
        assert r["dataset_languages"] == hp[key]["dataset_languages"], f"Language mismatch for {key}"
        final[key] = r

    old_best = None
    if args.old_best_lrs_file:
        old_best = {}
        for r in read_jsonl(args.old_best_lrs_file):
            if r["model"] not in final_models:
                continue
            old_best[(r["model"], r["dataset"])] = r["best_lr"]
        assert set(old_best) == set(hp), "Old best_lrs.jsonl covers different (model, dataset) pairs"

    best_lrs, still_valid = [], []
    n_stale = 0
    for key, r in hp.items():
        best_lr, best_f1 = get_best_learning_rate(r["results"])
        assert best_lr in r["results"], f"{key}: best lr {best_lr} is not a candidate"
        assert best_lr != args.excluded_lr
        assert best_f1 is not None and math.isfinite(best_f1), f"{key}: invalid f1 {best_f1}"
        if old_best is not None and old_best[key] != args.excluded_lr:
            # removing a non-winning candidate must not change the winner
            assert old_best[key] == best_lr, f"{key}: best lr changed {old_best[key]} -> {best_lr}"

        best_lrs.append({
            "dataset": r["dataset"],
            "model": r["model"],
            "language": r["dataset_languages"][0],
            "best_lr": best_lr,
            "mean_f1": best_f1,
        })
        if key in final:
            if list(final[key]["results"])[0] == best_lr:
                still_valid.append(final[key])
            else:
                n_stale += 1

    assert len(best_lrs) == len(hp)
    assert len(still_valid) + n_stale == len(final)
    assert all(list(r["results"])[0] != args.excluded_lr for r in still_valid)

    os.makedirs(args.output_dir, exist_ok=True)
    write_jsonl(out_best, best_lrs)
    write_jsonl(out_final, still_valid)

    print(f"Final models kept: {len(final_models)}")
    print(f"Candidate learning rates: {list(candidate_lr_sets.pop())}")
    print(f"(model, dataset) pairs in best_lrs: {len(best_lrs)}")
    print(f"Final results still valid: {len(still_valid)}")
    print(f"Final results with outdated lr, removed: {n_stale}")
    print(f"Pairs that will be launched: {len(hp) - len(still_valid)}")
    print(f"Wrote {out_best}, {out_final}")
