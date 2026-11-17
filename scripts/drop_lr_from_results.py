"""Create a copy of a EuroEval hyperparameter search results file without one learning rate.

Example:
    python scripts/drop_lr_from_results.py \
        --results_file data/euroeval/euroeval_benchmark_results_hyperparameter_search.jsonl \
        --output_file data/updated_files/euroeval_benchmark_results_hyperparameter_search_no_1e-05.jsonl \
        --drop_lr 1e-05 \
        --final_results_file data/euroeval/euroeval_benchmark_results_spesific_lrs.jsonl
"""
import json
import argparse
import os


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


def get_final_models(final_results_file):
    """Models reported in the final result tables: those with final results, excluding failed runs
    (create_latex_tables.py skips models with "failed" in the name)."""
    models = {r["model"] for r in read_jsonl(final_results_file) if "failed" not in r["model"]}
    assert models, f"No final models found in {final_results_file}"
    return models


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_file', required=True, help="original hyperparameter search results file")
    ap.add_argument('--output_file', required=True, help="new results file without --drop_lr")
    ap.add_argument('--drop_lr', default="1e-05", help="learning rate key to remove, e.g. 1e-05")
    ap.add_argument('--final_results_file', default=None,
                    help="if given, keep only models that appear in the final result tables")
    ap.add_argument('--overwrite', action='store_true', help="allow overwriting --output_file")
    args = ap.parse_args()

    assert os.path.abspath(args.results_file) != os.path.abspath(args.output_file), \
        "Output file must differ from the original results file"
    assert args.overwrite or not os.path.exists(args.output_file), \
        f"{args.output_file} already exists, use --overwrite"

    rows = read_jsonl(args.results_file)
    assert rows, f"No results found in {args.results_file}"

    pairs = [(r["model"], r["dataset"]) for r in rows]
    assert len(pairs) == len(set(pairs)), "Duplicate (model, dataset) pairs in results file"

    if args.final_results_file:
        final_models = get_final_models(args.final_results_file)
        missing = final_models - {r["model"] for r in rows}
        assert not missing, f"Final models missing from results file: {sorted(missing)}"
        n_before = len(rows)
        rows = [r for r in rows if r["model"] in final_models]
        print(f"Kept {len(final_models)} final models, dropped {n_before - len(rows)} rows of other models")

    remaining_lr_sets = set()
    new_rows = []
    for r in rows:
        assert args.drop_lr in r["results"], \
            f"{r['model']} / {r['dataset']} has no results for lr {args.drop_lr}"
        new_r = dict(r)
        new_r["results"] = {lr: res for lr, res in r["results"].items() if lr != args.drop_lr}
        assert new_r["results"], f"{r['model']} / {r['dataset']} has no results left after dropping"
        assert len(new_r["results"]) == len(r["results"]) - 1
        remaining_lr_sets.add(tuple(sorted(new_r["results"])))
        new_rows.append(new_r)

    assert len(remaining_lr_sets) == 1, f"Rows have differing remaining LR sets: {remaining_lr_sets}"
    assert len(new_rows) == len(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Dropped lr {args.drop_lr} from {len(new_rows)} rows")
    print(f"Remaining learning rates: {list(remaining_lr_sets.pop())}")
    print(f"Wrote {args.output_file}")
