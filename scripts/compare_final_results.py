"""Show the final results that changed between the original and the updated results file.

Only models found in --updated_file are compared, and only (model, dataset) pairs found in both
files. A pair is shown if its learning rate or any of its total scores changed. Rows are sorted
by language, dataset and model. A per model summary of the average change follows the table.

Example:
    python scripts/compare_final_results.py \
        --original_file data/euroeval/euroeval_benchmark_results_spesific_lrs.jsonl \
        --updated_file data/updated_files/euroeval_benchmark_results_spesific_lrs.jsonl
"""
import argparse
from statistics import mean

from create_latex_tables import format_model_name, format_result, score_map, task_metrics
from drop_lr_from_results import read_jsonl


def by_pair(rows, path):
    pairs = {}
    for r in rows:
        key = (r["model"], r["dataset"])
        assert key not in pairs, f"Duplicate result {key} in {path}"
        assert len(r["results"]) == 1, f"{key} in {path} has {len(r['results'])} learning rates"
        pairs[key] = r
    return pairs


def lr_and_total(r):
    lr, res = next(iter(r["results"].items()))
    return lr, res["total"]


def print_table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda cols: " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))
    print(line(header))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--original_file', default="../data/euroeval/euroeval_benchmark_results_spesific_lrs.jsonl")
    ap.add_argument('--updated_file', default="../data/updated_files/euroeval_benchmark_results_spesific_lrs.jsonl")
    args = ap.parse_args()

    original = by_pair(read_jsonl(args.original_file), args.original_file)
    updated = by_pair(read_jsonl(args.updated_file), args.updated_file)
    models = {m for m, _ in updated}

    changed, n_same, n_only_original, n_only_updated = [], 0, 0, 0
    # model -> [(Δ mean, changed)] for every pair compared, unchanged pairs have Δ 0
    model_deltas = {}
    for key in sorted(set(original) | set(updated)):
        if key[0] not in models:
            continue
        if key not in updated:
            n_only_original += 1
            continue
        if key not in original:
            n_only_updated += 1
            continue
        old, new = original[key], updated[key]
        assert old["task"] == new["task"] and old["dataset_languages"] == new["dataset_languages"], \
            f"Task or language differs for {key}"
        old_lr, old_total = lr_and_total(old)
        new_lr, new_total = lr_and_total(new)
        metric_keys = task_metrics[new["task"]]
        if old_lr == new_lr and all(old_total[k] == new_total[k] for k in metric_keys):
            n_same += 1
            model_deltas.setdefault(key[0], []).append((0.0, False))
            continue
        old_str, old_metrics = format_result(old_total, metric_keys)
        new_str, new_metrics = format_result(new_total, metric_keys)
        delta = mean(new_metrics[k][0] for k in metric_keys) - mean(old_metrics[k][0] for k in metric_keys)
        model_deltas.setdefault(key[0], []).append((delta, True))
        changed.append([new["dataset_languages"][0], new["dataset"],
                        format_model_name(new["model"].split("/")[-1]),
                        " / ".join(score_map[k] for k in metric_keys),
                        old_lr, old_str, new_lr, new_str, f"{delta:+.1f}"])

    changed.sort(key=lambda r: (r[0], r[1], r[2].lower()))
    if changed:
        print_table(["Lang", "Dataset", "Model", "Metrics", "Old LR", "Old", "New LR", "New", "Δ mean"], changed)
    else:
        print("No changed results")
    print()

    summary = []
    for model, entries in model_deltas.items():
        deltas = [d for d, _ in entries]
        changed_deltas = [d for d, c in entries if c]
        summary.append([format_model_name(model.split("/")[-1]), len(entries), len(changed_deltas),
                        f"{mean(deltas):+.2f}",
                        f"{mean(changed_deltas):+.2f}" if changed_deltas else "-",
                        f"{min(deltas):+.1f}", f"{max(deltas):+.1f}"])
    summary.sort(key=lambda r: r[0].lower())
    if summary:
        print("Per model average change of the mean score (Δ mean), over all compared pairs and over changed pairs only")
        print_table(["Model", "Pairs", "Changed", "Mean Δ", "Mean Δ changed", "Min Δ", "Max Δ"], summary)
        print()
    print(f"Models compared: {len(models)}")
    print(f"Changed: {len(changed)}, unchanged: {n_same}")
    print(f"Only in original (not yet rerun): {n_only_original}, only in updated: {n_only_updated}")
