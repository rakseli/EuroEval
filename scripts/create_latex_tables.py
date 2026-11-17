import json
import argparse
from pathlib import Path
from collections import defaultdict
from statistics import mean
import math


score_map = {"test_mcc":"MCC",
            "test_macro_f1": "F1 Ma",
            "test_micro_f1_no_misc": "F1 Mi no misc",
            "test_f1":"F1",
            "test_micro_f1":"F1 Mi",
            "test_em":"EM",
            }

# metric order shown in the table for each task
task_metrics = {"linguistic-acceptability": ["test_mcc", "test_macro_f1"],
                "sentiment-classification": ["test_mcc", "test_macro_f1"],
                "named-entity-recognition": ["test_micro_f1_no_misc", "test_micro_f1"],
                "reading-comprehension": ["test_f1", "test_em"],
                }

lang_map = {"fi":"Finnish",
            "sv":"Swedish",
            "en":"English"}

# display names for models whose name is not formatted by the rules in format_model_name
model_name_map = {"xlm-roberta-large": "XLM-RoBERTa-large",
                  "finnish-modernbert-large-seq-len-1024-117300-annealed": "Finnish-ModernBERT-large-nce",
                  "deberta-v3-base": "DeBERTaV3-base",
                  "deberta-v3-large": "DeBERTaV3-large",
                  }


def format_model_name(model):
    """Turn a short HF model name into the name shown in the table."""
    if model in model_name_map:
        return model_name_map[model]
    if model.startswith("finnish-modernbert-"):
        return "Finnish-ModernBERT-" + model[len("finnish-modernbert-"):]
    return model


def format_result(res, metric_keys):
    """Return (formatted_string, metrics_dict) where metrics_dict maps metric -> (val, se_or_None)."""
    metrics = {}
    parts = []
    for k in metric_keys:
        assert k in res, f"Metric {k} missing from results {list(res)}"
        val = res[k]
        se = res.get(k + "_se")
        metrics[k] = (val, se)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            parts.append("-")
        elif se is not None:
            parts.append(f"{val:.1f}±{se:.1f}")
        else:
            parts.append(f"{val:.1f}")

    return " / ".join(parts) if parts else "-", metrics


def latex_escape(s):
    if s is None:
        return "-"
    s = str(s)
    s = s.replace("\\", "\\textbackslash{}")
    for ch, esc in [("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#"),
                    ("{", "\\{"), ("}", "\\}"), ("$", "\\$"), ("~", "\\textasciitilde{}"),
                    ("^", "\\textasciicircum{}")]:
        s = s.replace(ch, esc)
    return s


def create_latex_tables(input_file, output_file, monolingual_models):
    # Load JSONL
    rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            l = json.loads(line)
            if "failed" in l['model']:
                continue
            else:
                rows.append(l)

    # Group by primary language (first in dataset_languages)
    by_lang = defaultdict(list)
    for r in rows:
        langs = r.get("dataset_languages", [])
        if langs:
            by_lang[langs[0]].append(r)

    def get_total(row):
        for val in row.get("results", {}).values():
            if isinstance(val, dict) and "total" in val:
                return val["total"]
        return {}

    tex_lines = []
    for lang, items in sorted(by_lang.items()):
        recs = []
        for r in items:
            total = get_total(r) or {}
            metric_keys = task_metrics[r["task"]]
            formatted, metrics = format_result(total, metric_keys)
            recs.append({
                "dataset": r.get("dataset"),
                "task": r.get("task"),
                "model": r.get("model").split("/")[-1],
                "num_params": r.get("num_model_parameters"),
                "formatted": formatted,
                "metrics": metrics,
                "metrics_keys": [score_map[k] for k in metric_keys]
            })


        if not recs:
            continue
        d_m_keys = {}
        for r in recs:
            d_m_keys[r['dataset']]=r['metrics_keys']

        datasets = sorted({r["dataset"] for r in recs})
        n_cols = 2 + len(datasets)

        best_mono_per_dataset = {}
        best_multi_per_dataset = {}
        for ds in datasets:
            ds_entries = [r for r in recs if r["dataset"] == ds]
            best_mono_model = None
            best_mono_mean = float("-inf")
            best_multi_model = None
            best_multi_mean = float("-inf")
            for entry in ds_entries:
                mean_score = mean(v[0] for v in entry['metrics'].values())
                if entry['model'] in monolingual_models:
                    if mean_score>best_mono_mean:
                        best_mono_model = entry["model"]
                        best_mono_mean = mean_score
                else:
                    if mean_score>best_multi_mean:
                        best_multi_model = entry["model"]
                        best_multi_mean = mean_score


            best_mono_per_dataset[ds] = best_mono_model
            best_multi_per_dataset[ds] = best_multi_model


        # split subsets
        mono = [r for r in recs if r["model"] in monolingual_models]
        multi = [r for r in recs if r["model"] not in monolingual_models]
        # === Start table for language ===
        tex_lines.append(f"% --- Language: {latex_escape(lang)} ---")
        tex_lines.append("\\begin{table*}[!htbp]")
        tex_lines.append("\\centering")
        tex_lines.append("\\scriptsize")
        # column spec: two left columns then one column per dataset
        col_spec = "l l " + " ".join(["c"] * len(datasets))
        tex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        # Single header (no repetition)
        header = ["\\textbf{Model}", "\\textbf{N params (M)}"] + [fr'\textbf{{{latex_escape(ds)}}}' for ds in datasets]
        tex_lines.append(" & ".join(header) + " \\\\")
        score_header = [fr'\textit{{{" / ".join(d_m_keys[ds])}}}' for ds in datasets]
        tex_lines.append(f"\\cmidrule(lr){{3-{n_cols}}} & & " + " & ".join(score_header) + " \\\\")

        tex_lines.append("\\midrule")

        def section_rows(subset, title,best_models):
            if not subset:
                return []
            lines = []
            # section title spanning all columns
            lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{latex_escape(title)}}}}} \\\\")
            models = sorted({r["model"] for r in subset}, key=lambda m: format_model_name(m).lower())
            for m in models:
                row_entries = [r for r in subset if r["model"] == m]
                num_params = row_entries[0].get("num_params")
                num_params_m = f"{(num_params / 1e6):.1f}" if isinstance(num_params, (int, float)) else "-"
                model_label = latex_escape(format_model_name(m))
                cols = [model_label, num_params_m]
                for ds in datasets:
                    item = next((r for r in row_entries if r["dataset"] == ds), None)
                    if not item:
                        cols.append("-")
                    else:
                        val = latex_escape(item["formatted"])
                        if best_models.get(ds) == m:
                            val = f"\\textbf{{{val}}}"
                        cols.append(val)
                lines.append(" & ".join(cols) + " \\\\")
            return lines

        # add monolingual section (if any)
        if mono:
            tex_lines.extend(section_rows(mono, "Monolingual",best_mono_per_dataset))
        # add a bit of vertical space between sections if both exist
        if mono and multi:
            tex_lines.append("\\addlinespace")
            tex_lines.append("\\midrule")

        if multi:
            tex_lines.extend(section_rows(multi, "Multilingual",best_multi_per_dataset))

        tex_lines.append("\\bottomrule")
        tex_lines.append("\\end{tabular}")
        tex_lines.append(f'\\caption{{Results for tasks included in EuroEval in the {latex_escape(lang_map[lang])} language. In the metrics header, "MCC" is an abbreviation for the Matthews Correlation Coefficient, "Ma" denotes macro, "Mi" denotes micro, and "EM" denotes exact match. The best average scores for monolingual and multilingual models are highlighted in bold. The value after the "±" symbol is a 95\\% confidence interval.}}')
        tex_lines.append(f"\\label{{tab:euroeval-{lang}}}")
        tex_lines.append("\\end{table*}")
        tex_lines.append("")

    Path(output_file).write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"LaTeX tables written to {output_file}")


MONOLINGUAL_MODELS = {
    "bert-base-finnish-cased-v1",
    "bert-large-finnish-cased-v1",
    "roberta-large-1160k",
    "deberta-v3-base",
    "deberta-v3-large",
    # add more monolingual models here
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_file',default="euroeval_benchmark_results_spesific_lrs.jsonl",help="final training results file")
    ap.add_argument('--output_file',default="tables.tex")
    args = ap.parse_args()
    create_latex_tables(args.input_file, args.output_file, monolingual_models=MONOLINGUAL_MODELS)
