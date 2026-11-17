"""Create a LaTeX longtable of the best learning rate for every (language, dataset, model) pair.

Rows are sorted by language, dataset and model. The table needs the longtable and booktabs packages.

Example:
    python scripts/create_best_lr_table.py \
        --input_file data/updated_files/best_lrs.jsonl \
        --output_file best_lrs_table.tex
"""
import argparse
from pathlib import Path

from create_latex_tables import format_model_name, latex_escape, lang_map
from drop_lr_from_results import read_jsonl


def format_lr(lr):
    """Turn a learning rate such as "4e-05" into $4 \\times 10^{-5}$."""
    mantissa, exponent = f"{float(lr):e}".split("e")
    # keep every significant digit, e.g. "2.500000" -> "2.5" and "4.000000" -> "4"
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"${mantissa} \\times 10^{{{int(exponent)}}}$"


def create_best_lr_table(input_file, output_file):
    rows = read_jsonl(input_file)
    assert rows, f"No rows found in {input_file}"
    keys = [(r["language"], r["dataset"], r["model"]) for r in rows]
    assert len(keys) == len(set(keys)), "Duplicate (language, dataset, model) rows"
    for r in rows:
        assert r["language"] in lang_map, f"Unknown language {r['language']}"

    rows.sort(key=lambda r: (r["language"], r["dataset"],
                             format_model_name(r["model"].split("/")[-1]).lower()))

    n_cols = 5
    header = ("\\textbf{Language} & \\textbf{Dataset} & \\textbf{Model} & "
              "\\textbf{Best LR} & \\textbf{Val. F1} \\\\")
    tex_lines = [
        "\\begin{longtable}{l l l c c}",
        "\\small \\\\",
        "\\caption{Best learning rate for each model and EuroEval dataset, selected by the mean of the ",
        "F1 scores on the validation split in the hyperparameter search.}",
        "\\label{tab:euroeval-best-lrs}\\\\",
        "\\toprule",
        header,
        "\\midrule",
        "\\endfirsthead",
        f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Table \\thetable{{}} continued from previous page}}}} \\\\",
        "\\toprule",
        header,
        "\\midrule",
        "\\endhead",
        "\\midrule",
        f"\\multicolumn{{{n_cols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]

    prev_lang, prev_ds = None, None
    for r in rows:
        lang, ds = r["language"], r["dataset"]
        # show language and dataset only on the first row of their group
        if prev_ds is not None and ds != prev_ds:
            tex_lines.append("\\midrule")
        lang_col = latex_escape(lang_map[lang]) if lang != prev_lang else ""
        ds_col = latex_escape(ds) if (lang, ds) != (prev_lang, prev_ds) else ""
        model_col = latex_escape(format_model_name(r["model"].split("/")[-1]))
        cols = [lang_col, ds_col, model_col, r["best_lr"], f"{r['mean_f1']:.1f}"]
        tex_lines.append(" & ".join(cols) + " \\\\")
        prev_lang, prev_ds = lang, ds

    tex_lines.append("\\end{longtable}")
    tex_lines.append("")

    Path(output_file).write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {output_file}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_file', default="best_lrs.jsonl", help="best_lrs.jsonl from make_final_training_list.py")
    ap.add_argument('--output_file', default="best_lrs_table.tex")
    args = ap.parse_args()
    create_best_lr_table(args.input_file, args.output_file)
