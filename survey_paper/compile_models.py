#!/usr/bin/env python3
r"""
compile_models.py
-----------------
Converts the survey models CSV into a LaTeX table, grouped by architecture
category (using the 'Architecture' column when present, otherwise 'Approach').

Displayed columns are inferred dynamically from the CSV rather than being
hard-coded. The script assumes the CSV contains:
  - a model-name column ('Model' or 'Name')
  - a bibtex key column ('Bibtex')
  - a grouping column ('Architecture' or 'Approach')

Model names are rendered as: ModelName~(\citeyear{bibtex_key})
which shows just the year as a hyperlink in the compiled PDF.

Usage:
    python3 compile_models.py [input.csv] [output.tex]

Defaults:
    input  → models_table.csv
    output → models_table.tex
"""

import re
import sys
from pathlib import Path
import pandas as pd

# ── CLI args ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV  = sys.argv[1] if len(sys.argv) > 1 else str(BASE_DIR / "models_table.csv")
OUTPUT_TEX = sys.argv[2] if len(sys.argv) > 2 else str(BASE_DIR / "models_table.tex")

DISPLAY_NAME_MAP = {
    "Name": "Model",
    "Model": "Model",
    "Downstream Tasks": "Tasks",
    "Modality": "Mod.",
    "Prompt Strategy": "Prompt Strat.",
    "Prompting Strategy": "Prompt Strat.",
    "RAG Strategy": "RAG Strat.",
    "Agentic Strategy": "Agentic",
}

COLUMN_SPEC_MAP = {
    "Name": "l",
    "Model": "l",
    "Venue": "l",
    "Downstream Tasks": "p{2.8cm}",
    "Modality": "c",
    "Vision Encoder": "p{2.8cm}",
    "LLM Backbone": "p{3.0cm}",
    "Prompt Strategy": "c",
    "Prompting Strategy": "c",
    "RAG Strategy": "c",
    "Agentic Strategy": "c",
}

HIDDEN_COLUMNS = {"ID", "Bibtex", "OCR", "Architecture", "Approach"}

PREFERRED_ORDER = [
    "Model",
    "Name",
    "Venue",
    "Downstream Tasks",
    "Modality",
    "Vision Encoder",
    "LLM Backbone",
    "Prompt Strategy",
    "Prompting Strategy",
    "RAG Strategy",
    "Agentic Strategy",
]

GROUP_ORDER = [
    "Hierarchical Document Transformer",
    "Backbone-Centric MLLM Adaptation",
    "Retriever-Generator",
    "Agentic Pipeline",
]


def first_present(columns, candidates):
    """Return the first candidate column present in the DataFrame."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def make_column_config(df_columns):
    """Infer the displayed columns, labels, and column specs from the CSV."""
    visible = [col for col in df_columns if col not in HIDDEN_COLUMNS]

    ordered = [col for col in PREFERRED_ORDER if col in visible]
    ordered.extend(col for col in visible if col not in ordered)

    return [
        (
            col,
            DISPLAY_NAME_MAP.get(col, col),
            COLUMN_SPEC_MAP.get(col, r"p{2.6cm}"),
        )
        for col in ordered
    ]

def escape_latex(val):
    """Escape special LaTeX characters in a cell value."""
    if pd.isna(val):
        return "--"
    val = str(val).strip()
    for char, repl in [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]:
        val = val.replace(char, repl)
    return val if val else "--"

def extract_year(bibtex_key):
    """Pull a 4-digit year out of a bibtex key, e.g. 'wang2025foo' → '2025'."""
    m = re.search(r"(20\d{2}|19\d{2})", bibtex_key)
    return m.group(1) if m else None

def make_model_cell(row, model_col):
    r"""
    Format:  \textbf{Name}~(\citeyear{key})
    \citeyear requires natbib or biblatex; it prints just the year as a link.
    Falls back to \cite{key} if no year can be extracted.
    """
    name   = escape_latex(row[model_col])
    bibtex = str(row["Bibtex"]).strip() if pd.notna(row["Bibtex"]) else ""
    if not bibtex:
        return rf"\textbf{{{name}}}"
    year = extract_year(bibtex)
    if year:
        return rf"\textbf{{{name}}}~(\citeyear{{{bibtex}}})"
    return rf"\textbf{{{name}}}~\cite{{{bibtex}}}"

def build_table_rows(df, columns, model_col):
    lines = []
    for _, row in df.iterrows():
        cells = []
        for col, _, _ in columns:
            if col == model_col:
                cells.append(make_model_cell(row, model_col))
            else:
                cells.append(escape_latex(row.get(col, "")))
        lines.append("  " + " & ".join(cells) + r" \\")
    return "\n".join(lines)

def main():
    df = pd.read_csv(INPUT_CSV)

    model_col = first_present(df.columns, ["Model", "Name"])
    group_col = first_present(df.columns, ["Architecture", "Approach"])

    required = {"Bibtex"}
    missing = required - set(df.columns)
    if not model_col:
        missing.add("Model/Name")
    if not group_col:
        missing.add("Architecture/Approach")
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    columns = make_column_config(df.columns)
    col_spec = "|".join(spec for _, _, spec in columns)
    col_headers = " & ".join(
        rf"\textbf{{{hdr}}}" for _, hdr, _ in columns
    )
    n_cols = len(columns)
    section_span = rf"\multicolumn{{{n_cols}}}{{|l|}}"

    df[group_col] = df[group_col].fillna("Uncategorized").astype(str).str.strip()

    seen_groups = list(dict.fromkeys(df[group_col].tolist()))
    ordered_groups = [group for group in GROUP_ORDER if group in seen_groups]
    ordered_groups.extend(group for group in seen_groups if group not in ordered_groups)

    section_lines = []
    for idx, group in enumerate(ordered_groups):
        group_df = df[df[group_col] == group].copy()
        if group_df.empty:
            continue
        if idx > 0:
            section_lines.append(r"\midrule[0.4pt]")
            section_lines.append("")
        section_lines.append(rf"% ── {group} ─────────────────────────────────────────────────────────────")
        section_lines.append(r"\rowcolor{sectionbg}")
        section_lines.append(rf"{section_span}{{\textbf{{{escape_latex(group)}}}}} \\")
        section_lines.append(r"\midrule")
        section_lines.append(build_table_rows(group_df, columns, model_col))
        section_lines.append("")

    section_block = "\n".join(section_lines).rstrip()

    latex = rf"""% ─────────────────────────────────────────────────────────────────────────────
% Auto-generated by compile_models.py — do NOT edit manually.
% Re-generate:  python3 compile_models.py models_table.csv models_table.tex
%
% Required packages (add to your preamble):
%   \usepackage{{booktabs}}
%   \usepackage{{longtable}}
%   \usepackage{{array}}
%   \usepackage{{xcolor}}
%   \usepackage{{colortbl}}
%   \usepackage{{caption}}
%   \usepackage{{adjustbox}}   % for \begin{{adjustbox}}
%   \usepackage{{natbib}}      % for \citeyear  (or use biblatex)
%   \usepackage{{lscape}}      % optional: landscape
% ─────────────────────────────────────────────────────────────────────────────

\definecolor{{sectionbg}}{{gray}}{{0.88}}

% \begin{{landscape}}   % ← uncomment for landscape orientation
\begin{{table}}[htbp]
\centering
\caption{{Survey of VRDU models, split by architecture.
  \textbf{{Mod.}} = modality of document used by model.
  \textbf{{T}} = text, \textbf{{V}} = visual/image, \textbf{{L}} = layout.
  `--' indicates not reported.}}
\label{{tab:model_survey}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{{col_spec}}}
\toprule
{col_headers} \\
\midrule

{section_block}

\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table}}
% \end{{landscape}}   % ← uncomment if you used landscape above
"""

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)

    print(
        f"✓  Wrote {OUTPUT_TEX}  "
        f"({len(df)} rows across {len(ordered_groups)} {group_col.lower()} groups)"
    )

if __name__ == "__main__":
    main()
