#!/usr/bin/env python3
"""
pdf_to_markdown.py

Converts all PDF files in a directory to Markdown format.
Output files are saved to a separate folder, named after the document
title extracted from the first heading (# or ##) in the converted text.

Uses pymupdf4llm for clean text extraction with simple heading formatting.
Tables and figures are not converted — only text and headings are preserved.

Usage:
    python pdf_to_markdown.py <input_dir> [output_dir]

    input_dir   — folder containing PDF files
    output_dir  — (optional) folder for markdown output; defaults to
                  <input_dir>/markdown_output

Dependencies:
    pip install pymupdf4llm
"""

import sys
import re
import argparse
from pathlib import Path


def extract_title(markdown_text: str, fallback: str) -> str:
    """
    Extract the document title from the first # or ## heading in the
    markdown text. Falls back to the PDF filename stem if no heading
    is found.
    """
    for line in markdown_text.splitlines():
        stripped = line.strip()
        match = re.match(r"^#{1,2}\s+(.+)", stripped)
        if match:
            title = match.group(1).strip()
            return sanitise_filename(title)

    return sanitise_filename(fallback)


def sanitise_filename(name: str) -> str:
    """
    Remove or replace characters that are unsafe in filenames.
    Collapses whitespace and trims to a reasonable length.
    """
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", name)
    name = re.sub(r"[-\s]+", " ", name).strip(" -")
    return name[:200] if name else "untitled"


def convert_pdfs(input_dir: Path, output_dir: Path) -> None:
    """
    Convert all PDFs in input_dir to Markdown files in output_dir.
    """
    import pymupdf4llm

    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pdf_files)} PDF(s). Converting...\n")

    success_count = 0
    fail_count = 0

    for pdf_path in pdf_files:
        print(f"  Converting: {pdf_path.name}")
        try:
            markdown_text = pymupdf4llm.to_markdown(
                str(pdf_path),
                show_progress=False,
            )

            title = extract_title(markdown_text, fallback=pdf_path.stem)

            output_path = output_dir / f"{title}.md"
            
            output_path.write_text(markdown_text, encoding="utf-8")
            print(f"    → Saved as: {output_path.name}")
            success_count += 1

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            fail_count += 1

    print(f"\nDone. {success_count} converted, {fail_count} failed.")
    print(f"Output folder: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert all PDFs in a directory to Markdown files."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Directory for output Markdown files (default: <input_dir>/markdown_output).",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    if not input_dir.exists():
        print(f"Error: input directory '{input_dir}' does not exist.")
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        sys.exit(1)

    output_dir: Path = (
        args.output_dir.resolve()
        if args.output_dir
        else input_dir / "markdown_output"
    )

    convert_pdfs(input_dir, output_dir)


if __name__ == "__main__":
    main()