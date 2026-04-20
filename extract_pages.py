#!/usr/bin/env python3
"""Extract an inclusive page range from a PDF into a new PDF file.

Usage:
	python extract_pages.py input.pdf 3 7

This writes a new file beside the input PDF named:
	input_pages_3_7.pdf
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Extract a range of pages from a PDF into a new PDF file."
	)
	parser.add_argument("input_file", type=Path, help="Path to the source PDF file.")
	parser.add_argument(
		"start_page",
		type=int,
		help="First page to extract (1-based, inclusive).",
	)
	parser.add_argument(
		"end_page",
		type=int,
		help="Last page to extract (1-based, inclusive).",
	)
	return parser.parse_args()


def build_output_path(input_path: Path, start_page: int, end_page: int) -> Path:
	return input_path.with_name(
		f"{input_path.stem}_pages_{start_page}_{end_page}{input_path.suffix}"
	)


def extract_pages(input_path: Path, start_page: int, end_page: int) -> Path:
	try:
		pdf_module = importlib.import_module("pypdf")
	except ImportError:
		try:
			pdf_module = importlib.import_module("PyPDF2")
		except ImportError as exc:
			raise RuntimeError(
				"Missing dependency: install 'pypdf' or 'PyPDF2' before running this script."
			) from exc

	PdfReader = pdf_module.PdfReader
	PdfWriter = pdf_module.PdfWriter

	if not input_path.exists():
		raise FileNotFoundError(f"Input file does not exist: {input_path}")

	if input_path.suffix.lower() != ".pdf":
		raise ValueError(f"Input file is not a PDF: {input_path}")

	if start_page < 1 or end_page < 1:
		raise ValueError("Page numbers must be positive integers.")

	if start_page > end_page:
		raise ValueError("start_page must be less than or equal to end_page.")

	reader = PdfReader(str(input_path))
	total_pages = len(reader.pages)

	if end_page > total_pages:
		raise ValueError(
			f"Page range {start_page}-{end_page} exceeds PDF length of {total_pages} pages."
		)

	writer = PdfWriter()
	for page_index in range(start_page - 1, end_page):
		writer.add_page(reader.pages[page_index])

	output_path = build_output_path(input_path, start_page, end_page)
	with output_path.open("wb") as output_file:
		writer.write(output_file)

	return output_path


def main() -> int:
	args = parse_args()

	try:
		output_path = extract_pages(
			args.input_file.resolve(),
			args.start_page,
			args.end_page,
		)
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	print(f"Wrote extracted pages to: {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
