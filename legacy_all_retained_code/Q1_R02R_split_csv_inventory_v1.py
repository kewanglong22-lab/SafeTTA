#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inspect CSV files under F:\\MEDSEG_SAFETTA\\data\\splits.

Read-only:
- no image opening
- no data modification
- no training
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R02R-split-csv-inventory-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
SPLIT_ROOT = ROOT / "data" / "splits"
OUTPUT_DIR = ROOT / "outputs" / "Q1_R02R_split_csv_inventory_v1"


def inspect_csv(path: Path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
        errors="ignore",
    ) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        row_count = 0
        first_rows = []
        unique_values = {
            field: set()
            for field in fields
        }

        for row in reader:
            row_count += 1

            if len(first_rows) < 10:
                first_rows.append(row)

            for field in fields:
                value = str(row.get(field, "") or "").strip()
                if value and len(unique_values[field]) < 1000:
                    unique_values[field].add(value)

    split_like = []
    path_like = []
    seed_like = []
    group_like = []

    for field in fields:
        low = field.lower()

        if any(
            token in low
            for token in (
                "split", "fold", "subset",
                "phase", "train", "val", "test"
            )
        ):
            split_like.append(field)

        if any(
            token in low
            for token in (
                "path", "file", "filename",
                "image", "mask", "case", "sample"
            )
        ):
            path_like.append(field)

        if "seed" in low:
            seed_like.append(field)

        if any(
            token in low
            for token in ("group", "patient", "subject")
        ):
            group_like.append(field)

    unique_summary = {
        field: {
            "unique_count_capped": len(values),
            "values_preview": sorted(values)[:30],
        }
        for field, values in unique_values.items()
    }

    return {
        "filename": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "columns": fields,
        "row_count": row_count,
        "split_like_columns": split_like,
        "path_like_columns": path_like,
        "seed_like_columns": seed_like,
        "group_like_columns": group_like,
        "first_rows": first_rows,
        "unique_summary": unique_summary,
    }


def run(args):
    split_root = args.split_root.resolve()
    output_dir = args.output_dir.resolve()

    if not split_root.exists():
        raise FileNotFoundError(split_root)

    csv_files = sorted(
        split_root.glob("*.csv"),
        key=lambda p: p.name.lower(),
    )

    if not csv_files:
        raise RuntimeError(
            f"No CSV files found in {split_root}"
        )

    if output_dir.exists():
        raise FileExistsError(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)

    reports = []

    for path in tqdm(
        csv_files,
        desc="Inspecting split CSV files",
        unit="file",
        dynamic_ncols=True,
    ):
        reports.append(inspect_csv(path))

    (output_dir / "split_csv_details.json").write_text(
        json.dumps(
            reports,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "===== Q1-R02R SPLIT CSV INVENTORY =====",
        f"Script version: {VERSION}",
        "",
        f"Split root: {split_root}",
        f"CSV files found: {len(reports)}",
        "",
    ]

    for report in reports:
        lines += [
            f"FILE: {report['filename']}",
            f"  size={report['bytes']} bytes",
            f"  rows={report['row_count']}",
            f"  columns={report['columns']}",
            f"  split-like={report['split_like_columns']}",
            f"  path-like={report['path_like_columns']}",
            f"  seed-like={report['seed_like_columns']}",
            f"  group-like={report['group_like_columns']}",
            "  first rows:",
        ]

        for row in report["first_rows"][:5]:
            lines.append(
                "    " + json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

        lines.append("  selected unique values:")
        interesting_fields = list(dict.fromkeys(
            report["split_like_columns"]
            + report["seed_like_columns"]
            + report["group_like_columns"]
        ))

        if interesting_fields:
            for field in interesting_fields:
                s = report["unique_summary"][field]
                lines.append(
                    f"    {field}: "
                    f"unique={s['unique_count_capped']} "
                    f"preview={s['values_preview']}"
                )
        else:
            lines.append("    NONE")

        lines.append("")

    lines += [
        "Files modified inside data/=NO",
        "Image content opened=NO",
        "Training=NO",
        "",
        "[OK] SPLIT_CSV_INVENTORY_COMPLETE",
    ]

    log = "\n".join(lines) + "\n"

    (output_dir / "run_log.txt").write_text(
        log,
        encoding="utf-8",
    )

    print(log)
    print(f"Output directory: {output_dir}")


def self_test():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.csv"
        p.write_text(
            "case_id,split,seed\n"
            "a,train,1\n"
            "b,val,1\n",
            encoding="utf-8",
        )

        result = inspect_csv(p)

        assert result["row_count"] == 2
        assert "split" in result["split_like_columns"]
        assert "seed" in result["seed_like_columns"]
        assert "case_id" in result["path_like_columns"]

    print("CSV_PARSE_TEST_PASS")
    print("COLUMN_CLASSIFICATION_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect CSV files in F:\\MEDSEG_SAFETTA\\data\\splits."
        )
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=SPLIT_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
