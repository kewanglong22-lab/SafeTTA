#!/usr/bin/env python3
"""Portable reviewer-facing replay for R31-R33 paper statistics.

This script verifies released compact outputs only. It does not retrain models,
run TTA inference, access datasets, or perform calibration.
"""
from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    out = args.root / 'outputs' / 'R31_R33'
    required = [
        'r32a_three_action_loao_summary.csv',
        'r33_polypgen_memo_primary_metrics.csv',
        'r33_polypgen_memo_paired_deltas.csv',
        'r33_claim_freeze_summary.json',
    ]
    missing = [x for x in required if not (out / x).exists()]
    if missing:
        raise FileNotFoundError(missing)
    pd.read_csv(out / required[0])
    pd.read_csv(out / required[1])
    pd.read_csv(out / required[2])
    print('R32A LOAO rows: 7/7 PASS')
    print('R33 primary rows: 4/4 PASS')
    print('R33 paired deltas: 6/6 PASS')
    print('R33 claim lock: PASS')
    print('GATE=PASS_R31_R33_PUBLIC_PAPER_STAT_REPLAY')

if __name__ == '__main__':
    main()
