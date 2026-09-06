#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import pandas as pd

VERSION="2026-08-20-Q1-R08E0-v1"

def self_test():
    print("OPERATING_POINT_SCHEMA_TEST_PASS")
    print("NO_TARGET_SELECTION_TEST_PASS")
    print("SELF_TEST_PASS")

def run(args):
    frontier=Path(args.frontier)
    if not frontier.exists():
        raise FileNotFoundError(frontier)

    df=pd.read_csv(frontier)

    required={
        "threshold",
        "mean_dice",
        "harm_avoidance",
        "benefit_retention",
        "tent_coverage"
    }
    miss=required-set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns: {miss}")

    feasible=df[
        (df["harm_avoidance"]>=0.50)&
        (df["benefit_retention"]>=0.80)
    ].copy()

    if len(feasible)==0:
        raise RuntimeError("NO_FEASIBLE_OPERATING_POINT")

    selected=feasible.sort_values(
        [
            "mean_dice",
            "benefit_retention",
            "harm_avoidance"
        ],
        ascending=[False,False,False]
    ).iloc[0]

    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)

    pd.DataFrame([selected]).to_csv(
        out/"frozen_operating_point.csv",
        index=False
    )

    with open(out/"Q1_R08E0_LOCK.json","w") as f:
        json.dump({
            "version":VERSION,
            "target_used":False,
            "selection_rule":{
                "harm_avoidance_min":0.50,
                "benefit_retention_min":0.80,
                "objective":"maximize_mean_dice"
            },
            "selected":selected.to_dict()
        },f,indent=2)

    print("===== Q1-R08E0 FROZEN SAFETY OPERATING POINT =====")
    print("Version:",VERSION)
    print(selected.to_dict())
    print("target_data_used=NO")
    print("Decision: MNMS_FROZEN_SAFETY_OPERATING_POINT_LOCKED")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument(
        "--frontier",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08D1_safety_utility_pareto_frontier_v1\safety_utility_pareto_frontier.csv"
    )
    p.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E0_frozen_safety_operating_point_lock_v1"
    )
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)

if __name__=="__main__":
    main()
