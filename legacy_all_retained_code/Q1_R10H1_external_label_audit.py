
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--external", required=True)
args = parser.parse_args()

df = pd.read_csv(args.external, low_memory=False)

print("===== EXTERNAL LABEL AUDIT =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nPotential label columns:")
for c in df.columns:
    cl = c.lower()
    if any(k in cl for k in [
        "label", "target", "class", "gt",
        "outcome", "harm", "tumor",
        "neo", "cancer", "benign"
    ]):
        print(c)

print("\nAll columns:")
for c in df.columns:
    print(c)
