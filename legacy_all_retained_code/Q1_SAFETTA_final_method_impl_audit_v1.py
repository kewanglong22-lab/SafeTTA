from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"
OUT = OUTPUTS / "Q1_SAFETTA_final_method_impl_audit_v1"
REPORT_TXT = OUT / "final_method_impl_audit.txt"
REPORT_JSON = OUT / "final_method_impl_audit.json"

VERSION = "2026-09-05-Q1-SAFETTA-FINAL-IMPL-AUDIT-v1"

# Keep this audit read-only: it never imports or executes project scripts.
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_SNIPPET_LINES = 180
TOP_K = 12

CATEGORY_RULES = {
    "tent_action": {
        "must_any": ["tent", "entropy"],
        "boost": [
            "optimizer.step", "requires_grad", "batchnorm", "batch_norm", "norm", "softmax",
            "sigmoid", "adam", "sgd", "lr=", "learning_rate", "episodic", "reset",
            "deepLab", "pranet", "segformer", "unet", "prostate", "promise"
        ],
    },
    "mri_dice_empty": {
        "must_any": ["dice", "delta_dice", "promise12", "prostate158"],
        "boost": [
            "empty", "denom", "denominator", "sum() == 0", "sum()==0", "union", "intersection",
            "return 1.0", "return 0.0", "1.0 if", "np.isclose", "eps", "epsilon", "harm"
        ],
    },
    "mask_22_to_16": {
        "must_any": ["occupancy", "22x22", "unpack_mask_occupancy16", "patch_grid"],
        "boost": [
            "352", "22", "16", "reshape", "interpolate", "resize", "area", "bilinear", "nearest",
            "source_masks_packed", "unpackbits", "block"
        ],
    },
    "dinov2_preprocess": {
        "must_any": ["dinov2", "autoimageprocessor", "facebook/dinov2-base"],
        "boost": [
            "224", "bicubic", "do_resize", "do_center_crop", "image_mean", "image_std", "normalize",
            "processor(", "return_tensors", "pixel_values", "rgb"
        ],
    },
}


@dataclass
class Candidate:
    category: str
    path: str
    sha256: str
    score: int
    matched: List[str]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read_text(p: Path) -> str:
    try:
        if p.stat().st_size > MAX_FILE_BYTES:
            return ""
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def norm(s: str) -> str:
    return s.lower().replace(" ", "")


def score_text(text: str, rule: Dict[str, List[str]]) -> Tuple[int, List[str]]:
    low = text.lower()
    compact = norm(text)
    matched = []
    base = 0
    for kw in rule["must_any"]:
        if kw.lower() in low or norm(kw) in compact:
            base += 7
            matched.append(kw)
    if base == 0:
        return 0, []
    score = base
    for kw in rule["boost"]:
        if kw.lower() in low or norm(kw) in compact:
            score += 2
            matched.append(kw)
    return score, sorted(set(matched))


def iter_candidate_files() -> Iterable[Path]:
    suffixes = {".py", ".json", ".txt", ".md", ".log", ".yaml", ".yml"}
    roots = [CODE, OUTPUTS]
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in suffixes:
                continue
            # Avoid scanning our own output directory recursively.
            try:
                if OUT in p.parents:
                    continue
            except Exception:
                pass
            key = str(p).lower()
            if key in seen:
                continue
            seen.add(key)
            yield p


def line_numbered(lines: List[str], start: int, end: int) -> str:
    start = max(1, start)
    end = min(len(lines), end)
    if end - start + 1 > MAX_SNIPPET_LINES:
        end = start + MAX_SNIPPET_LINES - 1
    return "\n".join(f"{i:5d}: {lines[i-1]}" for i in range(start, end + 1))


def extract_ast_blocks(text: str, category: str) -> List[Dict[str, object]]:
    blocks: List[Dict[str, object]] = []
    try:
        tree = ast.parse(text)
    except Exception:
        return blocks
    lines = text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if not start or not end:
            continue
        seg = "\n".join(lines[start-1:end])
        sc, matched = score_text(seg, CATEGORY_RULES[category])
        name = getattr(node, "name", "")
        # Strong boosts for exact functions / likely evaluator blocks.
        lname = name.lower()
        if category == "mask_22_to_16" and "occupancy" in lname:
            sc += 20
        if category == "tent_action" and ("tent" in lname or "entropy" in lname):
            sc += 12
        if category == "mri_dice_empty" and "dice" in lname:
            sc += 12
        if category == "dinov2_preprocess" and any(x in lname for x in ["dino", "image", "feature", "token"]):
            sc += 6
        if sc <= 0:
            continue
        blocks.append({
            "kind": type(node).__name__,
            "name": name,
            "start": start,
            "end": end,
            "score": sc,
            "matched": matched,
            "snippet": line_numbered(lines, start, end),
        })
    blocks.sort(key=lambda x: (-int(x["score"]), int(x["start"])))
    return blocks[:8]


def extract_regex_contexts(text: str, patterns: List[str], context: int = 10) -> List[Dict[str, object]]:
    lines = text.splitlines()
    out = []
    used = set()
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        hits = [p for p in patterns if p.lower() in low]
        if not hits:
            continue
        s = max(1, i-context)
        e = min(len(lines), i+context)
        key = (s, e)
        if key in used:
            continue
        used.add(key)
        out.append({
            "line": i,
            "hits": hits,
            "snippet": line_numbered(lines, s, e),
        })
        if len(out) >= 10:
            break
    return out


def inspect_serialized_estimators() -> Dict[str, object]:
    result: Dict[str, object] = {"status": "not_found", "artifacts": []}
    candidates = []
    if OUTPUTS.exists():
        for p in OUTPUTS.rglob("*.joblib"):
            n = p.name.lower()
            if any(k in n for k in ["pca64", "safety_head", "safety_estimator", "conddino"]):
                candidates.append(p)
    if not candidates:
        return result
    result["status"] = "found"
    try:
        import joblib
    except Exception as e:
        result["status"] = f"joblib_import_failed: {e}"
        return result

    for p in sorted(candidates)[:20]:
        row: Dict[str, object] = {"path": str(p)}
        try:
            row["sha256"] = sha256_file(p)
            obj = joblib.load(p)
            row["python_type"] = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            if hasattr(obj, "get_params"):
                params = obj.get_params(deep=True)
                # JSON-safe representation.
                row["get_params"] = {str(k): repr(v) for k, v in sorted(params.items())}
            for attr in ["n_features_in_", "n_components_", "classes_", "explained_variance_ratio_"]:
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    try:
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        json.dumps(val)
                        row[attr] = val
                    except Exception:
                        row[attr] = repr(val)
            if hasattr(obj, "steps"):
                row["steps"] = [(n, f"{s.__class__.__module__}.{s.__class__.__name__}") for n, s in obj.steps]
        except Exception as e:
            row["error"] = repr(e)
        result["artifacts"].append(row)
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===== SAFETTA FINAL METHOD IMPLEMENTATION AUDIT =====")
    print("Version:", VERSION)
    print("Root:", ROOT)
    print("Read-only scan: YES")
    print("Project scripts executed/imported: NO")
    print("Target-side fitting/tuning: NO")
    print("Output:", OUT)
    print()

    if not CODE.exists():
        raise FileNotFoundError(f"Code directory not found: {CODE}")

    files = list(iter_candidate_files())
    print("Candidate text/code files:", len(files))

    all_candidates: Dict[str, List[Candidate]] = {k: [] for k in CATEGORY_RULES}
    text_cache: Dict[str, str] = {}

    for p in tqdm(files, desc="Scan implementation files", unit="file", dynamic_ncols=True):
        text = safe_read_text(p)
        if not text:
            continue
        text_cache[str(p)] = text
        for category, rule in CATEGORY_RULES.items():
            score, matched = score_text(text, rule)
            if score <= 0:
                continue
            # filename boosts
            ln = p.name.lower()
            if category == "tent_action" and "tent" in ln:
                score += 10
            if category == "mri_dice_empty" and any(k in ln for k in ["promise", "prostate", "dice", "gt_reveal", "outcome"]):
                score += 8
            if category == "mask_22_to_16" and any(k in ln for k in ["dinov2", "representation", "mask"]):
                score += 8
            if category == "dinov2_preprocess" and any(k in ln for k in ["dinov2", "representation", "feature"]):
                score += 8
            try:
                sha = sha256_file(p)
            except Exception:
                sha = "UNAVAILABLE"
            all_candidates[category].append(Candidate(category, str(p), sha, score, matched))

    for category in all_candidates:
        all_candidates[category].sort(key=lambda c: (-c.score, c.path.lower()))
        all_candidates[category] = all_candidates[category][:TOP_K]

    details: Dict[str, List[Dict[str, object]]] = {}
    regex_patterns = {
        "tent_action": [
            "optimizer =", "optimizer=", "optim.", "optimizer.step", "requires_grad", "entropy",
            "lr=", "learning_rate", "batchnorm", "batch_norm", "episodic", "reset", "tent"
        ],
        "mri_dice_empty": [
            "def dice", "dice_score", "delta_dice", "both_empty", "empty", "denom", "denominator",
            "return 1.0", "return 0.0", "harm"
        ],
        "mask_22_to_16": [
            "def unpack_mask_occupancy16", "occupancy16", "22x22", "patch_grid", "reshape", "interpolate", "resize"
        ],
        "dinov2_preprocess": [
            "AutoImageProcessor", "from_pretrained", "do_resize", "do_center_crop", "pixel_values",
            "bicubic", "resize", "image_mean", "image_std"
        ],
    }

    for category, candidates in all_candidates.items():
        dlist = []
        for c in candidates[:6]:
            text = text_cache.get(c.path, "")
            if not text:
                continue
            item = {
                "candidate": asdict(c),
                "ast_blocks": extract_ast_blocks(text, category) if c.path.lower().endswith(".py") else [],
                "contexts": extract_regex_contexts(text, regex_patterns[category]),
            }
            dlist.append(item)
        details[category] = dlist

    serialized = inspect_serialized_estimators()

    report = {
        "version": VERSION,
        "root": str(ROOT),
        "read_only": True,
        "candidate_file_count": len(files),
        "candidates": {k: [asdict(x) for x in v] for k, v in all_candidates.items()},
        "details": details,
        "serialized_estimators": serialized,
        "requested_closure_items": [
            "TENT1 exact optimizer/lr/updated-parameter scope/loss for colonoscopy and MRI",
            "MRI/PROMISE12 both-empty Dice rule",
            "exact 352->22x22 occupancy->16x16 mapping",
            "DINOv2 preprocessing and processor normalization behavior",
            "PCA and logistic-regression serialized estimator parameters",
        ],
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = []
    lines.append("===== SAFETTA FINAL METHOD IMPLEMENTATION AUDIT =====")
    lines.append(f"Version: {VERSION}")
    lines.append(f"Root: {ROOT}")
    lines.append("Read-only scan: YES")
    lines.append("Project scripts executed/imported: NO")
    lines.append(f"Candidate files scanned: {len(files)}")
    lines.append("")

    for category in ["tent_action", "mri_dice_empty", "mask_22_to_16", "dinov2_preprocess"]:
        lines.append("=" * 90)
        lines.append(category.upper())
        lines.append("=" * 90)
        cand = all_candidates[category]
        if not cand:
            lines.append("NO CANDIDATE FOUND")
            lines.append("")
            continue
        for rank, c in enumerate(cand, start=1):
            lines.append(f"[{rank}] score={c.score} sha256={c.sha256}")
            lines.append(f"    {c.path}")
            lines.append(f"    matched={c.matched}")
        lines.append("")
        for item in details[category]:
            c = item["candidate"]
            lines.append("-" * 90)
            lines.append(f"DETAIL: {c['path']}")
            for b in item["ast_blocks"][:5]:
                lines.append(f"\n[AST {b['kind']} {b['name']} lines {b['start']}-{b['end']} score={b['score']}]\n")
                lines.append(str(b["snippet"]))
            for ctx in item["contexts"][:6]:
                lines.append(f"\n[CONTEXT around line {ctx['line']} hits={ctx['hits']}]\n")
                lines.append(str(ctx["snippet"]))
            lines.append("")

    lines.append("=" * 90)
    lines.append("SERIALIZED PCA / SAFETY HEAD")
    lines.append("=" * 90)
    lines.append(json.dumps(serialized, indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("NEXT: send this TXT report back for manuscript v9 closure.")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\nTop candidates:")
    for category, cand in all_candidates.items():
        print(f"  {category}: {len(cand)}")
        for c in cand[:3]:
            print(f"    score={c.score:3d} {Path(c.path).name}")
    print("\nSerialized estimator inspection:", serialized.get("status"))
    print("\nOutputs:")
    print(" ", REPORT_TXT)
    print(" ", REPORT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
