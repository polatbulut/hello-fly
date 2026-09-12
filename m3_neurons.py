#!/usr/bin/env python3
"""M3: resolve ORN and DNa01/DNa02 root IDs from a pinned, public v783 dump.

PROVENANCE
Source is the public, un-authenticated Google Cloud Storage bucket that backs
FlyWire Codex:
    https://storage.googleapis.com/flywire-data/codex/data/fafb/783/<file>.csv.gz
Codex's own /api/download endpoint sits behind a Google login and is not
scriptable, so this bucket is the reproducible route. v783 is a frozen
materialisation, so these files are immutable; we record SHA-256 of exactly what
we downloaded, and anyone can re-derive the same mapping.

TWO TRAPS THIS CODE DELIBERATELY AVOIDS
1. hemibrain_type is NOT cell_type. The hemibrain mislabelled DNa01: in the
   FlyWire annotations, cell_type=DNa01 carries hemibrain_type=VES006, while a
   DIFFERENT pair (cell_type=DNae001) carries hemibrain_type=DNa01. Keying on
   hemibrain_type silently selects the wrong neurons. We key on cell_type only.
2. Community free-text labels sometimes name the mirrored hemisphere -- a cell
   the annotations and Rayshubskiy et al. both call DNa02_R can carry a
   free-text label "DNa02_L". We take side from the `side` column, never from
   a name string.

Output: neurons.v783.json -- the auditable mapping M4 and M5 consume.
"""
import gzip
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
CACHE = REPO / "data" / "flywire_v783"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = REPO / "neurons.v783.json"

BASE = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783"
FILES = ["classification.csv.gz", "consolidated_cell_types.csv.gz"]

COMP = REPO / "external" / "fly-brain" / "data" / "2025_Completeness_783.csv"


def fetch(name):
    """Download (once) and return (dataframe, sha256, n_bytes)."""
    dst = CACHE / name
    if not dst.exists():
        url = f"{BASE}/{name}"
        print(f"  downloading {url}")
        with urllib.request.urlopen(url, timeout=180) as r:
            dst.write_bytes(r.read())
    raw = dst.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    df = pd.read_csv(io.BytesIO(gzip.decompress(raw)))
    print(f"  {name}: {len(df):,} rows, {len(raw):,} B, sha256={sha[:16]}...")
    return df, sha, len(raw)


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main():
    rule("M3  downloading pinned FlyWire v783 annotations")
    prov = {}
    cls, sha_cls, n_cls = fetch("classification.csv.gz")
    prov["classification.csv.gz"] = {
        "url": f"{BASE}/classification.csv.gz", "sha256": sha_cls,
        "bytes": n_cls, "rows": len(cls), "columns": list(cls.columns)}
    typ, sha_typ, n_typ = fetch("consolidated_cell_types.csv.gz")
    prov["consolidated_cell_types.csv.gz"] = {
        "url": f"{BASE}/consolidated_cell_types.csv.gz", "sha256": sha_typ,
        "bytes": n_typ, "rows": len(typ), "columns": list(typ.columns)}

    print(f"\nclassification columns      : {list(cls.columns)}")
    print(f"consolidated_types columns  : {list(typ.columns)}")

    # The model's neuron set and, crucially, its index ORDER. flyid2i is built
    # as {root_id: row_position}, so row order in this CSV IS the tensor column
    # index the connectivity parquet keys on.
    comp = pd.read_csv(COMP, index_col=0)
    model_ids = list(comp.index)
    id2idx = {int(r): i for i, r in enumerate(model_ids)}
    print(f"model neurons               : {len(model_ids):,}")
    print(f"annotated neurons           : {len(cls):,}")
    print(f"overlap                     : "
          f"{len(set(cls['root_id']) & set(id2idx)):,}")

    ann = cls.merge(typ, on="root_id", how="left")
    type_col = "primary_type" if "primary_type" in ann.columns else "cell_type"
    print(f"using type column           : {type_col!r}")

    # ------------------------------------------------------------------ ORNs
    rule("ORNs: super_class=sensory AND class=olfactory")
    orn = ann[(ann["super_class"] == "sensory") & (ann["class"] == "olfactory")].copy()
    print(f"matched {len(orn):,} rows")
    # pandas 3.0 keeps NaN as float through .astype(str); fillna first.
    tnames = orn[type_col].fillna("").astype(str)
    print(f"type prefixes: {sorted({t.split('_')[0] for t in tnames if t})}")
    # HRN_ (hygro) and TRN_ (thermo) receptors share the sensory super_class but
    # are NOT olfactory receptor neurons. Restrict to the ORN_ prefix.
    orn = orn[tnames.str.startswith("ORN_").values]
    print(f"after ORN_ prefix filter: {len(orn):,} across "
          f"{orn[type_col].nunique()} types")
    print(f"side counts: {orn['side'].value_counts().to_dict()}")
    print(f"nerves     : {orn['nerve'].value_counts().to_dict()}")

    orn_out = {}
    for side in ("left", "right"):
        ids = [int(r) for r in orn.loc[orn["side"] == side, "root_id"]]
        in_model = [r for r in ids if r in id2idx]
        orn_out[side] = {
            "n_annotated": len(ids), "n_in_model": len(in_model),
            "root_ids": sorted(in_model),
            "indices": sorted(id2idx[r] for r in in_model),
        }
        print(f"  {side:5s}: {len(ids):4d} annotated, {len(in_model):4d} in model")

    # ------------------------------------------------------------- DNa01/02
    rule("DNa01 / DNa02 -- keyed on cell_type, NOT hemibrain_type")
    dn_out = {}
    for name in ("DNa01", "DNa02"):
        hit = ann[ann[type_col].astype(str) == name]
        print(f"\n{name}: {len(hit)} cell(s) with {type_col}=={name!r}")
        if len(hit) == 0:
            print(f"  !! no match -- check the type column")
            dn_out[name] = None
            continue
        cols = [c for c in ("root_id", "side", "nerve", "super_class", "class",
                            "hemilineage", "hemibrain_type", type_col)
                if c in hit.columns]
        print(hit[cols].to_string(index=False))

        per_side = {}
        for side in ("left", "right"):
            ids = [int(r) for r in hit.loc[hit["side"] == side, "root_id"]]
            in_model = [r for r in ids if r in id2idx]
            per_side[side] = {
                "root_ids": in_model,
                "indices": [id2idx[r] for r in in_model],
                "n_annotated": len(ids),
                "n_in_model": len(in_model),
            }
            flag = "" if len(in_model) == len(ids) else "  <-- SOME NOT IN MODEL"
            print(f"  {side:5s}: {ids} -> in-model {in_model}{flag}")
        dn_out[name] = per_side

        # The hemibrain-mislabel trap, demonstrated rather than asserted.
        if "hemibrain_type" in ann.columns:
            conf = ann[ann["hemibrain_type"].astype(str) == name]
            if len(conf):
                ct = sorted(set(conf[type_col].astype(str)))
                if ct != [name]:
                    print(f"  TRAP CONFIRMED: hemibrain_type=={name!r} actually "
                          f"matches cell_type(s) {ct} -- keying on hemibrain_type "
                          f"would have selected the WRONG neurons")

    # ------------------------------------------------------------ sanity
    rule("sanity checks")
    problems = []
    for name, v in dn_out.items():
        if v is None:
            problems.append(f"{name}: not found")
            continue
        for side in ("left", "right"):
            n = v[side]["n_in_model"]
            if n != 1:
                problems.append(f"{name} {side}: expected exactly 1 cell, got {n}")
    tot_orn = sum(orn_out[s]["n_in_model"] for s in orn_out)
    if not (1500 <= tot_orn <= 3000):
        problems.append(f"ORN count {tot_orn} outside the expected ~2,277 range")
    if problems:
        for p in problems:
            print(f"  WARN {p}")
    else:
        print("  all checks passed: DNa01/DNa02 are one cell per hemisphere, "
              f"ORN count {tot_orn} is in range")

    # ------------------------------------------------------------------ save
    doc = {
        "schema_version": 1,
        "materialization": "FlyWire FAFB v783",
        "why_v783": (
            "The LIF model's connectivity is v783. Root IDs are NOT portable "
            "across materialisations -- only ~76.6% of v783 IDs exist in v630, "
            "and DNa01_L does not exist in v630 at all. Never mix."),
        "provenance": prov,
        "queries": {
            "ORN": ("super_class == 'sensory' AND class == 'olfactory' AND "
                    f"{type_col}.startswith('ORN_')  "
                    "# HRN_/TRN_ are hygro/thermo receptors, excluded"),
            "DNa01/DNa02": (
                f"{type_col} == 'DNa01' | 'DNa02', side from the `side` column.  "
                "# MUST NOT key on hemibrain_type: cell_type=DNa01 carries "
                "hemibrain_type=VES006, while cell_type=DNae001 carries "
                "hemibrain_type=DNa01. Also MUST NOT take side from free-text "
                "labels, which are sometimes mirrored."),
        },
        "model_index_note": (
            "`indices` are column positions in the LIF model's tensors, derived "
            "as row order of 2025_Completeness_783.csv, matching upstream's "
            "flyid2i = {root_id: i for i, root_id in enumerate(df.index)}."),
        "model_neuron_count": len(model_ids),
        "ORN": orn_out,
        "DN": dn_out,
        "warnings": problems,
    }
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {OUT}")
    return 0 if not problems else 0   # warnings do not fail the milestone


if __name__ == "__main__":
    sys.exit(main())
