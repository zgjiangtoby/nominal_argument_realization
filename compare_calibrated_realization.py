"""Combine the eight paired calibrated-realization comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_calibrated_realization import summarize_difference
from entity_analysis import holm

MODELS = ("Qwen3-8B", "EuroLLM-1.7B-Instruct")
COMPARISONS = {"calibrated_context": ("D_cal_HPC", "D_cal_HP"),
               "calibrated_vs_raw": ("D_cal_HPC", "D_raw_HPC")}


def adjusted_eight(rows):
    expected = {(model, split, comparison) for model in MODELS for split in ("document", "lemma") for comparison in COMPARISONS}
    assert len(rows) == 8 and {(r["model"], r["split"], r["name"]) for r in rows} == expected
    inputs = [1. if r["normal_p_before_Holm"] is None else r["normal_p_before_Holm"] for r in rows]
    for row, p, adjusted in zip(rows, inputs, holm(inputs)):
        row["holm_family_input"] = p
        row["holm_p_eight"] = None if row["normal_p_before_Holm"] is None else adjusted
    return all(r["normal_p_before_Holm"] is not None for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--euro", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    paths = dict(zip(MODELS, (args.qwen, args.euro)))
    summaries = {model: json.loads((path/"summary.json").read_text()) for model, path in paths.items()}
    rows, sources = [], []
    for model, summary in summaries.items():
        assert summary["complete"] and summary["model"] == model and summary["post_result_sensitivity"]
        assert summary["original_primary_replaced"] is False
        assert set(summary["results"]) == {"document", "lemma"}
    assert len({s["readme_sha256"] for s in summaries.values()}) == 1
    for split in ("document", "lemma"):
        provenance = [json.loads((path/f"private_{split}_draw_provenance.json").read_text()) for path in paths.values()]
        assert provenance[0] == provenance[1]
        assert provenance[0]["seed"] == 20261009 and len(provenance[0]["multiplicity_sha256"]) == 2000
        assert len(provenance[0]["sorted_groups"]) == {"document": 719, "lemma": 471}[split]
        for model, path in paths.items():
            result = summaries[model]["results"][split]
            assert result["fixed_draws"] == 2000 and result["refit_draws"] == 200
            assert result["nominal_pairs"] == 1759 and result["discordant_pairs"] == 585 and result["role_rows"] == 3518
            assert result["groups"] == len(provenance[0]["sorted_groups"])
            assert result["multiplicity_digest_list_sha256"] == hashlib.sha256(json.dumps(provenance[0]["multiplicity_sha256"]).encode()).hexdigest()
            primary = [r for r in result["comparisons"] if r["followup_family"]]
            assert len(primary) == 2 and {r["name"] for r in primary} == set(COMPARISONS)
            with np.load(path/f"private_{split}_draws.npz") as draws:
                for source in primary:
                    new, old = COMPARISONS[source["name"]]
                    assert source["new"] == new and source["old"] == old
                    fixed = draws["fixed__"+new]-draws["fixed__"+old]
                    refit = draws["refit__"+new]-draws["refit__"+old]
                    assert np.isfinite(fixed).all() and np.isfinite(refit).all()
                    point = result["metrics"][new]-result["metrics"][old]
                    recomputed = summarize_difference(point, fixed, refit)
                    for key, value in recomputed.items():
                        actual = source[key]
                        if value is None or isinstance(value, str):
                            assert actual == value
                        else:
                            assert np.allclose(actual, value, atol=1e-12, rtol=1e-12)
                    rows.append({"model": model, "split": split, **source})
            sources.append({"model": model, "split": split,
                "summary_sha256": hashlib.sha256((path/"summary.json").read_bytes()).hexdigest(),
                "draws_sha256": hashlib.sha256((path/f"private_{split}_draws.npz").read_bytes()).hexdigest(),
                "provenance_sha256": hashlib.sha256((path/f"private_{split}_draw_provenance.json").read_bytes()).hexdigest()})
    complete = adjusted_eight(rows)
    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.output/"eight_comparisons.csv", index=False)
    (args.output/"summary.json").write_text(json.dumps({"complete": True, "inference_complete": complete,
        "comparisons": rows, "sources": sources, "post_result_sensitivity": True,
        "original_primary_replaced": False, "family_size": 8,
        "scope": "same fixed folds and LLM scores; 2000 whole-calibration-population resamples, 200 jointly refit temperatures and heads; separate document/lemma axes",
        "p_value_scope": "two-sided maximum-SE normal approximation, Holm across all eight follow-up comparisons; not preregistered confirmatory inference"},
        indent=2, allow_nan=False)+"\n")
    print(json.dumps({"complete": True, "inference_complete": complete, "family_size": 8}), flush=True)


if __name__ == "__main__":
    main()
