"""Fit one inverse temperature per training fold for paired-role recovery."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
from entity_analysis import cluster_summary, read_jsonl, read_probes

CELLS = (("agt", "context"), ("agt", "prior"), ("pat", "context"), ("pat", "prior"))
MODELS = {"Qwen3-8B": ("b968826d9c46dd6066d109eabc6255188de91218", 8),
          "EuroLLM-1.7B-Instruct": ("a25c7fa65fc2a644e6270b8940dbe295b51da681", 32)}


def transformed(logp, mask, target, inverse_temperature):
    assert 0 <= inverse_temperature <= 5
    scores = np.where(mask, inverse_temperature*logp, -np.inf)
    normalizer = logsumexp(scores, axis=-1)
    probability = np.exp(scores-normalizer[..., None])
    target_log = np.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    nll = normalizer-inverse_temperature*target_log
    derivative = (probability*logp).sum(axis=-1)-target_log
    assert np.isfinite(nll).all() and np.isfinite(derivative).all()
    assert np.allclose(probability.sum(axis=-1), 1., atol=1e-12, rtol=0.)
    return nll, derivative, probability


def fit_inverse_temperature(logp, mask, target, training):
    assert training.dtype == bool and training.any() and (~training).any()
    a, b, c = logp[training], mask[training], target[training]

    def derivative(value):
        return float(transformed(a, b, c, value)[1].mean())

    low, high = derivative(0.), derivative(5.)
    assert high >= low-1e-10
    if low >= 0.:
        alpha, status = 0., "uniform_boundary"
    elif high <= 0.:
        alpha, status = 5., "upper_boundary"
    else:
        alpha, status = float(brentq(derivative, 0., 5., xtol=1e-12)), "interior"
    residual = derivative(alpha)
    assert (residual >= -1e-8 if alpha == 0 else residual <= 1e-8 if alpha == 5 else abs(residual) < 1e-8)
    loss = float(transformed(a, b, c, alpha)[0].mean())
    original = float(transformed(a, b, c, 1.)[0].mean())
    uniform = float(transformed(a, b, c, 0.)[0].mean())
    assert loss <= min(original, uniform)+1e-10
    return {"inverse_temperature": alpha, "temperature": None if alpha == 0 else 1/alpha,
        "status": status, "training_nominal_pairs": int(training.sum()), "training_probes": int(training.sum()*4),
        "training_nll": loss, "training_original_nll": original, "training_uniform_nll": uniform,
        "training_derivative": residual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path, required=True)
    parser.add_argument("--model-name", required=True, choices=("Qwen3-8B", "EuroLLM-1.7B-Instruct"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    summary = json.loads((args.reference_analysis/"summary.json").read_text())
    assert summary["completed"] and Path(summary["source_scores"]).resolve() == args.scores.resolve()
    metadata_path = args.scores.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    revision, batch = MODELS[args.model_name]
    assert Path(metadata["model_path"]).name == revision
    assert metadata["dtype"] == "float32" and metadata["batch_size"] == batch
    assert metadata["rows"] == 14072 and metadata["status"] == "complete" and metadata["gold_used"] is False
    input_path = HERE/"work/probes/inputs.jsonl"
    assert Path(metadata["input"]).resolve() == input_path.resolve()
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if "input_sha256" in metadata:
        assert metadata["input_sha256"] == input_sha256
    probes = read_probes(HERE/"work/probes", args.scores)
    items = read_jsonl(HERE/"work/probes/items.jsonl")
    assert len(items) == 1759 and len({r["item_id"] for r in items}) == 1759
    n = len(items); max_candidates = max(len(r["candidates"]) for r in items)
    logp = np.zeros((n, 4, max_candidates))
    mask = np.zeros_like(logp, dtype=bool)
    target, alternate = np.zeros((n, 4), dtype=int), np.zeros((n, 4), dtype=int)
    counts = np.array([len(r["candidates"]) for r in items])
    base = pd.DataFrame([{k: row[k] for k in ("item_id", "file", "lemma", "document_fold", "lemma_fold")} for row in items])
    reference = pd.read_csv(args.reference_analysis/"recovery.csv", keep_default_na=False, float_precision="round_trip")
    assert len(reference) == n*4 and not reference.duplicated(["item_id", "role", "condition"]).any()
    reference = reference.set_index(["item_id", "role", "condition"])
    clipped = 0
    for i, row in enumerate(items):
        entities = [c["entity_id"] for c in row["candidates"]]
        for cell, (role, condition) in enumerate(CELLS):
            probe = probes[row["item_id"], role, condition]
            p = np.array([probe["probabilities"][entity] for entity in entities])
            assert np.isfinite(p).all() and np.isclose(p.sum(), 1., atol=1e-12, rtol=0.)
            clipped += int((p < 1e-300).sum())
            logp[i, cell, :len(p)] = np.log(np.maximum(p, 1e-300))
            mask[i, cell, :len(p)] = True
            target[i, cell] = entities.index(row["gold_entities"][role])
            alternate[i, cell] = entities.index(row["gold_entities"]["pat" if role == "agt" else "agt"])
    raw, _, raw_p = transformed(logp, mask, target, 1.)
    uniform, _, _ = transformed(logp, mask, target, 0.)
    assert np.allclose(uniform, np.log(counts)[:, None], atol=1e-12, rtol=0.)
    reproduction = 0.
    for i, row in enumerate(items):
        for cell, (role, condition) in enumerate(CELLS):
            saved = reference.loc[(row["item_id"], role, condition), :]
            reproduction = max(reproduction, abs(raw[i, cell]-float(saved["surprisal"])))
    assert reproduction < 1e-12
    fits = []
    for split, key in (("document", "file"), ("lemma", "lemma")):
        assert base.groupby(key)[split+"_fold"].nunique().eq(1).all()
        folds = base[split+"_fold"].to_numpy(dtype=int)
        assert set(folds) == set(range(5))
        for fold in range(5):
            training = folds != fold
            assert set(base.loc[training, key]).isdisjoint(set(base.loc[~training, key]))
            record = fit_inverse_temperature(logp, mask, target, training)
            fits.append({"split": split, "fold": fold, **record})
    args.output.mkdir(parents=True)
    (args.output/"temperature_freeze.json").write_text(json.dumps({"fits": fits, "post_result": True,
        "model": args.model_name, "revision": revision, "source_scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
        "source_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "scoring_input_sha256_at_diagnostic": input_sha256,
        "input_sha256_recorded_at_scoring": metadata.get("input_sha256"),
        "reference_recovery_sha256": hashlib.sha256((args.reference_analysis/"recovery.csv").read_bytes()).hexdigest(),
        "probe_items_sha256": hashlib.sha256((HERE/"work/probes/items.jsonl").read_bytes()).hexdigest(),
        "fitting_target": "existing entity/role links; no realization outcome", "new_calibrated_evaluation_summaries_computed": False,
        "original_outcomes_already_known": True, "all_raw_recovery_reproduction_max_abs_difference": reproduction,
        "probabilities_below_1e_300": clipped, "readme_sha256": hashlib.sha256((HERE/"README.md").read_bytes()).hexdigest()},
        indent=2, allow_nan=False)+"\n")
    results = {}
    raw_gold = np.take_along_axis(raw_p, target[..., None], axis=-1)[..., 0]
    raw_other = np.take_along_axis(raw_p, alternate[..., None], axis=-1)[..., 0]
    for split, key in (("document", "file"), ("lemma", "lemma")):
        out = np.full_like(raw, np.nan); margins = np.full_like(raw, np.nan)
        for record in (r for r in fits if r["split"] == split):
            selected = base[split+"_fold"].eq(record["fold"]).to_numpy()
            loss, _, p = transformed(logp[selected], mask[selected], target[selected], record["inverse_temperature"])
            out[selected] = loss
            margins[selected] = (np.take_along_axis(p, target[selected, :, None], axis=-1)[..., 0]-
                                 np.take_along_axis(p, alternate[selected, :, None], axis=-1)[..., 0])
        assert np.isfinite(out).all() and np.isfinite(margins).all()
        cells, contexts, rows = {}, {}, []
        for cell, (role, condition) in enumerate(CELLS):
            frame = base.copy()
            frame["role"], frame["condition"] = role, condition
            frame["raw_nll"], frame["calibrated_nll"] = raw[:, cell], out[:, cell]
            frame["calibrated_minus_raw"] = out[:, cell]-raw[:, cell]
            frame["calibrated_minus_uniform"] = out[:, cell]-uniform[:, cell]
            frame["raw_gold_minus_opposite_role_probability"] = raw_gold[:, cell]-raw_other[:, cell]
            frame["calibrated_gold_minus_opposite_role_probability"] = margins[:, cell]
            cells[role+":"+condition] = cluster_summary(frame, ["raw_nll", "calibrated_nll", "calibrated_minus_raw",
                "calibrated_minus_uniform", "raw_gold_minus_opposite_role_probability", "calibrated_gold_minus_opposite_role_probability"], key, False)
            rows.append(frame)
        for role, context, prior in (("agt", 0, 1), ("pat", 2, 3)):
            frame = base.copy()
            frame["raw_context_minus_prior"] = raw[:, context]-raw[:, prior]
            frame["calibrated_context_minus_prior"] = out[:, context]-out[:, prior]
            contexts[role] = cluster_summary(frame, ["raw_context_minus_prior", "calibrated_context_minus_prior"], key, False)
        pd.concat(rows, ignore_index=True).to_csv(args.output/f"private_{split}_recovery.csv", index=False)
        results[split] = {"cells": cells, "context_contrasts": contexts}
    report = {"complete": True, "model": args.model_name, "source_scores": str(args.scores), "fits": fits,
        "results": results, "post_result_diagnostic": True, "primary_realization_results_replaced": False,
        "scope": "fixed out-of-fold temperature measurements; bootstrap intervals do not refit temperatures or establish complete training uncertainty"}
    (args.output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"complete": True, "model": args.model_name, "temperature_fits": len(fits)}), flush=True)


if __name__ == "__main__":
    main()
