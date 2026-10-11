#!/usr/bin/env python3
"""Estimate paired risk differences and conditional direction models."""
import argparse
import csv
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


SEED = 20261008
BOOTSTRAPS = 2000
STRATA = ("A0P0", "A1P0", "A0P1", "A1P1")
HISTORY = ("delta_log_terminal_distance", "delta_log_prior_count",
           "delta_prior_subject", "delta_prior_function_missing")
CONTEXT = ("A1P0", "A0P1", "A1P1")


def fit_direction(x, y):
    scale = np.r_[StandardScaler(with_mean=False).fit(x[:, :4]).scale_,
                  np.ones(x.shape[1] - 4)]
    model = LogisticRegression(C=1, penalty="l2", solver="lbfgs", tol=1e-9,
                               max_iter=2000).fit(x / scale, y)
    return model, scale


def metrics(y, probability):
    return {"log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "brier": float(brier_score_loss(y, probability)),
            "accuracy": float(accuracy_score(y, probability >= .5))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    coverage = json.loads((args.coverage / "coverage_summary.json").read_text())
    if not coverage["complete_manifest"] or coverage["failed_files"]:
        raise SystemExit("Full source coverage is required.")
    all_rows = [json.loads(line) for line in (args.coverage / "nominal_pairs.jsonl").open()]
    contexts = {(r["file"], r["nominal_path"]): r for r in
                map(json.loads, (args.coverage / "pair_context.jsonl").open())}
    documents = sorted(r["file"] for r in map(json.loads, (args.coverage / "by_file.jsonl").open()))
    doc_index = {name: i for i, name in enumerate(documents)}
    assert len(documents) == coverage["manifest_files"]
    rows = [r for r in all_rows if r["single_entity_prior_pair"] and not r["same_entity_set"]]
    units = []
    for row in rows:
        context = contexts[row["file"], row["nominal_path"]]
        roles = [row["roles"][role] for role in ("agt", "pat")]
        histories = [r["history"][0] for r in roles]
        assert all(h["prior_count"] > 0 and h["terminal_distance"] >= 0 for h in histories)
        vectors = [np.array([np.log1p(h["terminal_distance"]), np.log1p(h["prior_count"]),
                            h["latest"]["func"] == "suj", h["latest"]["func"] is None], dtype=float)
                   for h in histories]
        units.append({"file": row["file"], "nominal_path": row["nominal_path"],
                      "lemma": row["lemma"], "lexicalid": row["lexicalid"],
                      "pattern": row["pattern"],
                      "stratum": context["symmetric_external_same_sentence_stratum"],
                      "agent_minus_patient_direct": int(roles[0]["realization"] == "E") -
                                                    int(roles[1]["realization"] == "E"),
                      **dict(zip(HISTORY, vectors[0] - vectors[1]))})
    n_docs = len(documents)
    numerator = np.zeros((n_docs, 5))
    denominator = np.zeros((n_docs, 5))
    for unit in units:
        i = doc_index[unit["file"]]
        for j in (0, 1 + STRATA.index(unit["stratum"])):
            denominator[i, j] += 1
            numerator[i, j] += unit["agent_minus_patient_direct"]
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(n_docs, np.full(n_docs, 1 / n_docs), size=BOOTSTRAPS)
    boot_den = weights @ denominator
    assert np.all(boot_den > 0)
    boot_diff = (weights @ numerator) / boot_den
    differences = {}
    for j, label in enumerate(("all",) + STRATA):
        subset = [u for u in units if label == "all" or u["stratum"] == label]
        patterns = Counter(u["pattern"] for u in subset)
        differences[label] = {"n": len(subset), "documents": len({u["file"] for u in subset}),
            "lemmas": len({u["lemma"] for u in subset}),
            "patterns": {k: patterns[k] for k in ("E/E", "E/I", "I/E", "I/I")},
            "agent_direct_probability": (patterns["E/E"] + patterns["E/I"]) / len(subset),
            "patient_direct_probability": (patterns["E/E"] + patterns["I/E"]) / len(subset),
            "agent_minus_patient_direct": float(numerator[:, j].sum() / denominator[:, j].sum()),
            "document_bootstrap_95_percentile_interval": np.quantile(boot_diff[:, j], [.025, .975]).tolist()}
    discordant = [u for u in units if u["pattern"] in ("I/E", "E/I")]
    y = np.array([u["pattern"] == "I/E" for u in discordant], dtype=int)
    x = np.array([[u[name] for name in HISTORY] + [u["stratum"] == s for s in CONTEXT]
                  for u in discordant], dtype=float)
    fitted = {"M0": {"p_I_E": float(y.mean()), "intercept": float(np.log(y.mean() / (1 - y.mean())))}}
    for name, width in (("M1", 4), ("M2", 7)):
        model, scale = fit_direction(x[:, :width], y)
        fitted[name] = {"intercept": float(model.intercept_[0]),
                       "p_I_E_at_equal_history_reference_context": float(1 / (1 + np.exp(-model.intercept_[0]))),
                       "coefficients_original_units": dict(zip((HISTORY + CONTEXT)[:width],
                                                               (model.coef_[0] / scale).tolist())),
                       "training_scales": scale.tolist(), "iterations": model.n_iter_.tolist()}
    predictions, heldout = [], {}
    for split_name, group_key in (("document", "file"), ("lemma", "lemma")):
        groups = np.array([u[group_key] for u in discordant])
        fold_ids = np.full(len(y), -1)
        probabilities = {name: np.full(len(y), np.nan) for name in fitted}
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
            assert not set(groups[train]) & set(groups[test])
            assert len(np.unique(y[train])) == 2
            fold_ids[test] = fold
            probabilities["M0"][test] = y[train].mean()
            for name, width in (("M1", 4), ("M2", 7)):
                model, scale = fit_direction(x[train, :width], y[train])
                probabilities[name][test] = model.predict_proba(x[test, :width] / scale)[:, 1]
        assert np.all(fold_ids >= 0) and all(np.isfinite(p).all() for p in probabilities.values())
        heldout[split_name] = {name: metrics(y, prob) for name, prob in probabilities.items()}
        for i, unit in enumerate(discordant):
            predictions.append({"split": split_name, "fold": int(fold_ids[i]),
                                "file": unit["file"], "nominal_path": unit["nominal_path"],
                                "lemma": unit["lemma"], "stratum": unit["stratum"], "y_I_E": int(y[i]),
                                **{name: float(prob[i]) for name, prob in probabilities.items()}})
    output = {"source_commit": coverage["source_commit"], "seed": SEED,
              "bootstrap_replicates": BOOTSTRAPS, "bootstrap_document_universe": n_docs,
              "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
              "risk_differences": differences, "conditional_direction_n": len(y),
              "conditional_direction_models": fitted, "heldout_prediction": heldout,
              "support_construction_gold": "unknown_no_verified_lexicon_binding",
              "coefficient_intervals": "not_estimated"}
    assert sum(differences[s]["n"] for s in STRATA) == differences["all"]["n"]
    assert np.isclose(differences["all"]["agent_minus_patient_direct"],
                      (sum(u["pattern"] == "E/I" for u in units) -
                       sum(u["pattern"] == "I/E" for u in units)) / len(units))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for filename, records in (("analysis_units.csv", units), ("heldout_predictions.csv", predictions)):
        with (args.output / filename).open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    np.savez_compressed(args.output / "document_bootstrap.npz", differences=boot_diff,
                        denominators=boot_den, labels=np.array(("all",) + STRATA))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    warnings.simplefilter("error", ConvergenceWarning)
    main()
