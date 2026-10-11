"""Refit recovery temperature and realization heads within each outer fold."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
import entity_analysis
from entity_analysis import read_jsonl, read_probes, fit_predict, binary_losses
from diagnose_role_confidence import CELLS, transformed

D_HISTORY = ["delta_log_terminal_distance", "delta_log_prior_count", "delta_prior_subject", "delta_prior_function_missing"]
R_HISTORY = ["log_terminal_distance", "log_prior_count", "prior_subject", "prior_function_missing"]
NAMES = ("D_raw_HP", "D_raw_HPC", "D_cal_HP", "D_cal_HPC", "R_raw", "R_raw_role", "R_cal", "R_cal_role")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")


def fit_temperature(logp, mask, target, weights, training):
    selected = training & (weights > 0)
    assert selected.any() and np.isfinite(weights).all() and (weights >= 0).all()
    a, b, c, w = logp[selected], mask[selected], target[selected], weights[selected]

    def loss_derivative(value):
        loss, derivative, _ = transformed(a, b, c, value)
        return float(np.average(loss.mean(axis=1), weights=w)), float(np.average(derivative.mean(axis=1), weights=w))

    low, high = loss_derivative(0.)[1], loss_derivative(5.)[1]
    assert high >= low-1e-10
    if low >= 0:
        alpha, status = 0., "uniform_boundary"
    elif high <= 0:
        alpha, status = 5., "upper_boundary"
    else:
        alpha, status = float(brentq(lambda value: loss_derivative(value)[1], 0., 5., xtol=1e-12)), "interior"
    objective, gradient = loss_derivative(alpha)
    assert (gradient >= -1e-8 if alpha == 0 else gradient <= 1e-8 if alpha == 5 else abs(gradient) < 1e-8)
    assert objective <= min(loss_derivative(0.)[0], loss_derivative(1.)[0])+1e-10
    return {"inverse_temperature": alpha, "status": status, "training_nll": objective,
            "derivative": gradient, "training_distinct_pairs": int(selected.sum()),
            "training_expanded_pairs": int(weights[selected].sum())}


def load_data(args):
    diagnostic = json.loads((args.confidence/"summary.json").read_text())
    freeze = json.loads((args.confidence/"temperature_freeze.json").read_text())
    old = json.loads((args.reference_analysis/"summary.json").read_text())
    assert diagnostic["complete"] and old["completed"]
    assert diagnostic["model"] == freeze["model"] == args.model_name
    assert Path(diagnostic["source_scores"]).resolve() == Path(old["source_scores"]).resolve() == args.scores.resolve()
    assert sha(args.scores) == freeze["source_scores_sha256"]
    assert sha(args.scores.with_suffix(".metadata.json")) == freeze["source_metadata_sha256"]
    assert sha(HERE/"work/probes/inputs.jsonl") == freeze["scoring_input_sha256_at_diagnostic"]
    assert sha(HERE/"work/probes/items.jsonl") == freeze["probe_items_sha256"]
    assert sha(args.reference_analysis/"recovery.csv") == freeze["reference_recovery_sha256"]
    assert freeze["probabilities_below_1e_300"] == 0
    items = read_jsonl(HERE/"work/probes/items.jsonl")
    probes = read_probes(HERE/"work/probes", args.scores)
    assert len(items) == 1759
    frame = read_csv(args.reference_analysis/"features.csv")
    roles = read_csv(args.reference_analysis/"paired_role_features.csv")
    ids = [r["item_id"] for r in items]
    assert frame.item_id.tolist() == ids and frame.item_id.is_unique
    assert list(zip(roles.item_id, roles.role)) == [(identifier, role) for identifier in ids for role in ("agt", "pat")]
    for key in ("file", "lemma", "document_fold", "lemma_fold", "pattern"):
        assert frame[key].tolist() == [r[key] for r in items]
    for key in ("file", "lemma", "document_fold", "lemma_fold"):
        assert roles[key].tolist() == np.repeat(frame[key].to_numpy(), 2).tolist()
    assert roles.patient.tolist() == [0, 1]*len(items)
    assert roles.y.tolist() == [int(r["pattern"][position] == "E") for r in items for position in (0, 2)]
    assert frame.y.tolist() == [int(r["pattern"] == "I/E") for r in items]
    counts = np.array([len(r["candidates"]) for r in items])
    assert (counts >= 2).all()
    logp = np.zeros((len(items), 4, int(counts.max())))
    mask = np.zeros_like(logp, dtype=bool)
    target = np.zeros((len(items), 4), dtype=int)
    opposite = np.zeros_like(target)
    for i, row in enumerate(items):
        entities = [c["entity_id"] for c in row["candidates"]]
        for j, (role, condition) in enumerate(CELLS):
            p = np.array([probes[row["item_id"], role, condition]["probabilities"][e] for e in entities])
            assert np.isfinite(p).all() and (p >= 1e-300).all() and np.isclose(p.sum(), 1., atol=1e-12, rtol=0)
            logp[i, j, :len(p)] = np.log(p)
            mask[i, j, :len(p)] = True
            target[i, j] = entities.index(row["gold_entities"][role])
            opposite[i, j] = entities.index(row["gold_entities"]["pat" if role == "agt" else "agt"])
            assert target[i, j] != opposite[i, j]
    raw = transformed(logp, mask, target, 1.)[0]
    reproduction = 0.
    for j, (role, condition) in enumerate(CELLS):
        reproduction = max(reproduction, float(np.max(np.abs(raw[:, j]-frame[role+"_"+condition+"_surprisal"].to_numpy()))))
    assert reproduction < 1e-12
    discordant = np.flatnonzero(frame.pattern.isin(["I/E", "E/I"]).to_numpy())
    assert len(discordant) == 585
    return {"frame": frame, "roles": roles, "items": items, "logp": logp, "mask": mask,
            "target": target, "opposite": opposite, "counts": counts, "discordant": discordant,
            "diagnostic": diagnostic, "old": old, "raw_reproduction": reproduction}


def matrices(data, losses):
    frame, roles, selected = data["frame"], data["roles"], data["discordant"]
    prior_difference = losses[:, 1]-losses[:, 3]
    context_difference = losses[:, 0]-losses[:, 2]-prior_difference
    dh = frame.loc[selected, D_HISTORY].to_numpy(dtype=float)
    rh = roles[R_HISTORY].to_numpy(dtype=float)
    role_prior = losses[:, [1, 3]].ravel()
    role_context = (losses[:, [0, 2]]-losses[:, [1, 3]]).ravel()
    d_raw_hp = frame.loc[selected, D_HISTORY+["prior_role_surprisal_difference"]].to_numpy(dtype=float)
    d_raw_hpc = frame.loc[selected, D_HISTORY+["prior_role_surprisal_difference", "context_role_surprisal_increment"]].to_numpy(dtype=float)
    r_raw = roles[R_HISTORY+["prior_surprisal", "context_surprisal_increment"]].to_numpy(dtype=float)
    patient = roles.patient.to_numpy(dtype=float)
    return {"D_raw_HP": d_raw_hp, "D_raw_HPC": d_raw_hpc,
            "D_cal_HP": np.column_stack((dh, prior_difference[selected])),
            "D_cal_HPC": np.column_stack((dh, prior_difference[selected], context_difference[selected])),
            "R_raw": r_raw, "R_raw_role": np.column_stack((r_raw, patient)),
            "R_cal": np.column_stack((rh, role_prior, role_context)),
            "R_cal_role": np.column_stack((rh, role_prior, role_context, patient))}


def construction_values(losses, margins, counts):
    result = {}
    for role, context, prior in (("agt", 0, 1), ("pat", 2, 3)):
        result.update({role+":context_nll": losses[:, context], role+":prior_nll": losses[:, prior],
                       role+":context_minus_uniform": losses[:, context]-np.log(counts),
                       role+":context_minus_prior": losses[:, context]-losses[:, prior],
                       role+":gold_minus_opposite": margins[:, context]})
    return result


def summarize_difference(point, fixed, refit):
    assert len(fixed) == 2000 and len(refit) == 200
    fixed_se, refit_se = float(np.std(fixed, ddof=1)), float(np.std(refit, ddof=1))
    se = max(fixed_se, refit_se)
    if se <= 1e-12:
        p, state = (1., "numerically_zero_contrast") if abs(point) <= 1e-12 else (None, "nonzero_contrast_degenerate_se")
    else:
        p, state = float(2*norm.sf(abs(point)/se)), "normal_approximation"
    return {"delta_nll": float(point), "fixed_pipeline_95_interval": np.quantile(fixed, [.025, .975]).tolist(),
            "full_pipeline_refit_95_interval": np.quantile(refit, [.025, .975]).tolist(),
            "fixed_se": fixed_se, "refit_se": refit_se, "maximum_se": se,
            "normal_p_before_Holm": p, "normal_inference_status": state, "fixed_draws": 2000, "refit_draws": 200}


def run_split(data, split, output, reference):
    frame, roles, d_index = data["frame"], data["roles"], data["discordant"]
    group_key = "file" if split == "document" else "lemma"
    groups, group_index = np.unique(frame[group_key].astype(str), return_inverse=True)
    folds = frame[split+"_fold"].to_numpy(dtype=int)
    assert len(groups) == {"document": 719, "lemma": 471}[split]
    assert set(folds) == set(range(5)) and frame.groupby(group_key)[split+"_fold"].nunique().eq(1).all()
    population = len(frame)
    row_indices = {"D": d_index, "R": np.repeat(np.arange(population), 2)}
    outcomes = {"D": frame.y.to_numpy(dtype=int)[d_index], "R": roles.y.to_numpy(dtype=int)}
    fits, full_fits, temperature_records, fallback_records = [], {}, [], []
    maximum_gradient, fit_calls = 0., 0

    def record_fit(record, name, draw, fold, phase):
        nonlocal maximum_gradient, fit_calls
        fit_calls += 1
        maximum_gradient = max(maximum_gradient, record["gradient_max_per_training_weight"])
        if draw is None:
            fits.append({"model": name, "fold": fold, "phase": phase, **record})
        if record["fallback"] is not None:
            fallback_records.append({"model": name, "draw": draw, "fold": fold, "phase": phase, **record})

    def pipeline(pair_weights, draw):
        predictions = {name: np.full(len(outcomes[name[0]]), np.nan) for name in NAMES}
        oof_loss = np.full((population, 4), np.nan)
        oof_margin = np.full_like(oof_loss, np.nan)
        temperatures = []
        for fold in range(5):
            record = fit_temperature(data["logp"], data["mask"], data["target"], pair_weights, folds != fold)
            alpha = record["inverse_temperature"]
            temperatures.append(alpha)
            if draw is None:
                saved = next(r for r in data["diagnostic"]["fits"] if r["split"] == split and r["fold"] == fold)
                assert abs(alpha-saved["inverse_temperature"]) < 1e-10
                temperature_records.append({"fold": fold, **record})
            losses, _, probability = transformed(data["logp"], data["mask"], data["target"], alpha)
            margin = (np.take_along_axis(probability, data["target"][..., None], axis=-1)[..., 0]-
                      np.take_along_axis(probability, data["opposite"][..., None], axis=-1)[..., 0])
            oof_loss[folds == fold] = losses[folds == fold]
            oof_margin[folds == fold] = margin[folds == fold]
            feature_sets = matrices(data, losses)
            for name in NAMES:
                index = row_indices[name[0]]
                weights = pair_weights[index]
                train, test = (folds[index] != fold) & (weights > 0), (folds[index] == fold) & (weights > 0)
                if not test.any():
                    continue
                p, fitted = fit_predict(feature_sets[name], outcomes[name[0]], weights, train, test, int(weights[train].sum()))
                predictions[name][test] = p
                record_fit(fitted, name, draw, fold, "outer_fold")
        # Full-data coefficient summaries are descriptive fits, not held-out risk estimates.
        full_temperature = fit_temperature(data["logp"], data["mask"], data["target"], pair_weights, np.ones(population, dtype=bool))
        full_losses = transformed(data["logp"], data["mask"], data["target"], full_temperature["inverse_temperature"])[0]
        full_matrices = matrices(data, full_losses)
        coefficients = {}
        for name in ("R_raw_role", "R_cal_role"):
            weights = pair_weights[row_indices["R"]]
            _, fitted = fit_predict(full_matrices[name], outcomes["R"], weights, weights > 0, np.ones(len(weights), dtype=bool), int(weights.sum()))
            coefficients[name] = fitted["coefficients_original_units"][-1]
            record_fit(fitted, name, draw, None, "full_patient_coefficient")
            if draw is None:
                full_fits[name] = fitted
        risks = {}
        for name, p in predictions.items():
            weights = pair_weights[row_indices[name[0]]]
            selected = weights > 0
            assert np.isfinite(p[selected]).all()
            risks[name] = float(np.average(binary_losses(outcomes[name[0]][selected], p[selected]), weights=weights[selected]))
        construct = construction_values(oof_loss, oof_margin, data["counts"])
        construct_risks = {key: float(np.average(value, weights=pair_weights)) for key, value in construct.items()}
        return predictions, risks, coefficients, construct, construct_risks, temperatures, full_temperature["inverse_temperature"]

    point_p, point, point_coef, construct, construct_point, _, full_temperature = pipeline(np.ones(population), None)
    archived = read_csv(reference/"heldout_predictions.csv")
    archived = archived.loc[archived.split.eq(split)].reset_index(drop=True)
    assert archived.item_id.tolist() == frame.loc[d_index, "item_id"].tolist()
    reproduction = {}
    for name, saved in (("D_raw_HP", "HP"), ("D_raw_HPC", "HPC")):
        difference = float(np.max(np.abs(point_p[name]-archived[saved].to_numpy())))
        assert difference < 1e-7
        reproduction[name] = difference
    archived_roles = read_csv(reference/f"{split}_all_patterns_heldout.csv")
    assert list(zip(archived_roles.item_id, archived_roles.role)) == list(zip(roles.item_id, roles.role))
    for name, saved in (("R_raw", "recoverability"), ("R_raw_role", "recoverability_role")):
        difference = float(np.max(np.abs(point_p[name]-archived_roles[saved].to_numpy())))
        assert difference < 1e-7
        reproduction[name] = difference
    old_coefficient = data["old"]["results"][split]["all_patterns_paired_role_model"]["full_fits"]["recoverability_role"]["coefficients_original_units"][-1]
    assert abs(point_coef["R_raw_role"]-old_coefficient) < 1e-7
    point_losses = {name: binary_losses(outcomes[name[0]], p) for name, p in point_p.items()}
    fixed = {name: [] for name in NAMES}
    refitted = {name: [] for name in NAMES}
    coefficient_draws = {name: [] for name in point_coef}
    construct_fixed = {name: [] for name in construct}
    construct_refit = {name: [] for name in construct}
    temperature_draws, full_temperature_draws, draw_hashes = [], [], []
    rng = np.random.default_rng(20261009)
    for draw in range(2000):
        multiplicity = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)))
        weights = multiplicity[group_index]
        draw_hashes.append(hashlib.sha256(multiplicity.astype("<i8").tobytes()).hexdigest())
        for name in NAMES:
            fixed[name].append(float(np.average(point_losses[name], weights=weights[row_indices[name[0]]])))
        for name, values in construct.items():
            construct_fixed[name].append(float(np.average(values, weights=weights)))
        if draw < 200:
            _, risks, coefs, _, constructed, temperatures, full_temp = pipeline(weights, draw)
            for name in NAMES:
                refitted[name].append(risks[name])
            for name in coefficient_draws:
                coefficient_draws[name].append(coefs[name])
            for name in construct:
                construct_refit[name].append(constructed[name])
            temperature_draws.append(temperatures)
            full_temperature_draws.append(full_temp)
            if (draw+1) % 20 == 0:
                print(json.dumps({"split": split, "full_pipeline_refits": draw+1}), flush=True)
    fixed = {k: np.asarray(v) for k, v in fixed.items()}
    refitted = {k: np.asarray(v) for k, v in refitted.items()}
    with np.load(reference/f"{split}_all_patterns_bootstrap.npz") as saved:
        for label, values in (("fixed", fixed), ("refit", refitted)):
            for name, old_name in (("R_raw", "recoverability"), ("R_raw_role", "recoverability_role")):
                old_values = saved[label+"_"+old_name]
                assert old_values.shape == values[name].shape
                difference = float(np.max(np.abs(old_values-values[name])))
                assert difference < 1e-7
                reproduction[label+"_"+name] = difference
        difference = float(np.max(np.abs(np.asarray(coefficient_draws["R_raw_role"])-saved["coefficients_recoverability_role"][:, -1])))
        assert difference < 1e-7
        reproduction["patient_coefficient_draws"] = difference
    comparisons = []
    for key, new, old, family in (("calibrated_context", "D_cal_HPC", "D_cal_HP", True),
                                  ("calibrated_vs_raw", "D_cal_HPC", "D_raw_HPC", True),
                                  ("raw_context", "D_raw_HPC", "D_raw_HP", False),
                                  ("calibrated_role_indicator", "R_cal_role", "R_cal", False),
                                  ("raw_role_indicator", "R_raw_role", "R_raw", False)):
        comparisons.append({"name": key, "new": new, "old": old, "followup_family": family,
                            **summarize_difference(point[new]-point[old], fixed[new]-fixed[old], refitted[new]-refitted[old])})
    recovery = {key: {"mean": value, "fixed_pipeline_95_interval": np.quantile(construct_fixed[key], [.025, .975]).tolist(),
                     "full_pipeline_refit_95_interval": np.quantile(construct_refit[key], [.025, .975]).tolist(),
                     "fixed_draws": 2000, "refit_draws": 200} for key, value in construct_point.items()}
    np.savez_compressed(output/f"private_{split}_draws.npz",
        **{"fixed__"+k: v for k, v in fixed.items()}, **{"refit__"+k: v for k, v in refitted.items()},
        **{"coefficient__"+k: np.asarray(v) for k, v in coefficient_draws.items()},
        **{"construct_fixed__"+k: np.asarray(v) for k, v in construct_fixed.items()},
        **{"construct_refit__"+k: np.asarray(v) for k, v in construct_refit.items()},
        temperatures=np.asarray(temperature_draws), full_temperatures=np.asarray(full_temperature_draws))
    for task, base in (("D", frame.loc[d_index].reset_index(drop=True)), ("R", roles.copy())):
        for name, prediction in point_p.items():
            if name.startswith(task):
                base[name] = prediction
        base.to_csv(output/f"private_{split}_{task}_predictions.csv", index=False)
    (output/f"private_{split}_draw_provenance.json").write_text(json.dumps({"sorted_groups": groups.tolist(),
        "multiplicity_sha256": draw_hashes, "seed": 20261009}, indent=2)+"\n")
    return {"split": split, "groups": len(groups), "nominal_pairs": population, "discordant_pairs": len(d_index),
            "role_rows": len(roles), "fixed_draws": 2000, "refit_draws": 200, "metrics": point,
            "comparisons": comparisons, "calibrated_recovery": recovery,
            "patient_coefficients": {name: {"point": point_coef[name], "full_pipeline_refit_95_interval": np.quantile(values, [.025, .975]).tolist()}
                                     for name, values in coefficient_draws.items()},
            "temperature_fits": temperature_records, "full_temperature": full_temperature,
            "raw_archive_reproduction_max_abs": reproduction, "fits": fits, "full_fits": full_fits,
            "numerics": {"fit_calls": fit_calls, "maximum_gradient_per_weight": maximum_gradient, "solver_fallbacks": fallback_records},
            "multiplicity_digest_list_sha256": hashlib.sha256(json.dumps(draw_hashes).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path, required=True)
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--model-name", choices=("Qwen3-8B", "EuroLLM-1.7B-Instruct"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    protocol_path = HERE/"README.md"
    readme_sha256 = sha(protocol_path)
    source_paths = [Path(__file__).resolve(), HERE/"diagnose_role_confidence.py",
                    Path(entity_analysis.__file__).resolve()]
    implementation_sha256 = {str(path): sha(path) for path in source_paths}
    data = load_data(args)
    # Check every constructed raw feature before any new regression is fitted.
    raw_losses = transformed(data["logp"], data["mask"], data["target"], 1.)[0]
    check = matrices(data, raw_losses)
    for new, old in (("D_cal_HP", "D_raw_HP"), ("D_cal_HPC", "D_raw_HPC"), ("R_cal", "R_raw"), ("R_cal_role", "R_raw_role")):
        assert np.max(np.abs(check[new]-check[old])) < 1e-12
    args.output.mkdir(parents=True)
    (args.output/"run_manifest.json").write_text(json.dumps({"started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name, "readme_sha256": readme_sha256, "implementation_sha256": implementation_sha256,
        "source_scores_sha256": sha(args.scores), "confidence_freeze_sha256": sha(args.confidence/"temperature_freeze.json"),
        "reference_features_sha256": sha(args.reference_analysis/"features.csv"),
        "reference_role_features_sha256": sha(args.reference_analysis/"paired_role_features.csv"),
        "seed": 20261009, "bootstrap_draws": 2000, "full_pipeline_refits": 200,
        "scientific_outcomes_computed": False}, indent=2, allow_nan=False)+"\n")
    results = {split: run_split(data, split, args.output, args.reference_analysis) for split in ("document", "lemma")}
    assert sha(protocol_path) == readme_sha256
    assert {str(path): sha(path) for path in source_paths} == implementation_sha256
    report = {"complete": True, "model": args.model_name, "source_scores_sha256": sha(args.scores),
              "confidence_freeze_sha256": sha(args.confidence/"temperature_freeze.json"),
              "readme_sha256": readme_sha256, "implementation_sha256": implementation_sha256,
              "results": results, "post_result_sensitivity": True, "original_primary_replaced": False,
              "pending_family": "two contrasts by two splits by two checkpoints; eight-slot Holm after both finish",
              "uncertainty": "fixed folds and LLM scores; whole calibration-population groups jointly refit temperature and realization heads; axes separate"}
    (args.output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"complete": True, "model": args.model_name, "splits": 2, "refits_per_split": 200}), flush=True)


if __name__ == "__main__":
    main()
