"""Evaluate argument recovery and realization on document and lemma folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from entity_analysis import read_jsonl,read_probes,gold_features,grouped_evaluation,contrast,holm,cluster_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--refits",type=int,default=200)
    args = parser.parse_args()
    assert not args.output.exists()
    probes = read_probes(HERE/"work/probes",args.scores)
    original = {(r["file"],r["nominal_path"]):r for r in read_jsonl(HERE/"work/coverage/nominal_pairs.jsonl")
                if r.get("single_entity_prior_pair") and not r.get("same_entity_set")}
    rows,recovery,role_rows = [],[],[]
    for record in read_jsonl(HERE/"work/probes/items.jsonl"):
        row = {key:value for key,value in record.items() if key not in ("prefix","candidates","gold_entities")}
        row["y"] = int(record["pattern"] == "I/E")
        source = original[(record["file"],record["nominal_path"])]
        assert source["pattern"]==record["pattern"]
        aliases = [c["alias"].strip().casefold() for c in record["candidates"]]
        row["alias_collision"] = int(len(set(aliases))!=len(aliases))
        for role in ("agt","pat"):
            gold = record["gold_entities"][role]
            alternative = record["gold_entities"]["pat" if role=="agt" else "agt"]
            for condition in ("context","prior"):
                probe = probes[(record["item_id"],role,condition)]
                values = gold_features(probe,gold)
                row.update({role+"_"+condition+"_"+key:value for key,value in values.items()})
                recovery.append({"item_id":record["item_id"],"file":record["file"],"lemma":record["lemma"],
                                 "pattern":record["pattern"],"role":role,"condition":condition,
                                 "candidates":len(record["candidates"]),"uniform_nll":float(np.log(len(record["candidates"]))),
                                 "uniform_accuracy":1/len(record["candidates"]),
                                 "recency_correct":int(record["candidates"][0]["entity_id"]==gold),
                                 "gold_over_swapped_probability":probe["probabilities"][gold]-probe["probabilities"][alternative],**values})
            history = source["roles"][role]["history"][0]
            assert source["roles"][role]["entity_ids"]==[gold]
            assert history["latest"]["exit"]<record["cutoff_enter"]
            assert history["terminal_distance"]==record["cutoff_terminal"]-history["latest"]["terminal_end"]
            role_rows.append({"item_id":record["item_id"],"role":role,"file":record["file"],"lemma":record["lemma"],
                "document_fold":record["document_fold"],"lemma_fold":record["lemma_fold"],
                "y":int(source["roles"][role]["realization"]=="E"),"patient":int(role=="pat"),
                "log_terminal_distance":float(np.log1p(history["terminal_distance"])),
                "log_prior_count":float(np.log1p(history["prior_count"])),
                "prior_subject":int(history["latest"]["func"]=="suj"),
                "prior_function_missing":int(history["latest"]["func"] is None),
                "prior_surprisal":row[role+"_prior_surprisal"],
                "context_surprisal_increment":row[role+"_context_surprisal"]-row[role+"_prior_surprisal"]})
        row["prior_role_surprisal_difference"] = row["agt_prior_surprisal"]-row["pat_prior_surprisal"]
        row["context_role_surprisal_difference"] = row["agt_context_surprisal"]-row["pat_context_surprisal"]
        row["context_role_surprisal_increment"] = row["context_role_surprisal_difference"]-row["prior_role_surprisal_difference"]
        for context in ("A1P0","A0P1","A1P1"):
            row[context] = int(row["offline_context"]==context)
        rows.append(row)
    frame = pd.DataFrame(rows)
    roles = pd.DataFrame(role_rows)
    for feature,delta in [("log_terminal_distance","delta_log_terminal_distance"),("log_prior_count","delta_log_prior_count"),
                          ("prior_subject","delta_prior_subject"),("prior_function_missing","delta_prior_function_missing")]:
        wide = roles.pivot(index="item_id",columns="role",values=feature)
        saved = frame.set_index("item_id")[delta].reindex(wide.index)
        assert np.allclose(wide.agt-wide.pat,saved)
    baseline = ["delta_log_terminal_distance","delta_log_prior_count","delta_prior_subject","delta_prior_function_missing"]
    prior = baseline+["prior_role_surprisal_difference"]
    contextual = prior+["context_role_surprisal_increment"]
    columns = {"H":baseline,"HP":prior,"HPC":contextual,"HPC_offline":contextual+["A1P0","A0P1","A1P1"]}
    args.output.mkdir(parents=True)
    frame.to_csv(args.output/"features.csv",index=False)
    recovery = pd.DataFrame(recovery)
    recovery["nll_minus_uniform"] = recovery.surprisal-recovery.uniform_nll
    recovery["accuracy_minus_uniform"] = recovery.correct-recovery.uniform_accuracy
    recovery["accuracy_minus_recency"] = recovery.correct-recovery.recency_correct
    recovery.to_csv(args.output/"recovery.csv",index=False)
    roles.to_csv(args.output/"paired_role_features.csv",index=False)
    constructs = {}
    for (role,condition),part in recovery.groupby(["role","condition"]):
        constructs[role+":"+condition] = cluster_summary(part,["surprisal","correct","nll_minus_uniform","accuracy_minus_uniform",
            "accuracy_minus_recency","gold_over_swapped_probability","other_mass","answer_mass","order_total_variation",
            "order_top_disagreement"],"file",False)
    role_increment = []
    for role in ("agt","pat"):
        part = frame[["item_id","file"]].copy()
        part["nll_context_minus_prior"] = frame[role+"_context_surprisal"]-frame[role+"_prior_surprisal"]
        part["accuracy_context_minus_prior"] = frame[role+"_context_correct"]-frame[role+"_prior_correct"]
        role_increment.append({"role":role,"contrasts":cluster_summary(part,["nll_context_minus_prior","accuracy_context_minus_prior"],"file",False)})
    rates = frame[["file","pattern","alias_collision"]].copy()
    rates["agent_direct"] = rates.pattern.str[0].eq("E").astype(int)
    rates["patient_direct"] = rates.pattern.str[2].eq("E").astype(int)
    rates["patient_minus_agent_direct"] = rates.patient_direct-rates.agent_direct
    direct_rates = cluster_summary(rates,["agent_direct","patient_direct","patient_minus_agent_direct","alias_collision"],"file",False)
    discordant = frame[frame.pattern.isin(["I/E","E/I"])].reset_index(drop=True)
    summaries,primary,predictions = {},[],[]
    for split,key in [("document","file"),("lemma","lemma")]:
        result = grouped_evaluation(discordant,columns,key,split+"_fold",False,args.refits,20261009,True)
        summary = {name:result[name] for name in ("groups","rows","weighting","metrics","fits","full_fits","numerics")}
        summary["comparisons"] = [contrast(result,*pair) for pair in [("HPC","HP"),("HP","H"),("HPC_offline","HPC")]]
        summary["coefficient_intervals"] = {name:{"features":["intercept",*columns[name]],
            "paired_full_refit_95_intervals":np.quantile(values,[.025,.975],axis=0).T.tolist()} for name,values in result["coefficient_draws"].items() if len(values)}
        summaries[split] = summary
        primary.append(summary["comparisons"][0])
        prediction = discordant[["item_id","file","lemma","pattern","y",split+"_fold"]].copy()
        prediction["split"] = split
        for name,p in result["predictions"].items():
            prediction[name] = p
        predictions.append(prediction)
        np.savez_compressed(args.output/f"{split}_bootstrap.npz",**{"fixed_"+k:v for k,v in result["fixed_draws"].items()},
                            **{"refit_"+k:v for k,v in result["refit_draws"].items()},
                            **{"coefficients_"+k:v for k,v in result["coefficient_draws"].items()})
        print(json.dumps({"split":split,"main_comparison":primary[-1]}),flush=True)
        role_baseline = ["log_terminal_distance","log_prior_count","prior_subject","prior_function_missing",
                         "prior_surprisal","context_surprisal_increment"]
        role_columns = {"recoverability":role_baseline,"recoverability_role":role_baseline+["patient"]}
        role_result = grouped_evaluation(roles,role_columns,key,split+"_fold",False,args.refits,20261009,True)
        summary["all_patterns_paired_role_model"] = {k:role_result[k] for k in ("rows","groups","metrics","fits","full_fits","numerics")}
        summary["all_patterns_paired_role_model"]["comparison"] = contrast(role_result,"recoverability_role","recoverability")
        summary["all_patterns_paired_role_model"]["patient_coefficient_95_interval"] = np.quantile(
            role_result["coefficient_draws"]["recoverability_role"][:,-1],[.025,.975]).tolist() if args.refits else None
        np.savez_compressed(args.output/f"{split}_all_patterns_bootstrap.npz",**{"fixed_"+k:v for k,v in role_result["fixed_draws"].items()},
                            **{"refit_"+k:v for k,v in role_result["refit_draws"].items()},
                            **{"coefficients_"+k:v for k,v in role_result["coefficient_draws"].items()})
        role_prediction = roles.copy()
        for name,p in role_result["predictions"].items():
            role_prediction[name] = p
        role_prediction.to_csv(args.output/f"{split}_all_patterns_heldout.csv",index=False)
    for row,adjusted in zip(primary,holm([r["normal_approximation_p"] for r in primary])):
        row["holm_p_two_splits"] = adjusted
    pd.concat(predictions,ignore_index=True).to_csv(args.output/"heldout_predictions.csv",index=False)
    (args.output/"summary.json").write_text(json.dumps({"source_scores":str(args.scores),"results":summaries,
        "interpretation":"paired role association in the retained double-prior opportunity; no causal identification", "C":1,
        "construct_checks":constructs,"context_increment_checks":role_increment,"all_pattern_direct_rates":direct_rates,
        "all_pattern_counts":frame.pattern.value_counts().to_dict(),
        "primary":"HPC-HP for document and lemma holdout; Qwen primary, Euro replication", "completed":True},indent=2)+"\n")


if __name__ == "__main__":
    main()
