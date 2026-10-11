"""Build form-hidden agent and patient recovery probes from linked nominal pairs."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CODE = Path(__file__).resolve().parent
from extract_pairs import index_document


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def prompt_for(record,role,condition,order):
    indices = list(range(len(record["candidates"])))
    if order:
        indices.reverse()
    labels = list("ABCDEFGHIJKLM")[:len(indices)+1]
    options = []
    for letter,index in zip(labels,indices):
        candidate = record["candidates"][index]
        options.append(f'{letter}. "{candidate["alias"]}" (last mention: word {candidate["last_word"]} before the nominal phrase)')
    options.append(f"{labels[-1]}. Another entity not listed")
    definition = ("Agent: the participant carrying out or initiating the event." if role=="agt"
                  else "Patient: the participant affected by or undergoing the event.")
    prompt = ("A Spanish deverbal nominal expression is forthcoming. Select the entity most likely to fill "
              "the specified semantic role in that nominal event. Each option denotes a distinct previously mentioned "
              "entity. Current argument expressions have not been shown.\n\nROLE: "+definition+
              "\nNOMINAL LEMMA: "+record["lemma"]+"\n")
    if condition == "context":
        prompt += "\nPRECEDING DISCOURSE:\n"+record["prefix"]+"\n"
    prompt += "\nENTITIES:\n"+"\n".join(options)+"\n\nReturn only the letter."
    return prompt,labels,[record["candidates"][i]["entity_id"] for i in indices]+["OTHER"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=CODE/"data/ancora/Corpus/Ancora")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists(),args.output
    coverage = read_jsonl(CODE/"work/coverage/nominal_pairs.jsonl")
    selected = [r for r in coverage if r["single_entity_prior_pair"] and not r["same_entity_set"]]
    assert len(selected) == 2253 and len({r["file"] for r in selected}) == 748
    units = pd.read_csv(CODE/"work/baseline/analysis_units.csv").set_index(["file","nominal_path"])
    old_predictions = pd.read_csv(CODE/"work/baseline/heldout_predictions.csv")
    folds = {}
    rng = np.random.default_rng(20261009)
    for split,key in [("document","file"),("lemma","lemma")]:
        old = old_predictions[old_predictions.split.eq(split)]
        assert not old.empty and old.groupby(key).fold.nunique().eq(1).all()
        mapping = old.groupby(key).fold.first().to_dict()
        for value in sorted({r[key] for r in selected}):
            if value not in mapping:
                mapping[value] = int(rng.integers(0,5))
        folds[split] = mapping
    by_file = defaultdict(list)
    for row in selected:
        by_file[row["file"]].append(row)
    records,exclusions,patterns = [],Counter(),Counter()
    for file,group in sorted(by_file.items()):
        root = ET.parse(args.source_root/file).getroot()
        _,locations,nodes,mentions,_,_,_ = index_document(root)
        by_path = {loc["path"]:node for node,loc in locations.items()}
        words = [node.get("wd") for node in nodes if node.get("wd") is not None]
        for row in group:
            scope = locations[by_path[row["scope_path"]]]
            cutoff = scope["terminal_start"]
            start = max(0,cutoff-512)
            candidates = []
            for entity,chain in mentions.items():
                prior = [m for m in chain if m["exit"] < scope["enter"]]
                retained = [m for m in prior if m["terminal_start"] >= start and m["terminal_end"] <= cutoff]
                if not retained:
                    continue
                overt = [m for m in retained if m["terminal_end"] > m["terminal_start"]]
                if not overt:
                    exclusions["candidate_without_surface_alias"] += 1
                    continue
                lexical = [m for m in overt if any(n.tag in ("n","np") for n in by_path[m["path"]].iter())]
                name = max(lexical or overt,key=lambda m:(m["exit"],m["enter"]))
                latest = max(retained,key=lambda m:(m["exit"],m["enter"]))
                alias_words = words[name["terminal_start"]:name["terminal_end"]]
                assert alias_words
                candidates.append({"entity_id":entity,"alias":" ".join(alias_words[:24])+(" ..." if len(alias_words)>24 else ""),
                                   "alias_lexical":bool(lexical),"alias_path":name["path"],
                                   "alias_start":name["terminal_start"],"alias_end":name["terminal_end"],
                                   "last_word":latest["terminal_end"]-cutoff,"latest_exit":latest["exit"]})
            candidates.sort(key=lambda c:(-c["latest_exit"],c["entity_id"]))
            candidates = candidates[:12]
            gold = {role:row["roles"][role]["entity_ids"][0] for role in ("agt","pat")}
            available = {c["entity_id"] for c in candidates}
            if not all(entity in available for entity in gold.values()):
                exclusions["both_gold_not_in_pool:"+row["pattern"]] += 1
                continue
            assert len(set(gold.values())) == 2 and all(c["alias_end"] <= cutoff for c in candidates)
            unit = units.loc[(file,row["nominal_path"])]
            record = {"item_id":f"nominal:{file}:{row['nominal_path']}","file":file,
                      "nominal_path":row["nominal_path"],"lemma":row["lemma"],"pattern":row["pattern"],
                      "gold_entities":gold,"candidates":candidates,"prefix":" ".join(words[start:cutoff]),
                      "cutoff_terminal":cutoff,"cutoff_enter":scope["enter"],"retained_start_terminal":start,
                      "document_fold":int(folds["document"][file]),"lemma_fold":int(folds["lemma"][row["lemma"]]),
                      **{key:float(unit[key]) for key in ("delta_log_terminal_distance","delta_log_prior_count",
                                                        "delta_prior_subject","delta_prior_function_missing")},
                      "offline_context":unit["stratum"]}
            records.append(record)
            patterns[row["pattern"]] += 1
    technical_pool = [i for i,r in enumerate(records) if r["document_fold"] == 0]
    assert len(technical_pool) >= 24
    technical = set(rng.choice(technical_pool,size=24,replace=False).tolist())
    args.output.mkdir(parents=True)
    with (args.output/"items.jsonl").open("x") as items,(args.output/"inputs.jsonl").open("x") as full,(args.output/"pilot.jsonl").open("x") as pilot:
        for index,record in enumerate(records):
            record["technical_pilot"] = index in technical
            items.write(json.dumps(record,ensure_ascii=False)+"\n")
            for role in ("agt","pat"):
                for condition in ("context","prior"):
                    for order in (0,1):
                        prompt,labels,entities = prompt_for(record,role,condition,order)
                        input_row = {"id":f"{record['item_id']}:{role}:{condition}:o{order}","prompt":prompt,"labels":labels,
                                     "item_id":record["item_id"],"role":role,"condition":condition,"order":order,"entity_codes":entities}
                        line = json.dumps(input_row,ensure_ascii=False)+"\n"
                        full.write(line)
                        if index in technical:
                            pilot.write(line)
    report = {"original_pairs":2253,"original_documents":748,"included_pairs":len(records),
              "included_documents":len({r["file"] for r in records}),"patterns":dict(patterns),
              "exclusions":dict(exclusions),"technical_pilot_items":len(technical),
              "candidate_counts":dict(Counter(len(r["candidates"]) for r in records)),
              "word_truncated_pairs":sum(r["retained_start_terminal"]>0 for r in records),
              "old_discordant_folds_preserved":True,"source_root":str(args.source_root)}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
