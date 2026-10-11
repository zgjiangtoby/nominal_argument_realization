#!/usr/bin/env python3
"""Describe joint support without fitting effects or selecting lexical items."""
import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from extract_pairs import ROLES, index_document


def position(mention, scope):
    if mention["exit"] < scope["enter"]:
        return "before_np"
    if mention["enter"] > scope["exit"]:
        return "after_np"
    if scope["enter"] <= mention["enter"] and mention["exit"] <= scope["exit"]:
        return "inside_np"
    return "overlapping_np"


def counts(rows):
    patterns = Counter(r["pattern"] for r in rows)
    return {"nominals": len(rows),
            "patterns": {p: patterns[p] for p in ("E/E", "E/I", "I/E", "I/I")},
            "documents": len({r["file"] for r in rows}),
            "lemmas": len({r["lemma"] for r in rows}),
            "lexicalids": len({r["lexicalid"] for r in rows if r["lexicalid"]})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xml_dir", type=Path)
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.coverage / "coverage_summary.json").read_text())
    if not summary["complete_manifest"] or summary["failed_files"]:
        raise SystemExit("Complete source coverage is required before this support readout.")
    rows = [json.loads(line) for line in (args.coverage / "nominal_pairs.jsonl").open()]
    file_meta = {r["file"]: r for r in map(json.loads, (args.coverage / "by_file.jsonl").open())}
    grouped = defaultdict(list)
    for row in rows:
        if row["both_roles_bound"]:
            grouped[row["file"]].append(row)
    contexts = []
    for name, selected in grouped.items():
        source = args.xml_dir / name
        if source.stat().st_size != file_meta[name]["bytes"]:
            raise SystemExit(f"Source size changed: {name}")
        _, loc, nodes, mentions, _, _, _ = index_document(ET.parse(source).getroot())
        by_path = {loc[n]["path"]: n for n in nodes}
        for row in selected:
            noun = by_path[row["nominal_path"]]
            assert noun.tag == "n" and noun.get("origin") == "deverbal"
            scope = loc[by_path[row["scope_path"]]]
            role_context = {}
            for role in ROLES:
                record = row["roles"][role]
                current = []
                latest_attributes = []
                for h in record["history"]:
                    for mention in mentions[h["entity_id"]]:
                        if mention["sentence"] == scope["sentence"]:
                            current.append({"entity_id": h["entity_id"], **mention,
                                            "relative_to_np": position(mention, scope)})
                    if h.get("latest"):
                        node = by_path[h["latest"]["path"]]
                        latest_attributes.append({"entity_id": h["entity_id"],
                                                  "attributes": dict(node.attrib)})
                role_context[role] = {"same_sentence_mentions": current,
                                      "same_sentence_external_mentions": [m for m in current
                                          if m["relative_to_np"] in ("before_np", "after_np")],
                                      "latest_prior_source_attributes": latest_attributes}
            implicit_roles = [role for role in ROLES if row["roles"][role]["realization"] == "I"]
            symmetric = "A%dP%d" % tuple(bool(role_context[role]["same_sentence_external_mentions"])
                                         for role in ROLES)
            contexts.append({"file": name, "nominal_path": row["nominal_path"],
                             "roles": role_context,
                             "symmetric_external_same_sentence_stratum": symmetric,
                             "implicit_has_same_sentence_mention": any(
                                 role_context[role]["same_sentence_mentions"] for role in implicit_roles),
                             "support_construction": "unknown_no_verified_lexicon_binding"})
    context_by_key = {(x["file"], x["nominal_path"]): x for x in contexts}
    primary = [r for r in rows if r["single_entity_prior_pair"] and not r["same_entity_set"]]
    same_sentence, cross_sentence = [], []
    for row in primary:
        c = context_by_key[row["file"], row["nominal_path"]]
        if c["implicit_has_same_sentence_mention"]:
            same_sentence.append(row)
        elif row["both_roles_prior_sentence"]:
            cross_sentence.append(row)
    discordant = [r for r in primary if r["pattern"] in ("E/I", "I/E")]
    by_lemma = defaultdict(Counter)
    for row in discordant:
        by_lemma[row["lemma"]][row["pattern"]] += 1
    unknown = Counter()
    for row in rows:
        for role in ROLES:
            info = row["roles"][role]
            if info["status"] == "bound":
                continue
            for a in info["assignments"]:
                unknown[role, info["status"], a["realization"],
                        a.get("binding_rule", a.get("attribute"))] += 1
    output = {"source_commit": summary["source_commit"], "effect_fitting": False,
              "all_single_entity_prior_distinct_pairs": counts(primary),
              "discordant_single_entity_prior_distinct_pairs": counts(discordant),
              "implicit_role_has_same_sentence_mention": counts(same_sentence),
              "both_roles_prior_sentence_no_implicit_same_sentence_mention": counts(cross_sentence),
              "other_primary_pairs": counts([r for r in primary
                  if r not in same_sentence and r not in cross_sentence]),
              "discordant_lemmas_with_both_realization_directions": sum(len(c) == 2 for c in by_lemma.values()),
              "discordant_lemma_frequency_distribution": dict(Counter(sum(c.values()) for c in by_lemma.values())),
              "support_construction_status": {"unknown_no_verified_lexicon_binding": len(contexts)},
              "unknown_assignments": [dict(role=k[0], status=k[1], realization=k[2],
                    binding_rule_or_attribute=k[3], assignments=v) for k, v in sorted(unknown.items())]}
    output["symmetric_external_same_sentence_strata"] = {
        label: counts([r for r in primary if context_by_key[r["file"], r["nominal_path"]]
                       ["symmetric_external_same_sentence_stratum"] == label])
        for label in ("A0P0", "A1P0", "A0P1", "A1P1")}
    output["primary_latest_prior_attributes"] = {
        role: {attribute: dict(Counter(
            context_by_key[r["file"], r["nominal_path"]]["roles"][role]
                ["latest_prior_source_attributes"][0]["attributes"].get(attribute, "MISSING")
            for r in primary)) for attribute in ("func", "entityref", "ne", "coreftype")}
        for role in ROLES}
    assert len(primary) == len(same_sentence) + len(cross_sentence) + output["other_primary_pairs"]["nominals"]
    assert len(contexts) == summary["both_roles_bound"]["nominals"]
    assert sum(s["nominals"] for s in output["symmetric_external_same_sentence_strata"].values()) == len(primary)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "pair_context.jsonl").open("w") as out:
        for row in contexts:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "support_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k not in
                      ("unknown_assignments", "primary_latest_prior_attributes")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    scope = {"enter": 10, "exit": 20}
    assert [position({"enter": a, "exit": b}, scope) for a, b in
            ((1, 9), (21, 25), (12, 18), (5, 25))] == [
                "before_np", "after_np", "inside_np", "overlapping_np"]
    main()
