#!/usr/bin/env python3
"""Extract linked nominal arguments and their prior mentions from AnCora XML.

Unlinked roles and ambiguous nominal scopes retain explicit status fields."""
import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


IARG = re.compile(r"iarg(?:[0-4]|M)$")
ENTITY = re.compile(r"(?:entity|singleton)\d+$")
ROLES = ("agt", "pat")


def entity_parts(raw):
    parts = raw.split("+")
    if not parts or any(not ENTITY.fullmatch(x) for x in parts):
        return None
    return sorted(set(parts))


def index_document(root):
    parent, loc, nodes, sentences = {}, {}, [], {}
    clock, terminal, sentence = 0, 0, -1

    def walk(node, address):
        nonlocal clock, terminal, sentence
        if node.tag == "sentence":
            sentence += 1
        start = clock
        clock += 1
        begin = terminal
        if "wd" in node.attrib:
            terminal += 1
        nodes.append(node)
        sentences[node] = sentence
        for i, child in enumerate(node):
            parent[child] = node
            walk(child, f"{address}/{i}")
        loc[node] = dict(path=address, enter=start, exit=clock,
                         terminal_start=begin, terminal_end=terminal,
                         sentence=sentences[node])
        clock += 1

    walk(root, "0")
    mentions, seen = defaultdict(list), set()
    invalid_mentions = []
    for node in nodes:
        raw = node.get("entity")
        if raw is None:
            continue
        ids = entity_parts(raw)
        if ids is None:
            invalid_mentions.append({"path": loc[node]["path"], "raw": raw})
            continue
        pos = loc[node]
        for eid in ids:
            # An aggregate entity mention does not create separate atomic gold
            # mentions; only the raw singleton ID indexes the atomic history.
            if len(ids) != 1:
                continue
            span = (pos["terminal_start"], pos["terminal_end"])
            key = (eid, span) if span[0] != span[1] else (eid, pos["path"])
            if key in seen:
                continue
            seen.add(key)
            mentions[eid].append({**pos, "tag": node.tag,
                                  "func": node.get("func"),
                                  "coreftype": node.get("coreftype"),
                                  "entityref": node.get("entityref"),
                                  "elliptic": node.get("elliptic")})
    for history in mentions.values():
        history.sort(key=lambda x: (x["exit"], x["enter"]))
    return parent, loc, nodes, mentions, invalid_mentions, sentence + 1, terminal


def nominal_scope(noun, parent):
    group = parent.get(noun)
    if group is None or group.tag != "grup.nom":
        return None, [], "noun_not_direct_group_head"
    heads = [x for x in group if x.tag == "n"]
    if len(heads) != 1 or group.get("coord") == "yes":
        return None, [], "multiple_or_coordinated_group_heads"
    branches = [x for x in group if x is not noun]
    np = parent.get(group)
    if np is not None and np.tag == "sn":
        groups = [x for x in np if x.tag == "grup.nom"]
        if len(groups) != 1 or np.get("coord") == "yes":
            return None, [], "multiple_or_coordinated_np_groups"
        branches += [x for x in np if x is not group]
        return np, branches, "unique_np_head"
    return group, branches, "unique_group_head"


def explicit_entity(node):
    if node.get("entity") is not None:
        return node.get("entity"), "argument_node_entity"
    if node.tag == "sp":
        children = [x for x in node if x.tag in ("sn", "S")]
        if len(children) == 1:
            return children[0].get("entity"), "unique_pp_complement_entity"
        return None, "pp_complement_missing_or_multiple"
    if node.tag == "spec":
        children = [x for x in node if x.tag == "d"
                    and x.get("postype") == "possessive"]
        if len(children) == 1:
            return children[0].get("entity"), "possessive_determiner_entity"
        return None, "no_unique_possessive_entity"
    if node.tag in ("s.a", "grup.a", "relatiu"):
        children = [x for x in node if x.tag in ("grup.a", "a", "p", "r")]
        if len(children) == 1:
            child = children[0]
            if child.tag == "grup.a":
                return explicit_entity(child)
            return child.get("entity"), "unique_adjectival_or_relative_head_entity"
    return None, "no_argument_level_entity"


def summarize_role(assignments, mentions, cutoff, noun_loc):
    if not assignments:
        return {"status": "not_annotated", "assignments": []}
    if len(assignments) != 1:
        return {"status": "multiple_or_conflicting_assignments", "assignments": assignments}
    assignment = assignments[0]
    raw = assignment.get("raw_entity")
    ids = entity_parts(raw) if raw is not None else None
    if ids is None:
        return {"status": "invalid_or_missing_entity_reference", "assignments": assignments}
    missing = [eid for eid in ids if eid not in mentions]
    if missing:
        return {"status": "entity_without_atomic_mention", "assignments": assignments,
                "entity_ids": ids, "missing_entity_ids": missing}
    histories = []
    for eid in ids:
        prior = [m for m in mentions[eid] if m["exit"] < cutoff["enter"]]
        current_prefix = [m for m in mentions[eid]
                          if cutoff["enter"] <= m["enter"]
                          and m["exit"] < noun_loc["enter"]]
        h = {"entity_id": eid, "prior_count": len(prior),
             "current_np_mentions_before_head": len(current_prefix),
             "prior_sentence_count": sum(m["sentence"] < cutoff["sentence"] for m in prior)}
        if prior:
            last = prior[-1]
            h.update(latest=last,
                     sentence_distance=cutoff["sentence"] - last["sentence"],
                     terminal_distance=cutoff["terminal_start"] - last["terminal_end"])
        else:
            h["first_mention_enter"] = min(m["enter"] for m in mentions[eid])
        histories.append(h)
    return {"status": "bound", "realization": assignment["realization"],
            "entity_ids": ids, "composite": len(ids) > 1,
            "all_components_prior": all(x["prior_count"] > 0 for x in histories),
            "all_components_prior_sentence": all(x["prior_sentence_count"] > 0 for x in histories),
            "all_components_latest_prior_sentence": all(x.get("sentence_distance", 0) > 0
                                                          for x in histories),
            "history": histories, "assignments": assignments}


def extract_file(path, file_id):
    root = ET.parse(path).getroot()
    parent, loc, nodes, mentions, invalid_mentions, n_sent, n_term = index_document(root)
    rows = []
    for node in nodes:
        if node.tag != "n" or node.get("origin") != "deverbal":
            continue
        scope, branches, scope_status = nominal_scope(node, parent)
        assignments = {role: [] for role in ROLES}
        other_implicit = []
        for key, value in node.attrib.items():
            if not IARG.fullmatch(key):
                continue
            role, sep, raw = value.partition(":")
            source = {"realization": "I", "attribute": key, "raw_value": value,
                      "raw_entity": raw if sep else None, "node_path": loc[node]["path"]}
            if role in ROLES:
                assignments[role].append(source)
            else:
                other_implicit.append({"role": role, **source})
        for branch in branches:
            role = branch.get("tem")
            if role not in ROLES or branch.get("arg") is None:
                continue
            raw, rule = explicit_entity(branch)
            assignments[role].append({"realization": "E", "argument": branch.get("arg"),
                                      "raw_entity": raw, "binding_rule": rule,
                                      "node_path": loc[branch]["path"], "tag": branch.tag})
        row = {"file": file_id, "nominal_path": loc[node]["path"],
               "lemma": node.get("lem"), "lexicalid": node.get("lexicalid"),
               "sense": node.get("sense"), "nominal_location": loc[node],
               "scope_status": scope_status, "scope_path": loc[scope]["path"] if scope is not None else None,
               "other_implicit_roles": other_implicit, "roles": {},
               "both_roles_bound": False, "both_roles_prior": False,
               "both_roles_prior_sentence": False, "single_entity_prior_pair": False}
        row["both_roles_latest_prior_sentence"] = False
        if scope is None:
            row["roles"] = {role: {"status": "nominal_scope_unresolved",
                                    "assignments": assignments[role]} for role in ROLES}
        else:
            row["roles"] = {role: summarize_role(assignments[role], mentions, loc[scope], loc[node])
                            for role in ROLES}
            a, p = [row["roles"][r] for r in ROLES]
            row["both_roles_bound"] = a["status"] == p["status"] == "bound"
            if row["both_roles_bound"]:
                row["pattern"] = a["realization"] + "/" + p["realization"]
                row["both_roles_prior"] = a["all_components_prior"] and p["all_components_prior"]
                row["both_roles_prior_sentence"] = (a["all_components_prior_sentence"]
                                                       and p["all_components_prior_sentence"])
                row["both_roles_latest_prior_sentence"] = (a["all_components_latest_prior_sentence"]
                                                              and p["all_components_latest_prior_sentence"])
                row["single_entity_prior_pair"] = (row["both_roles_prior"]
                                                      and not a["composite"] and not p["composite"])
                row["same_entity_set"] = a["entity_ids"] == p["entity_ids"]
        rows.append(row)
    return rows, {"sentences": n_sent, "terminals": n_term,
                  "atomic_entity_ids": len(mentions),
                  "entity_mentions": sum(len(v) for v in mentions.values()),
                  "invalid_entity_mentions": invalid_mentions,
                  "noun_lexicalids_without_deverbal": sum(n.tag == "n" and n.get("lexicalid", "").startswith("noun.")
                                                         and n.get("origin") != "deverbal" for n in nodes)}


def summarize_rows(rows):
    result = {"nominalizations": len(rows),
              "scope_status": dict(Counter(r["scope_status"] for r in rows)),
              "role_status": {role: dict(Counter(r["roles"][role]["status"] for r in rows)) for role in ROLES},
              "missing_lexicalid": sum(not r["lexicalid"] for r in rows),
              "missing_sense": sum(not r["sense"] for r in rows)}
    result["observed_role_sources"] = {}
    for role in ROLES:
        source_counts = Counter()
        for row in rows:
            assignments = row["roles"][role]["assignments"]
            label = "+".join(sorted(a["realization"] for a in assignments)) if assignments else "none"
            source_counts[label] += 1
        result["observed_role_sources"][role] = dict(source_counts)
    for criterion in ("both_roles_bound", "both_roles_prior", "both_roles_prior_sentence",
                      "both_roles_latest_prior_sentence", "single_entity_prior_pair"):
        subset = [r for r in rows if r[criterion]]
        patterns = Counter(r["pattern"] for r in subset)
        result[criterion] = {"nominals": len(subset),
                             "patterns": {k: patterns[k] for k in ("E/E", "E/I", "I/E", "I/I")},
                             "documents": len({r["file"] for r in subset}),
                             "lemmas": len({r["lemma"] for r in subset}),
                             "lexicalids": len({r["lexicalid"] for r in subset if r["lexicalid"]}),
                             "same_entity_pairs": sum(r.get("same_entity_set", False) for r in subset)}
    result["current_np_mentions_before_head"] = {
        role: sum(h["current_np_mentions_before_head"] for r in rows
                  for h in r["roles"][role].get("history", [])) for role in ROLES}
    return result


SOURCE_REPOSITORY = "https://github.com/PabloVelascoCrespo/MCR_to_Prolog"
SOURCE_COMMIT = "1addb59e3d9f18c794555d22a2efba263e6588a3"


def source_files(xml_dir):
    """Read the fixed XML inventory from the source repository's Git tree."""
    repository = xml_dir.resolve().parents[1]
    if xml_dir.resolve() != repository / "Corpus/Ancora":
        raise ValueError("Expected the source checkout's Corpus/Ancora directory")
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError("The source checkout does not match the required commit")
    tree = subprocess.check_output(
        ["git", "-C", str(repository), "ls-tree", "-r", "-l", SOURCE_COMMIT, "--", "Corpus/Ancora"], text=True)
    files = []
    for line in tree.splitlines():
        metadata, name = line.split("\t", 1)
        mode, kind, blob, size = metadata.split()
        if kind == "blob" and name.endswith(".xml"):
            files.append({"path": str(Path(name).relative_to("Corpus/Ancora")), "bytes": int(size)})
    files.sort(key=lambda row: row["path"])
    if len(files) != 1635 or sum(row["bytes"] for row in files) != 107488704:
        raise ValueError("Unexpected XML inventory in the source tree")
    return {"repository": SOURCE_REPOSITORY, "commit": SOURCE_COMMIT, "files": files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xml_dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = source_files(args.xml_dir)
    files = [(x, args.xml_dir / x["path"]) for x in manifest["files"]]
    available = [(m, path) for m, path in files if path.is_file()]
    if not available:
        raise SystemExit("No manifest XML file is present; no zero-coverage result was generated.")
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows, by_file, failed = [], [], []
    for meta, path in available:
        size = path.stat().st_size
        if size != meta["bytes"]:
            failed.append({"file": meta["path"], "reason": "source_size_mismatch", "expected_bytes": meta["bytes"], "local_bytes": size})
            continue
        try:
            rows, doc = extract_file(path, meta["path"])
        except ET.ParseError as exc:
            failed.append({"file": meta["path"], "reason": "xml_parse_error", "detail": str(exc)})
            continue
        all_rows.extend(rows)
        by_file.append({"file": meta["path"], "bytes": size, **doc, **summarize_rows(rows)})
    for name, records in (("nominal_pairs.jsonl", all_rows), ("by_file.jsonl", by_file)):
        with (args.output / name).open("w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"source_repository": manifest["repository"], "source_commit": manifest["commit"],
               "manifest_files": len(files), "present_files": len(available), "parsed_files": len(by_file),
               "parsed_bytes": sum(r["bytes"] for r in by_file),
               "complete_manifest": len(by_file) == len(files),
               "missing_files": [m["path"] for m, path in files if not path.is_file()], "failed_files": failed,
               "sample_scope": "fixed-source available files only; no effect fitting", **summarize_rows(all_rows)}
    (args.output / "coverage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("missing_files", "failed_files")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
