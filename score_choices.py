"""Score single-token answer candidates with a local causal language model."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODULE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_tokenizer(model_path):
    config = json.loads((Path(model_path)/"config.json").read_text())
    # Transformers 4.57.2 has a Qwen fast-tokenizer loading bug; use its slow tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,
                                              use_fast=config.get("model_type")!="qwen3")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None
    return tokenizer


def load_local(model_path, dtype="bfloat16"):
    tokenizer = load_tokenizer(model_path)
    torch.set_float32_matmul_precision("highest")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=getattr(torch,dtype),
        attn_implementation="sdpa").to("cuda").eval()
    return tokenizer, model


def encode_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)


def contextual_answer_ids(tokenizer,rows,sequences):
    """Derive continuations in their actual template context, including SentencePiece boundaries."""
    rendered = tokenizer.apply_chat_template([{"role":"user","content":rows[0]["prompt"]}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix = tokenizer.encode(rendered,add_special_tokens=False)
    assert prefix==sequences[0]
    special = set(tokenizer.all_special_ids)
    barriers = [i for i,token in enumerate(prefix) if token in special]
    assert barriers,"Native chat template must expose a special-token boundary"
    terminal = prefix[barriers[-1]:]
    assert len(terminal)<64
    # The final special token blocks token merges from preceding user content.
    assert all(sequence[-len(terminal):]==terminal for sequence in sequences)
    encoded = {}
    for label in {label for row in rows for label in row["labels"]}:
        joined = tokenizer.encode(rendered+label,add_special_tokens=False)
        assert joined[:len(prefix)]==prefix,(label,"answer resegments the template")
        suffix = joined[len(prefix):]
        assert len(suffix)==1,(label,suffix)
        encoded[label] = suffix[0]
    assert len(set(encoded.values()))==len(encoded)
    return encoded,terminal


def next_logits(model, tokenizer, sequences):
    batch = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to("cuda")
    positions = (batch.attention_mask.cumsum(-1)-1).clamp_min(0)
    hidden = model.model(**batch, position_ids=positions, use_cache=False).last_hidden_state[:, -1]
    return model.lm_head(hidden).float()


def forward_probe_indices(sequences):
    """Choose at most two sequences using only length, including padding if possible."""
    indices = [0]
    for i, sequence in enumerate(sequences[1:], 1):
        if len(sequence) != len(sequences[0]):
            indices.append(i)
            break
    return indices


def check_full_forward(model, tokenizer, sequences):
    indices = forward_probe_indices(sequences)
    probes = [sequences[i] for i in indices]
    batch = tokenizer.pad({"input_ids": probes}, padding=True, return_tensors="pt").to("cuda")
    positions = (batch.attention_mask.cumsum(-1)-1).clamp_min(0)
    standard = model(**batch, position_ids=positions, use_cache=False,
                     logits_to_keep=1).logits[:, -1].float()
    optimized = next_logits(model, tokenizer, probes)
    difference = float((standard-optimized).abs().max())
    assert torch.isfinite(standard).all() and difference < 1e-4, difference
    return {"source_row_indices": indices, "sequence_lengths": list(map(len, probes)),
            "full_vocabulary_max_abs_logit_difference": difference,
            "reference": "AutoModelForCausalLM forward with logits_to_keep=1",
            "tolerance": 1e-4, "passed": True}


def metadata(args,tokenizer=None):
    return {"model_path": str(Path(args.model).resolve()), "input": str(Path(args.input).resolve()),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(), "dtype": args.dtype,
            "float32_matmul_precision": "highest", "position_ids": "unpadded cumulative attention mask",
            "attention": "sdpa", "thinking": False, "batch_size": args.batch_size,
            "maximum_prompt_tokens": args.max_tokens, "gold_used": False,
            "runtime_source_sha256": {"score_choices.py": MODULE_SOURCE_SHA256},
            "tokenizer_class":type(tokenizer).__name__ if tokenizer is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bfloat16","float32"), default="bfloat16")
    parser.add_argument("--check-batch-equivalence", action="store_true")
    parser.add_argument("--check-full-forward", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available() and args.batch_size > 0
    assert not args.output.exists(), args.output
    input_bytes = args.input.read_bytes()
    rows = [json.loads(line) for line in input_bytes.splitlines() if line.strip()]
    assert rows and len({row["id"] for row in rows}) == len(rows)
    tokenizer, model = load_local(args.model,args.dtype)
    sequences = [encode_prompt(tokenizer, row["prompt"]) for row in rows]
    encoded,terminal = contextual_answer_ids(tokenizer,rows,sequences)
    assert max(map(len, sequences)) <= args.max_tokens, "Input exceeds the token limit"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata(args,tokenizer)
    meta["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
    meta["answer_token_ids"] = encoded
    meta["answer_tokenization"] = "canonical joint template+answer tokenization; common terminal special-token segment verified"
    meta["terminal_template_token_ids"] = terminal
    meta["answer_probability_mass_upper_roundoff_tolerance"] = 1e-5
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), args.output.open("x") as output:
        if args.check_full_forward:
            meta["full_forward_check"] = check_full_forward(model, tokenizer, sequences)
        if args.check_batch_equivalence:
            probes = sequences[:min(args.batch_size, len(sequences))]
            batched = next_logits(model, tokenizer, probes)
            differences = []
            for i, sequence in enumerate(probes):
                solo = next_logits(model, tokenizer, [sequence])[0]
                labels = [encoded[label] for label in rows[i]["labels"]]
                p = batched[i, labels].softmax(-1)
                q = solo[labels].softmax(-1)
                differences.append(float((p-q).abs().max()))
            meta["batch_probability_max_abs_difference"] = max(differences)
            assert max(differences) < (1e-4 if args.dtype == "float32" else 0.02), differences
        for offset in range(0, len(rows), args.batch_size):
            block = rows[offset:offset+args.batch_size]
            logits = next_logits(model, tokenizer, sequences[offset:offset+args.batch_size])
            full_log_probs = logits.log_softmax(-1)
            for j, row in enumerate(block):
                ids = [encoded[label] for label in row["labels"]]
                scores = full_log_probs[j, ids]
                probabilities = scores.softmax(-1)
                assert torch.isfinite(scores).all() and torch.isfinite(probabilities).all()
                answer_mass = float(scores.logsumexp(-1).exp())
                # Preserve the raw FP32 diagnostic; exp/logsumexp can round just above one.
                assert 0 <= answer_mass <= 1+1e-5, answer_mass
                record = {"id": row["id"], "labels": row["labels"],
                          "answer_log_probabilities": scores.cpu().tolist(),
                          "choice_probabilities": probabilities.cpu().tolist(),
                          "answer_probability_mass": answer_mass,
                          "prompt_tokens": len(sequences[offset+j])}
                output.write(json.dumps(record)+"\n")
                if offset==0 and j==0:
                    top = full_log_probs[j].topk(10)
                    meta["first_prompt_top_next_tokens"] = [{"id":int(token),"text":tokenizer.decode([int(token)]),
                        "probability":float(value.exp())} for token,value in zip(top.indices,top.values)]
            output.flush()
            if offset == 0 or (offset//args.batch_size+1) % 25 == 0:
                print(json.dumps({"completed": min(offset+args.batch_size, len(rows)),
                                  "total": len(rows), "elapsed_seconds": time.monotonic()-start}), flush=True)
    torch.cuda.synchronize()
    meta.update(rows=len(rows), elapsed_seconds=time.monotonic()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), status="complete")
    args.output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
