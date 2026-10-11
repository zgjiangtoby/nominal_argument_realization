# LLM Role Expectations and Paired Argument Realization in Spanish Nominalizations

The code for *LLM Role Expectations and Paired Argument Realization in Spanish Nominalizations*.

Extract linked agent/patient pairs from AnCora, measure role-referent recovery, and compare realization models under document and lemma holdouts.

## Environment

Run the commands below in Bash. Use Python 3.11 and a CUDA GPU with enough memory for FP32 inference; the 8B runs used GPUs reporting 49,140 MiB. Keep the batch settings below when comparing the reported results.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 pandas==2.3.3 scipy==1.16.3 \
  transformers==4.57.2 tokenizers==0.22.1 accelerate==1.14.0 \
  safetensors==0.7.0 sentencepiece==0.2.1 huggingface-hub==0.36.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data models work
```

Use new output directories for a run. Inference writes score metadata beside the score file; analyses use those files and intermediate outputs from the earlier commands. Raw texts, labels, model weights and generated results are not included. Obtain the sources below under their own terms and retain their notices.

```bash
python -m pip install scikit-learn==1.7.2
```

## AnCora XML and paired units

```bash
git clone https://github.com/PabloVelascoCrespo/MCR_to_Prolog.git data/ancora
git -C data/ancora checkout 1addb59e3d9f18c794555d22a2efba263e6588a3
python extract_pairs.py data/ancora/Corpus/Ancora --output work/coverage
python summarize_support.py data/ancora/Corpus/Ancora --coverage work/coverage --output work/coverage
python analyze_pairs.py --coverage work/coverage --output work/baseline
python prepare_entity_probe.py --source-root data/ancora/Corpus/Ancora --output work/probes
```

The extractor reads the XML file list from the pinned Git tree, in lexicographic order. Keep the checkout's `.git` directory: it supplies the exact 1,635-file inventory and byte sizes. The source is the Spanish AnCora XML copy under `Corpus/Ancora`; a differently normalized download may not have the same bytes. Its original corpus terms still apply.

The baseline creates the paired units and document/lemma folds used by probe preparation. Expected eligible probes cover 1,759 pairs and 14,072 prompts. Unlinked roles and ambiguous scopes remain explicitly marked. The prompt strings, candidate ordering and role/form eligibility are defined in the preparation code.

## Inference and realization models

```bash
qwen_revision=b968826d9c46dd6066d109eabc6255188de91218
euro_revision=a25c7fa65fc2a644e6270b8940dbe295b51da681
hf download Qwen/Qwen3-8B --revision "$qwen_revision" --local-dir "models/$qwen_revision"
hf download utter-project/EuroLLM-1.7B-Instruct --revision "$euro_revision" --local-dir "models/$euro_revision"
```

Keep the revision hash as each model directory's basename: the comparisons check it against the score metadata. Inference uses only these local files.

```bash
for checkpoint in qwen euro; do
  if [ "$checkpoint" = qwen ]; then
    model="$qwen_revision"; model_name=Qwen3-8B; batch=8
  else
    model="$euro_revision"; model_name=EuroLLM-1.7B-Instruct; batch=32
  fi
  python score_choices.py --model "models/$model" --input work/probes/inputs.jsonl \
    --output "work/scores/$checkpoint.jsonl" --batch-size "$batch" --max-tokens 4096 \
    --dtype float32 --check-batch-equivalence --check-full-forward
  python analyze_probe.py --scores "work/scores/$checkpoint.jsonl" --refits 200 \
    --output "work/analysis_$checkpoint"
  python diagnose_role_confidence.py --scores "work/scores/$checkpoint.jsonl" \
    --reference-analysis "work/analysis_$checkpoint" --model-name "$model_name" \
    --output "work/confidence_$checkpoint"
  python analyze_calibrated_realization.py --scores "work/scores/$checkpoint.jsonl" \
    --reference-analysis "work/analysis_$checkpoint" --confidence "work/confidence_$checkpoint" \
    --model-name "$model_name" --output "work/calibrated_$checkpoint"
done
python compare_calibrated_realization.py --qwen work/calibrated_qwen --euro work/calibrated_euro \
  --output work/comparison_calibrated
```

The confidence diagnostic fits five inverse temperatures on each holdout axis per checkpoint. The calibrated realization analysis refits temperature and prediction heads inside the same outer fold, with 2,000 conditional draws and 200 joint refits. It retains the raw analysis and combines all eight calibrated comparisons. A checked Newton recovery is available for an L-BFGS convergence warning at the same penalized objective. Outputs include recovery summaries, realization losses, coefficients and paired draw records.
