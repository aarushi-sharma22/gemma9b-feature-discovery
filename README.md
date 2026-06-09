# Encoding Without Influence: Dissociating Demographic Representation from Causal Effect in Large Language Models

Code and data for the Gemma 2 9B experiments in *Encoding Without Influence: Dissociating Demographic Representation from Causal Effect in Large Language Models* (Sharma & Le, TMLR).

> **Scope note:** This repository covers the **Gemma 2 9B** experiments only. The paper also reports Qwen 2.5 7B and Llama 3.1 8B; those pipelines are not included here.

## What this does

The pipeline identifies sparse autoencoder (SAE) features whose activations encode respondent demographics (income, age, gender, education, vote) across five survey domains, then tests whether those *encoding* features have a *causal* effect on the model's survey responses via activation patching, steering, and ablation. The headline result is a dissociation: features can robustly encode a demographic without causally driving the behaviour.

Model: `google/gemma-2-9b-it`. SAEs: `google/gemma-scope-9b-pt-res` (GemmaScope, 16K width), accessed through TransformerLens / SAE-Lens.

## Repository layout

```
code/
  gemma-9b-full-pipeline.ipynb   # end-to-end pipeline (Colab-ready)
  sequence_level_scoring.py      # scoring-method validation (Appendix D.2)
data/
  feature_stats-gemma.csv        # per-feature selection statistics (derived)
  behavioral_effects-gemma.csv   # behavioural effects per demographic x domain x layer (derived)
  validation_results-gemma.csv   # causal validation results (derived)
  codebook                       # ESS variable codebook (JSON)
  stratified_sample_template.csv # HEADER ONLY - see "Data" below
requirements.txt
LICENSE                          # TODO: add a license (see below)
```

## Pipeline stages (notebook cell order)

1. Install + imports + Hugging Face auth (Gemma is gated; accept the license first).
2. Model and GemmaScope SAE loading.
3. Contrastive prompt generation from real respondent demographics.
4. Vocabulary-sensitivity check (3 paraphrase variants per demographic).
5. Feature extraction across 8 layers in a single forward pass.
6. Statistical feature selection (noise filters, FDR-corrected t-tests, effect-size threshold).
7. Extraction figures and tables.
8. Causal validation (patching, steering, ablation) and analysis.
9. Feature interpretation (encoding matrix, location analysis).
10. Concern diagnostics (robustness checks).

`sequence_level_scoring.py` runs separately, after the causal-validation cell, and reproduces the scoring-method sensitivity analysis in Appendix D.2 (it depends on objects defined in that cell: `model`, `tokenizer`, `sae_manager`, `prompts_df`, `compute_ev`, `build_pairs`, etc.).

## Setup

**Colab (recommended; the notebook is written for it):** open the badge link at the top of the notebook. You need a GPU runtime and a Hugging Face token with access to gated Gemma.

**Local:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Requires a CUDA GPU with enough memory for Gemma 2 9B. Set your Hugging Face token in the auth cell (or via `HF_TOKEN`); never commit a real token.

## Data

The **derived** outputs needed to reproduce the figures and statistics are included in `data/` (`*-gemma.csv`).

The **raw respondent data is NOT included.** It comes from the European Social Survey CRONOS-3 panel, Wave 4 (ESS ERIC, 2026), a cross-national self-completion panel fielded in 2024-2025 across 11 European countries. The data is governed by ESS terms of use, which (to our reading) do not permit redistribution; obtain it directly from ESS. The analysis uses respondent demographic variables and question text only, not respondent-level response data. Experiments used edition 1.0; edition 1.1 (Feb 2026) corrected a reversed scale coding for `w4eq1`, which does not affect this analysis.

To reconstruct the input sample:

1. Register and download CRONOS-3 Wave 4 from the ESS Data Portal: https://www.europeansocialsurvey.org/ (DOI for edition 1.1: https://doi.org/10.21338/cron3w4e01_1).
2. Select 150 respondents with complete demographic information across all five attributes (`gndr`, `age`, `eisced`, `hincfel`, `vote`), partitioned into 120 for feature selection and 30 for held-out validation. The exact respondent IDs used are recoverable from the `pair_key` column of `validation_results-gemma.csv` (format: `<idno>_<question>_<demographic>`), so you can match the precise sample rather than re-deriving it.
3. Take the variable columns listed in `data/stratified_sample_template.csv` (`idno, cntry, age, gndr, hincfel, vote, eisced` plus the survey items). The survey-answer columns are left blank in the template because the model fills them in.
4. We use extreme demographic contrasts for feature discovery: income (wealthy vs. poor), age (75 vs. 22), gender (man vs. woman), education (PhD vs. no secondary), and vote (regular vs. non-voter). Non-tested attributes use each respondent's real ESS values rendered as natural language. This yields 66,000 contrastive pairs for selection and 16,500 for validation across 110 questions in five domains.

If you determine the ESS license permits redistribution of this subsample, you may include the populated file instead; verify the terms first.

## Citing

```
@article{sharma_le_encoding,
  title   = {Encoding Without Influence: Dissociating Demographic Representation from Causal Effect in Large Language Models},
  author  = {Sharma, Aarushi and Le, Phong},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=TQbXHsI3Lm}
}
```
<!-- Confirm the year matches the official TMLR publication date before release. -->

## License

<!-- TODO: add a LICENSE file. MIT is the simplest permissive choice for research code (Apache-2.0 if you want an explicit patent grant). Confirm with St Andrews / your funder that you are cleared to open-source PhD code before committing. The ESS data terms govern the data regardless of the code license. -->
