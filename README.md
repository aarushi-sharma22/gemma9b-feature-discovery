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
  feature_stats-gemma.csv        # per-feature selection statistics (derived, aggregate)
  behavioral_effects-gemma.csv   # behavioural effects per demographic x domain x layer (derived, aggregate)
  codebook                       # ESS variable codebook (JSON)
requirements.txt
LICENSE                          # MIT (covers this repo's code)
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

## Data

**Included (aggregate derived outputs, no respondent identifiers):** `feature_stats-gemma.csv` (per-feature selection statistics by demographic x domain) and `behavioral_effects-gemma.csv` (behavioural effects by demographic x domain x layer). These are sufficient to reproduce the corresponding figures and summary statistics.

**Not included:**
- *Raw respondent data.* It comes from the European Social Survey CRONOS-3 panel, Wave 4 (ESS ERIC, 2026), fielded 2024-2025 across 11 European countries. Obtain it directly from ESS (see below).
- *Per-pair causal-validation results.* The per-pair output file embedded respondent IDs in its key, so it is omitted; regenerate it by running the causal-validation cell of the notebook on the reconstructed sample.

### ESS licensing

ESS data is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0) and is available without restriction for not-for-profit purposes; ESS documentation is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0). The licence permits non-commercial redistribution with attribution and share-alike, but ESS **recommends** linking to the ESS Data Portal rather than hosting datasets externally, and we follow that recommendation: the raw panel is not redistributed here. The aggregate statistics we do include are derived from, and should be cited as adapted from, the dataset below.

### Reconstructing the input sample

1. Register and download CRONOS-3 Wave 4 from the ESS Data Portal: https://www.europeansocialsurvey.org/ (edition 1.1 DOI: https://doi.org/10.21338/cron3w4e01_1). Experiments used edition 1.0; edition 1.1 (Feb 2026) corrected a reversed scale coding for `w4eq1`, which does not affect this analysis (it uses demographic variables and question text only, not respondent-level responses).
2. Select 150 respondents with complete demographic information across all five attributes (`gndr`, `age`, `eisced`, `hincfel`, `vote`), partitioned into 120 for feature selection and 30 for held-out validation. The exact respondent set depends on the stratified draw; the per-respondent ID list is not redistributed here, so the precise sample is reproducible only up to the selection criteria.
3. Use the columns `idno, cntry, age, gndr, hincfel, vote, eisced` plus the survey items, with the survey-answer columns left blank (the model fills them in).
4. Demographic contrasts (extreme values, for feature discovery): income (wealthy vs. poor), age (75 vs. 22), gender (man vs. woman), education (PhD vs. no secondary), vote (regular vs. non-voter). Non-tested attributes use each respondent's real ESS values rendered as natural language. This yields 66,000 contrastive pairs for selection and 16,500 for validation across 110 questions in five domains.

Please cite the ESS data per their requirements: European Social Survey European Research Infrastructure (ESS ERIC) (2026) CRONOS3 Wave 4 [Data set]. Sikt. https://doi.org/10.21338/cron3w4e01_1.

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


## License

The code in this repository is released under the MIT License (see `LICENSE`).

The data is governed separately: ESS CRONOS-3 Wave 4 is CC BY-NC-SA 4.0 and ESS documentation is CC BY-SA 4.0; the aggregate ESS-derived statistics in `data/` carry those terms and the citation above, regardless of the MIT license on the code.

