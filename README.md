Encoding Without Influence: Dissociating Demographic Representation from Causal Effect in Large Language Models
Code and data for the Gemma 2 9B experiments in Encoding Without Influence: Dissociating Demographic Representation from Causal Effect in Large Language Models (Sharma & Le, TMLR).

Scope note: This repository covers the Gemma 2 9B experiments only.

What this does
The pipeline identifies sparse autoencoder (SAE) features whose activations encode respondent demographics (income, age, gender, education, vote) across five survey domains, then tests whether those encoding features have a causal effect on the model's survey responses via activation patching. The headline result is a dissociation: features can robustly encode a demographic without causally driving the behaviour.
Model: google/gemma-2-9b-it. SAEs: google/gemma-scope-9b-pt-res (GemmaScope), accessed through TransformerLens / SAE-Lens.
Repository layout
code/
  gemma-9b-full-pipeline.ipynb   # end-to-end pipeline (Colab-ready)
data/
  feature_stats-gemma.csv        # per-feature selection statistics (derived)
  behavioral_effects-gemma.csv   # per-layer behavioural effects (derived)
  validation_results-gemma.csv   # causal validation results (derived)
  codebook                       # CRONOS-4 variable codebook (JSON)
  stratified_sample_template.csv # HEADER ONLY — see "Data" below
requirements.txt
LICENSE                          # TODO: add a license (see below)
Pipeline stages (notebook cell order)

Install + imports + Hugging Face auth (Gemma is gated — accept the license first).
Model and GemmaScope SAE loading.
Contrastive prompt generation from real respondent demographics.
Vocabulary-sensitivity check (3 paraphrase variants per demographic).
Feature extraction across 8 layers in a single forward pass.
Statistical feature selection (noise filters → FDR-corrected t-tests → effect-size threshold).
Extraction figures and tables.
Causal validation (activation patching) and analysis.
Feature interpretation (encoding matrix, location analysis).
Concern diagnostics (robustness checks).

Setup
Colab (recommended — the notebook is written for it): open the badge link at the top of the notebook. You need a GPU runtime and a Hugging Face token with access to gated Gemma.
Local:
bashpython -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
Requires a CUDA GPU with enough memory for Gemma 2 9B. Set your Hugging Face token in the auth cell (or via HF_TOKEN); never commit a real token.



Data
The derived outputs needed to reproduce the figures and statistics are included in data/ (*-gemma.csv).
The raw respondent data is NOT included. It comes from the European Social Survey CRONOS-3 panel (Wave 4) and is governed by ESS terms of use, which (to our reading) do not permit redistribution. To reconstruct the input sample:

Register and download CRONOS-3 Wave 4 from the ESS Data Portal: https://www.europeansocialsurvey.org/

<!-- TODO: state the exact dataset edition/DOI you used. -->

Select the variables in data/stratified_sample_template.csv (the header lists the demographic columns idno, cntry, age, gndr, hincfel, vote, eisced plus the survey items used).
Apply the stratified sampling used in the paper: <!-- TODO: describe strata (e.g. by country/age/gender) and the random seed, so others land on the same 200 respondents. -->
Save the result with the template's column order. The survey-answer columns are left blank in the template because the model fills them in.

If you believe the ESS license does permit redistribution of this subsample, you can include the populated file instead — but verify the terms first.
Citing
<!-- TODO: add BibTeX once the OpenReview/TMLR entry is final. -->
@article{sharma_le_encoding,
  title  = {Encoding Without Influence: Dissociating Demographic Representation from Causal Effect in Large Language Models},
  author = {Sharma, Aarushi and Le, Phong},
  journal = {Transactions on Machine Learning Research},
  year   = {YYYY},
  url    = {https://openreview.net/forum?id=XXXX}
}
License
<!-- TODO: choose a license. MIT or Apache-2.0 are common for research code; add a LICENSE file. Note the ESS data terms govern the data regardless of your code license. -->
