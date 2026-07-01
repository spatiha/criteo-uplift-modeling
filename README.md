# Uplift Modeling on Criteo — which customers do ads actually persuade?

A causal-measurement project on a large, real, randomized advertising experiment
(**Criteo, ~14M users**). It answers two questions advertisers get wrong all the time:
did the ads actually *cause* the sales, and *who* is worth advertising to?

## View the project

- **Full analysis (notebook)** — renders here on GitHub with every chart:
  [`criteo_uplift_project.ipynb`](https://github.com/spatiha/criteo-uplift-modeling/blob/main/criteo_uplift_project.ipynb)
- **Illustrated walkthrough** — a plain-English tour of the method and findings:
  [open the walkthrough](https://raw.githack.com/spatiha/criteo-uplift-modeling/main/project_walkthrough.html)

## The problem

Advertisers routinely take credit for sales that would have happened without any ad.
If a customer was going to buy anyway, the ad spent on them changed nothing. The only
spend that matters goes to customers the ad genuinely *persuades* — and finding those
people is what this project does.

## What I did

1. **Measured the real effect.** Compared people shown ads (*treatment*) to a randomized
   group who weren't (*control*). The difference in buy-rate is the ad's true causal
   impact, tested for significance.
2. **Scored each customer's persuadability.** Built an *uplift model* (a T-learner: two
   gradient-boosted models, one trained on the ad group and one on the control group; the
   gap between their predictions is each person's individual uplift).
3. **Validated and turned it into a decision.** Confirmed the score tracks real behavior
   (uplift-by-decile chart) and measured how concentrated the payoff is (Qini curve), then
   made a budget recommendation.

## Key results

| Finding                     | Result                                               |
| --------------------------- | ---------------------------------------------------- |
| Baseline buy-rate (control) | ~0.20%                                               |
| Buy-rate when shown the ad  | ~0.30%                                               |
| Real incremental lift       | **~54% relative**, highly significant (p < 0.001)    |
| Payoff concentration        | **~top 30% of customers = ~half of all extra sales** |

**Bottom line:** the ads work, but the effect is concentrated. Advertising to the
persuadable minority captures most of the sales for a fraction of the spend — and the
model even flags *"sleeping dogs,"* customers the ad slightly backfires on.

## Repository contents

| File                          | What it is                                                        |
| ----------------------------- | ----------------------------------------------------------------- |
| `criteo_uplift_project.ipynb` | The full project, explained step by step (renders on GitHub)      |
| `project_walkthrough.html`    | Illustrated plain-English walkthrough (best viewed via the link above) |
| `criteo_uplift_project.html`  | Static rendered copy of the notebook                              |
| `criteo_project_src.py`       | The same analysis as a plain Python script                        |

## Data source

Criteo Uplift Prediction dataset (v2.1) — ~14M users drawn from real randomized
advertising experiments, released by the **Criteo AI Lab**. Published in *"A Large Scale
Benchmark for Uplift Modeling"* (Diemert, Betlei, Renaudin & Amini, AdKDD 2018). Each row
has 12 anonymized features (`f0`–`f11`), a binary `treatment` flag, and two outcome labels
(`visit`, `conversion`); about 85% of users are treated.

## How to run it

**On Kaggle (uses the real 14M-row data):** open a notebook, add the *Uplift Modeling*
dataset, paste in `criteo_project_src.py`, and Run All. `kagglehub` is preinstalled and the
data loads automatically. Set `MAX_ROWS = None` to use all rows.

**Locally:**

```
pip install kagglehub numpy pandas scipy scikit-learn matplotlib
python criteo_project_src.py
```

The first local run asks for a free Kaggle API token (Kaggle → *Settings → Create New
Token*). If the data can't be reached, the notebook builds a realistic stand-in with the
same structure so it always runs.

## Limitations & next steps

- **Treatment vs. exposure.** In Criteo, `treatment` means the advertiser entered the ad
  auction for a user, while `exposure` means the ad was actually shown. This analysis measures
  *intent-to-treat* (the effect of being targeted), which keeps the randomization clean.
- **Rare outcome.** Conversion is ~0.3%, so stable estimates need large samples — the full
  ~14M-row run gives tight intervals. I validated the method on a real 10k-row Criteo sample
  (the *visit* lift reproduces at p < 0.001); the headline conversion numbers come from the full
  dataset.
- **One uplift method.** I used a T-learner for interpretability; a natural next step is to
  benchmark it against S-learner, X-learner, and causal-forest approaches on Qini / AUUC.
- **From conversions to profit.** With cost and margin inputs, the Qini curve becomes a
  spend-optimization tool — solving for the audience cutoff that maximizes incremental *profit*.

## Methods

Randomized treatment/control comparison · two-proportion z-test with 95% confidence
intervals · T-learner (gradient-boosted models) for individual treatment effects ·
out-of-sample validation via uplift-by-decile and Qini curve.
