# Uplift Modeling on Criteo — which customers do ads actually persuade?

A causal-measurement project on a large, real, randomized advertising experiment
(**Criteo, ~14M users**). It answers two questions advertisers get wrong all the time:
did the ads actually *cause* the sales, and *who* is worth advertising to?

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

| Finding | Result |
|---|---|
| Baseline buy-rate (control) | ~0.20% |
| Buy-rate when shown the ad | ~0.30% |
| Real incremental lift | **~54% relative**, highly significant (p < 0.001) |
| Payoff concentration | **~top 30% of customers = ~half of all extra sales** |

**Bottom line:** the ads work, but the effect is concentrated. Advertising to the
persuadable minority captures most of the sales for a fraction of the spend — and the
model even flags *"sleeping dogs,"* customers the ad slightly backfires on.

## Files

- `criteo_uplift_project.ipynb` — the full project, explained step by step
- `criteo_uplift_project.html` — a rendered preview (opens in any browser, no setup)
- `criteo_project_src.py` — the same notebook as a plain script

## How to run it

**On Kaggle (uses the real 14M-row data):** open the Criteo *Uplift Modeling* dataset,
create a notebook, paste in this project, and Run All. The loader finds the real CSV
automatically — no path editing.

**Anywhere else (no download needed):**

    pip install numpy pandas scipy scikit-learn matplotlib
    python criteo_project_src.py

If the real CSV isn't present, the notebook automatically builds a realistic stand-in
with the same structure so every cell still runs and the full method is demonstrated.

## Methods

Randomized treatment/control comparison · two-proportion z-test with 95% confidence
intervals · T-learner (gradient-boosted models) for individual treatment effects ·
out-of-sample validation via uplift-by-decile and Qini curve.
