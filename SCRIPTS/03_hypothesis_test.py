"""
Hypothesis testing: one-proportion z-tests on LDA topic proportions.

Positive reviews: H0 p >= 0.30 vs Ha p < 0.30 (quality of food topic).
Negative reviews: H0 p >= 0.30 vs Ha p < 0.30 (customer service topic).

"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "DATA"
# LDA pipeline output: one row per review with topic_label and sentiment
TOPICS_CSV = DATA_DIR / "McDonalds_Reviews_With_Topics.csv"

# Topic labels from LDA (k=6)
TOPIC_BAD_SERVICE = "Bad service, wrong orders, rude staff"  # (1)
TOPIC_FAST_SERVICE = "Fast and quick service, clean place, friendly staff"  # (2)
TOPIC_GOOD_PRICE_QUALITY = "Overall good price, quality, service"  # (3)
TOPIC_FOOD_QUALITY_ISSUES = "Food quality issues (cold/old fries, burgers, nuggets)"  # (4)
TOPIC_LONG_WAIT = "Long wait times, slow drive-thru"  # (5)
TOPIC_MISCELLANEOUS = "Miscellanous, general complains about environment"  # (6)

ALL_TOPICS = [
    TOPIC_BAD_SERVICE,
    TOPIC_FAST_SERVICE,
    TOPIC_GOOD_PRICE_QUALITY,
    TOPIC_FOOD_QUALITY_ISSUES,
    TOPIC_LONG_WAIT,
    TOPIC_MISCELLANEOUS,
]

# Topics used in the two one-proportion z-tests
FOOD_QUALITY_TOPIC = TOPIC_GOOD_PRICE_QUALITY  # positive reviews
CUSTOMER_SERVICE_TOPIC = TOPIC_BAD_SERVICE  # negative reviews

# Null proportion and significance level for both tests
P0 = 0.30
ALPHA = 0.05


def one_proportion_ztest_left(x, n, p0=P0):
    """Left-tailed one-proportion z-test: H0 p >= p0 vs Ha p < p0."""
    # Sample proportion: count in topic / total count
    p_hat = x / n
    # Standard error under H0
    se = np.sqrt(p0 * (1 - p0) / n)
    # Test statistic
    z = (p_hat - p0) / se if se > 0 else 0
    # Left-tailed p-value: P(Z <= z)
    p_value = stats.norm.cdf(z)
    return z, p_value, p_hat


def main():
    if not TOPICS_CSV.exists():
        raise FileNotFoundError(f"Expected {TOPICS_CSV}. Run LDA pipeline first.")

    with open(TOPICS_CSV, "r", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f)
    # Need topic and sentiment for both tests
    df = df.dropna(subset=["topic_label", "sentiment"])

    # Split by sentiment (exclude neutral)
    positive = df[df["sentiment"] == "positive"].copy()
    negative = df[df["sentiment"] == "negative"].copy()

    n_pos = len(positive)
    n_neg = len(negative)

    # --- All 6 topics: counts and proportions by sentiment ---
    def topic_counts_proportions(subset, n):
        """Count and proportion in each of the 6 topics (order matches ALL_TOPICS)."""
        counts = [ (subset["topic_label"] == t).sum() for t in ALL_TOPICS ]
        props = [ c / n for c in counts ]
        return counts, props

    pos_counts, pos_props = topic_counts_proportions(positive, n_pos)
    neg_counts, neg_props = topic_counts_proportions(negative, n_neg)

    print("Distribution across all 6 topics\n")
    print("Positive reviews (n = {}):".format(n_pos))
    for t, c, p in zip(ALL_TOPICS, pos_counts, pos_props):
        print("  {:8.2%}  {:5d}  {}".format(p, c, t))
    print("\nNegative reviews (n = {}):".format(n_neg))
    for t, c, p in zip(ALL_TOPICS, neg_counts, neg_props):
        print("  {:8.2%}  {:5d}  {}".format(p, c, t))

    # Chi-squared goodness-of-fit: distribution across 6 topics vs uniform (1/6 each)
    exp_pos = np.full(6, n_pos / 6)
    exp_neg = np.full(6, n_neg / 6)
    chi2_pos, p_chi_pos = stats.chisquare(pos_counts, exp_pos)
    chi2_neg, p_chi_neg = stats.chisquare(neg_counts, exp_neg)
    print("\nChi-squared (all 6 topics vs uniform):")
    print("  Positive: chi2 = {:.3f}, p = {:.4f}  →  {}".format(
        chi2_pos, p_chi_pos, "Reject uniform" if p_chi_pos < ALPHA else "Do not reject uniform"))
    print("  Negative: chi2 = {:.3f}, p = {:.4f}  →  {}".format(
        chi2_neg, p_chi_neg, "Reject uniform" if p_chi_neg < ALPHA else "Do not reject uniform"))

    # One-proportion z-tests
    x_pos = (positive["topic_label"] == FOOD_QUALITY_TOPIC).sum()
    x_neg = (negative["topic_label"] == CUSTOMER_SERVICE_TOPIC).sum()
    z_pos, pval_pos, p_hat_pos = one_proportion_ztest_left(x_pos, n_pos)
    z_neg, pval_neg, p_hat_neg = one_proportion_ztest_left(x_neg, n_neg)

    print("\nOne-proportion z-test (H0: p >= 0.30, alpha = 0.05)\n")
    print(f"Positive (food quality):  p̂ = {p_hat_pos:.2%}, z = {z_pos:.3f}, p = {pval_pos:.4f}  →  {'Reject H0' if pval_pos < ALPHA else 'Do not reject H0'}")
    print(f"Negative (customer svc): p̂ = {p_hat_neg:.2%}, z = {z_neg:.3f}, p = {pval_neg:.4f}  →  {'Reject H0' if pval_neg < ALPHA else 'Do not reject H0'}")

    return {
        "positive": {"x": x_pos, "n": n_pos, "z": z_pos, "p_value": pval_pos, "reject": pval_pos < ALPHA,
                     "topic_counts": pos_counts, "topic_proportions": pos_props, "chi2": chi2_pos, "chi2_p": p_chi_pos},
        "negative": {"x": x_neg, "n": n_neg, "z": z_neg, "p_value": pval_neg, "reject": pval_neg < ALPHA,
                     "topic_counts": neg_counts, "topic_proportions": neg_props, "chi2": chi2_neg, "chi2_p": p_chi_neg},
    }


# Run main when script is executed directly
if __name__ == "__main__":
    main()
