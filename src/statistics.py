"""
Core statistical functions for A/B Testing.
Includes: sample size calculator, SRM check, Z-test, Bayesian test.
"""
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower


def calculate_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Calculate required sample size per group.

    Args:
        baseline_rate: Current conversion rate (e.g. 0.10 = 10%)
        mde: Minimum Detectable Effect as absolute change (e.g. 0.02 = +2pp)
        alpha: Significance level (Type I error rate)
        power: Statistical power (1 - Type II error)
    """
    treatment_rate = baseline_rate + mde
    effect_size = proportion_effectsize(baseline_rate, treatment_rate)
    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    n = int(np.ceil(n))
    return {
        "n_per_group": n,
        "n_total": n * 2,
        "baseline_rate": baseline_rate,
        "treatment_rate": treatment_rate,
        "mde": mde,
        "relative_mde": mde / baseline_rate,
        "effect_size": effect_size,
        "alpha": alpha,
        "power": power,
    }


def check_srm(
    n_control: int,
    n_treatment: int,
    expected_split: float = 0.5,
    alpha: float = 0.05,
) -> dict:
    """
    Sample Ratio Mismatch (SRM) check using chi-square test.
    If SRM is detected, the experiment assignment is broken — results are invalid.
    """
    total = n_control + n_treatment
    expected_control = total * expected_split
    expected_treatment = total * (1 - expected_split)

    chi2, p_value = stats.chisquare(
        [n_control, n_treatment],
        [expected_control, expected_treatment],
    )
    return {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "total": total,
        "expected_split": expected_split,
        "actual_split_control": n_control / total,
        "chi2": chi2,
        "p_value": p_value,
        "srm_detected": p_value < alpha,
        "expected_control": int(expected_control),
        "expected_treatment": int(expected_treatment),
    }


def run_z_test(
    n_control: int,
    n_treatment: int,
    conv_control: int,
    conv_treatment: int,
    alpha: float = 0.05,
) -> dict:
    """
    Two-proportion Z-test (frequentist approach).
    Returns: rates, difference, confidence interval, z-stat, p-value.
    """
    rate_c = conv_control / n_control
    rate_t = conv_treatment / n_treatment

    counts = np.array([conv_treatment, conv_control])
    nobs = np.array([n_treatment, n_control])
    z_stat, p_value = proportions_ztest(counts, nobs, alternative="two-sided")

    # 95% CI for the difference
    se = np.sqrt(
        rate_c * (1 - rate_c) / n_control + rate_t * (1 - rate_t) / n_treatment
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = rate_t - rate_c
    ci_lower = diff - z_crit * se
    ci_upper = diff + z_crit * se

    # Retrospective power
    effect_size = proportion_effectsize(rate_c, rate_t)
    analysis = NormalIndPower()
    achieved_power = analysis.solve_power(
        effect_size=abs(effect_size),
        nobs1=n_control,
        alpha=alpha,
        alternative="two-sided",
    )

    return {
        "rate_control": rate_c,
        "rate_treatment": rate_t,
        "diff": diff,
        "relative_uplift": (rate_t - rate_c) / rate_c * 100,
        "z_stat": z_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_value < alpha,
        "alpha": alpha,
        "achieved_power": achieved_power,
    }


def bayesian_ab_test(
    conv_control: int,
    n_control: int,
    conv_treatment: int,
    n_treatment: int,
    n_samples: int = 100_000,
) -> dict:
    """
    Bayesian A/B test using Beta-Binomial conjugate model.
    Prior: Beta(1, 1) — uninformative uniform prior.
    Returns posterior samples and probability that treatment > control.
    """
    rng = np.random.default_rng(42)

    # Posterior parameters (prior Beta(1,1) + likelihood)
    a_c = conv_control + 1
    b_c = n_control - conv_control + 1
    a_t = conv_treatment + 1
    b_t = n_treatment - conv_treatment + 1

    samples_c = rng.beta(a_c, b_c, n_samples)
    samples_t = rng.beta(a_t, b_t, n_samples)

    prob_t_better = float(np.mean(samples_t > samples_c))
    expected_loss_if_control = float(np.mean(np.maximum(samples_t - samples_c, 0)))
    expected_loss_if_treatment = float(np.mean(np.maximum(samples_c - samples_t, 0)))

    # Credible interval for the difference
    diff_samples = samples_t - samples_c
    ci_95_low = float(np.percentile(diff_samples, 2.5))
    ci_95_high = float(np.percentile(diff_samples, 97.5))

    return {
        "prob_treatment_better": prob_t_better,
        "prob_control_better": 1 - prob_t_better,
        "expected_loss_control": expected_loss_if_control,
        "expected_loss_treatment": expected_loss_if_treatment,
        "samples_control": samples_c,
        "samples_treatment": samples_t,
        "diff_samples": diff_samples,
        "ci_95_low": ci_95_low,
        "ci_95_high": ci_95_high,
        "posterior_mean_control": a_c / (a_c + b_c),
        "posterior_mean_treatment": a_t / (a_t + b_t),
    }


def revenue_impact(
    rate_control: float,
    rate_treatment: float,
    avg_order_value: float,
    monthly_visitors: int,
) -> dict:
    """Project monthly revenue impact if treatment is shipped."""
    baseline_revenue = rate_control * monthly_visitors * avg_order_value
    treatment_revenue = rate_treatment * monthly_visitors * avg_order_value
    uplift = treatment_revenue - baseline_revenue
    return {
        "baseline_monthly": baseline_revenue,
        "treatment_monthly": treatment_revenue,
        "monthly_uplift": uplift,
        "annual_uplift": uplift * 12,
    }
