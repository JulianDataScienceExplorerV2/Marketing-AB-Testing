"""
Synthetic e-commerce A/B test data generator.
Simulates a landing page test for a Latin American e-commerce store.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_ab_data(
    n_users: int = 45_000,
    control_rate: float = 0.104,
    treatment_rate: float = 0.127,
    avg_revenue: float = 88.0,
    revenue_std: float = 34.0,
    days: int = 14,
    split: float = 0.50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic A/B test dataset.

    Scenario:
        An e-commerce store (Colombia/LATAM) tests a redesigned product
        page (treatment) vs. the original (control) over 14 days.

    Returns:
        DataFrame with columns:
        user_id, timestamp, group, page_version, converted,
        revenue, device, country, session_duration_s
    """
    rng = np.random.default_rng(seed)
    n_control = int(n_users * split)
    n_treatment = n_users - n_control
    start = datetime(2024, 10, 1)

    records = []
    for group, n, rate, page in [
        ("control", n_control, control_rate, "old_page"),
        ("treatment", n_treatment, treatment_rate, "new_page"),
    ]:
        converted = rng.binomial(1, rate, n)

        revenue = np.where(
            converted == 1,
            rng.normal(avg_revenue, revenue_std, n).clip(min=5.0),
            0.0,
        ).round(2)

        devices = rng.choice(
            ["mobile", "desktop", "tablet"],
            size=n,
            p=[0.61, 0.31, 0.08],
        )
        countries = rng.choice(
            ["CO", "MX", "AR", "BR", "PE"],
            size=n,
            p=[0.35, 0.28, 0.16, 0.13, 0.08],
        )
        session_s = rng.integers(30, 600, size=n)

        day_offsets = rng.integers(0, days, n)
        hour_offsets = rng.integers(0, 24, n)
        minute_offsets = rng.integers(0, 60, n)
        timestamps = [
            start + timedelta(days=int(d), hours=int(h), minutes=int(m))
            for d, h, m in zip(day_offsets, hour_offsets, minute_offsets)
        ]

        user_ids = rng.integers(10_000_000, 99_999_999, n)

        for i in range(n):
            records.append(
                {
                    "user_id": int(user_ids[i]),
                    "timestamp": timestamps[i],
                    "group": group,
                    "page_version": page,
                    "converted": int(converted[i]),
                    "revenue": float(revenue[i]),
                    "device": devices[i],
                    "country": countries[i],
                    "session_duration_s": int(session_s[i]),
                }
            )

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
