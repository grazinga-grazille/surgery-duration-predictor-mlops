"""Linear regression assumption diagnostics.

Checks the classical OLS assumptions on log-duration residuals and returns a
structured report. Designed for the surgery-duration feature matrix
(numeric + one-hot + SVD components).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.tools import add_constant


@dataclass
class AssumptionResult:
    name: str
    passed: bool
    statistic: float | None
    p_value: float | None
    detail: str


def _corr_residual_fitted(y_hat: np.ndarray, resid: np.ndarray) -> AssumptionResult:
    """Linearity proxy: residuals should be uncorrelated with fitted values."""
    if np.std(y_hat) < 1e-12 or np.std(resid) < 1e-12:
        return AssumptionResult(
            "linearity (resid vs fitted corr)",
            True,
            0.0,
            None,
            "Degenerate fitted/residual variance; skipped.",
        )
    r, p = stats.pearsonr(y_hat, resid)
    # For OLS with intercept, this correlation is ~0 by construction on *training*
    # residuals. We still report it; Rainbow is the stronger linearity check.
    return AssumptionResult(
        "linearity (resid vs fitted corr)",
        abs(r) < 0.05,
        float(r),
        float(p),
        f"|corr(ŷ, e)|={abs(r):.4f} (expect ≈0 for a well-specified linear fit).",
    )


def _rainbow_test(y: np.ndarray, X: np.ndarray) -> AssumptionResult:
    """Rainbow test for linearity (statsmodels)."""
    try:
        import statsmodels.api as sm

        ols = sm.OLS(y, add_constant(X, has_constant="add")).fit()
        f_stat, p_value = linear_rainbow(ols)
        return AssumptionResult(
            "linearity (Rainbow test)",
            p_value >= 0.05,
            float(f_stat),
            float(p_value),
            "H0: model is linear. Fail if p < 0.05.",
        )
    except Exception as exc:
        return AssumptionResult(
            "linearity (Rainbow test)",
            False,
            None,
            None,
            f"Could not compute Rainbow test: {exc}",
        )


def _normality_tests(resid: np.ndarray, sample_size: int = 5000) -> list[AssumptionResult]:
    """Jarque-Bera (full) + Shapiro-Wilk (subsample for n large)."""
    results: list[AssumptionResult] = []

    jb_stat, jb_p = stats.jarque_bera(resid)
    results.append(
        AssumptionResult(
            "normality (Jarque-Bera)",
            jb_p >= 0.05,
            float(jb_stat),
            float(jb_p),
            "H0: residuals are normal. Fail if p < 0.05.",
        )
    )

    rng = np.random.default_rng(42)
    sample = resid if len(resid) <= sample_size else rng.choice(resid, size=sample_size, replace=False)
    sh_stat, sh_p = stats.shapiro(sample)
    results.append(
        AssumptionResult(
            "normality (Shapiro-Wilk subsample)",
            sh_p >= 0.05,
            float(sh_stat),
            float(sh_p),
            f"H0: residuals are normal (n={len(sample)}). Fail if p < 0.05.",
        )
    )

    skew = float(stats.skew(resid))
    kurt = float(stats.kurtosis(resid))  # excess kurtosis
    results.append(
        AssumptionResult(
            "normality (skew / excess kurtosis)",
            abs(skew) < 0.5 and abs(kurt) < 1.0,
            None,
            None,
            f"skew={skew:.3f}, excess_kurtosis={kurt:.3f} "
            "(rule of thumb: |skew|<0.5 and |kurt|<1).",
        )
    )
    return results


def _homoscedasticity(resid: np.ndarray, y_hat: np.ndarray, X: np.ndarray) -> AssumptionResult:
    """Breusch-Pagan test against fitted values / design matrix."""
    try:
        Xc = add_constant(X, has_constant="add")
        lm, lm_p, f_stat, f_p = het_breuschpagan(resid, Xc)
        return AssumptionResult(
            "homoscedasticity (Breusch-Pagan)",
            lm_p >= 0.05,
            float(lm),
            float(lm_p),
            "H0: residual variance is constant. Fail if p < 0.05.",
        )
    except Exception as exc:
        # Fallback: correlate |resid| with fitted
        r, p = stats.pearsonr(y_hat, np.abs(resid))
        return AssumptionResult(
            "homoscedasticity (|e| vs fitted corr)",
            abs(r) < 0.1,
            float(r),
            float(p),
            f"Breusch-Pagan failed ({exc}); used |corr(ŷ, |e|)|={abs(r):.4f}.",
        )


def _independence(resid: np.ndarray) -> AssumptionResult:
    """Durbin-Watson (note: cases are not a true time series)."""
    dw = float(durbin_watson(resid))
    # Rule of thumb: ~2 is independent; <1.5 or >2.5 suggests autocorrelation
    passed = 1.5 <= dw <= 2.5
    return AssumptionResult(
        "independence (Durbin-Watson)",
        passed,
        dw,
        None,
        f"DW={dw:.3f} (≈2 ideal). Cases are not ordered in time — treat cautiously.",
    )


def _multicollinearity(
    X: pd.DataFrame,
    vif_max_features: int = 40,
    vif_row_sample: int = 8000,
) -> list[AssumptionResult]:
    """Condition number + VIF on a reduced feature set for tractability."""
    import warnings

    results: list[AssumptionResult] = []

    X_arr = X.to_numpy(dtype=float)
    # Condition number of standardized design (already scaled upstream ideally)
    singular_vals = np.linalg.svd(X_arr, compute_uv=False)
    cond = float(singular_vals.max() / max(singular_vals.min(), 1e-12))
    results.append(
        AssumptionResult(
            "multicollinearity (condition number)",
            cond < 30,
            cond,
            None,
            f"κ={cond:.1f} (<10 mild, 10–30 moderate, >30 severe).",
        )
    )

    # Prefer non-SVD columns for VIF (SVD components are orthogonal by design)
    non_svd = [c for c in X.columns if not str(c).startswith("SVD_")]
    vif_cols = non_svd if non_svd else list(X.columns[:vif_max_features])

    # Drop near-constant columns (rare one-hot levels) that make VIF unstable
    variances = X[vif_cols].var()
    vif_cols = [c for c in vif_cols if variances.get(c, 0) > 1e-8]
    if len(vif_cols) > vif_max_features:
        keep = []
        if "SurgicalPriority" in vif_cols:
            keep.append("SurgicalPriority")
        for col in variances[vif_cols].sort_values(ascending=False).index:
            if col not in keep:
                keep.append(col)
            if len(keep) >= vif_max_features:
                break
        vif_cols = keep

    rng = np.random.default_rng(42)
    if len(X) > vif_row_sample:
        idx = rng.choice(len(X), size=vif_row_sample, replace=False)
        X_vif = X.iloc[idx][vif_cols].reset_index(drop=True)
    else:
        X_vif = X[vif_cols].reset_index(drop=True)

    X_vif_const = add_constant(X_vif, has_constant="add")
    vifs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, col in enumerate(X_vif_const.columns):
            if col == "const":
                continue
            try:
                vifs.append((col, float(variance_inflation_factor(X_vif_const.values, i))))
            except Exception:
                vifs.append((col, float("nan")))

    vif_series = pd.Series({k: v for k, v in vifs}).replace([np.inf, -np.inf], np.nan).dropna()
    max_vif = float(vif_series.max()) if len(vif_series) else float("nan")
    mean_vif = float(vif_series.mean()) if len(vif_series) else float("nan")
    n_high = int((vif_series > 10).sum()) if len(vif_series) else 0
    results.append(
        AssumptionResult(
            "multicollinearity (VIF on non-SVD features)",
            max_vif < 10 if np.isfinite(max_vif) else False,
            max_vif,
            None,
            f"max VIF={max_vif:.2f}, mean VIF={mean_vif:.2f}, "
            f"n(VIF>10)={n_high} over {len(vif_cols)} columns "
            f"(SVD components excluded — orthogonal by construction).",
        )
    )
    return results


def validate_linear_assumptions(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    y_hat: np.ndarray,
) -> dict:
    """Run OLS assumption checks and return a serialisable report.

    Parameters
    ----------
    X : feature matrix used to fit the linear model (preferably scaled).
    y : training target (log duration).
    y_hat : in-sample fitted values on the log scale.
    """
    y = np.asarray(y, dtype=float).ravel()
    y_hat = np.asarray(y_hat, dtype=float).ravel()
    resid = y - y_hat

    checks: list[AssumptionResult] = []
    checks.append(_corr_residual_fitted(y_hat, resid))
    checks.append(_rainbow_test(y, X.to_numpy(dtype=float)))
    checks.extend(_normality_tests(resid))
    checks.append(_homoscedasticity(resid, y_hat, X.to_numpy(dtype=float)))
    checks.append(_independence(resid))
    checks.extend(_multicollinearity(X))

    n_pass = sum(1 for c in checks if c.passed)
    suitable = all(
        c.passed
        for c in checks
        if c.name
        in {
            "linearity (Rainbow test)",
            "homoscedasticity (Breusch-Pagan)",
            "normality (Jarque-Bera)",
            "multicollinearity (condition number)",
        }
    )

    return {
        "n_checks": len(checks),
        "n_passed": n_pass,
        "ols_assumptions_ok": suitable,
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid)),
        "checks": [asdict(c) for c in checks],
    }


def format_assumption_report(report: dict) -> str:
    """Pretty-print the assumption report for the terminal."""
    lines = [
        "── Linear Regression Assumption Checks ─────────────────",
        f"  Residual mean={report['residual_mean']:.4e}  "
        f"std={report['residual_std']:.4f}",
        f"  Passed {report['n_passed']}/{report['n_checks']} checks",
        f"  OLS suitable (strict gate)? "
        f"{'YES' if report['ols_assumptions_ok'] else 'NO — prefer tree/boosted models'}",
        "",
    ]
    for c in report["checks"]:
        flag = "PASS" if c["passed"] else "FAIL"
        stat = f"stat={c['statistic']:.4g}" if c["statistic"] is not None else "stat=—"
        p = f"p={c['p_value']:.4g}" if c["p_value"] is not None else "p=—"
        lines.append(f"  [{flag}] {c['name']}")
        lines.append(f"         {stat}  {p}")
        lines.append(f"         {c['detail']}")
    return "\n".join(lines)
