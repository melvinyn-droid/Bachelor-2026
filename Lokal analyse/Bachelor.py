#%%
"""Fuld out-of-sample HRP-backtest pa CRSP-aktier."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import signal
import time

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import PercentFormatter
except ModuleNotFoundError:
    plt = None
    mcolors = None
    PercentFormatter = None

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


DATA_PATH = Path(__file__).with_name("bachelor data v1.csv")

DEFAULT_LOOKBACK_MONTHS = 120
DEFAULT_TOP_N_UNIVERSE = 100
DEFAULT_ROLLING_SHARPE_WINDOW_MONTHS = 24
DEFAULT_LONG_ONLY_MV_MAXITER = 20000
DEFAULT_WEIGHT_SNAPSHOT_MONTH = "2015-05"
DEFAULT_WEIGHT_DISTRIBUTION_TOP_N = 15
MODEL_COLUMNS = {
    "Equal Weight": "ret_equal_weight",
    "Minimum Variance Classical": "ret_min_variance_classical",
    "Minimum Variance Long Only": "ret_min_variance_long_only",
    "HRP": "ret_hrp",
}
MODEL_TURNOVER_COLUMNS = {
    "Equal Weight": "turnover_equal_weight",
    "Minimum Variance Classical": "turnover_min_variance_classical",
    "Minimum Variance Long Only": "turnover_min_variance_long_only",
    "HRP": "turnover_hrp",
}
MODEL_SSPW_COLUMNS = {
    "Equal Weight": "sspw_equal_weight",
    "Minimum Variance Classical": "sspw_min_variance_classical",
    "Minimum Variance Long Only": "sspw_min_variance_long_only",
    "HRP": "sspw_hrp",
}
MODEL_COLORS = {
    "Equal Weight": "#4f7faa",
    "Minimum Variance Classical": "#bf4d24",
    "Minimum Variance Long Only": "#e07a2d",
    "HRP": "#24507a",
}
MODEL_SHORT_LABELS = {
    "Equal Weight": "Ligevægt",
    "Minimum Variance Classical": "MinVar klassisk",
    "Minimum Variance Long Only": "MinVar uden korte pos.",
    "HRP": "HRP",
}
MODEL_PLOT_LABELS = {
    "Equal Weight": "Ligevægt",
    "Minimum Variance Classical": "Klassisk minimumsvarians",
    "Minimum Variance Long Only": "Minimumsvarians uden korte positioner",
    "HRP": "HRP",
}
DANISH_MONTH_NAMES = {
    1: "januar",
    2: "februar",
    3: "marts",
    4: "april",
    5: "maj",
    6: "juni",
    7: "juli",
    8: "august",
    9: "september",
    10: "oktober",
    11: "november",
    12: "december",
}

# CBS palette
CBS_BLUE = "#4967AA"
CBS_MIDDLE_BLUE = "#6793D6"
CBS_LIGHT_BLUE = "#C9E0F5"
CBS_DARK_BLUE = "#242E70"
CBS_RED = "#E66A57"
CBS_CHESTNUT = "#6B1C26"
CBS_DARK_GREEN = "#114739"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, output_path: Path) -> None:
    fig.savefig(output_path, dpi=150)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)
    print(f"Gemt: {output_path}")


def _prepare_cov(cov: np.ndarray) -> Optional[np.ndarray]:
    """Validate and symmetrise a covariance matrix. Returns None if invalid."""
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        return None
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    if not np.isfinite(cov).all():
        return None
    return cov


def estimation_mode_label(cfg: "BacktestConfig") -> str:
    return "komplette observationer" if cfg.require_complete_case_months else "parvis"


def format_danish_month_start(period: pd.Period) -> str:
    timestamp = period.to_timestamp()
    return f"1. {DANISH_MONTH_NAMES[timestamp.month]} {timestamp.year}"


def effective_t_stats(history: pd.DataFrame, cfg: "BacktestConfig") -> dict[str, float]:
    if history.empty or "n_common_sample_months" not in history:
        return {
            "min": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "max": np.nan,
            "full_share": np.nan,
        }

    values = history["n_common_sample_months"].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {
            "min": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "max": np.nan,
            "full_share": np.nan,
        }

    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "full_share": float(np.mean(values == cfg.lookback_months)),
    }


def effective_t_caption(history: pd.DataFrame, cfg: "BacktestConfig") -> str:
    stats = effective_t_stats(history, cfg)
    if not np.isfinite(stats["mean"]):
        return f"Estimering: {estimation_mode_label(cfg)} | T_common ikke tilgængelig"

    return (
        f"Estimering: {estimation_mode_label(cfg)} | "
        f"T_common min/median/gennemsnit/max = "
        f"{stats['min']:.0f}/{stats['median']:.0f}/{stats['mean']:.1f}/{stats['max']:.0f} "
        f"ud af tilbagebliksvindue på {cfg.lookback_months}"
    )


def add_plot_subtitle(ax, text: str) -> None:
    ax.text(
        0.0,
        1.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#475569",
    )


def snapshot_estimation_caption(snapshot: "CorrelationSnapshot", cfg: "BacktestConfig") -> str:
    return (
        f"{estimation_mode_label(cfg)} kovarians | "
        f"T_common = {snapshot.n_common_sample_months}/{cfg.lookback_months} | "
        f"N = {snapshot.n_eligible_assets} | "
        f"T/N = {snapshot.diagnostics['time_to_assets_ratio']:.2f}"
    )


def marchenko_pastur_bounds(n_assets: int, n_obs: int) -> tuple[float, float, float]:
    q = float(n_assets / max(n_obs, 1))
    sqrt_q = float(np.sqrt(q))
    lambda_minus = float(max(0.0, (1.0 - sqrt_q) ** 2))
    lambda_plus = float((1.0 + sqrt_q) ** 2)
    return q, lambda_minus, lambda_plus


def marchenko_pastur_density(
    eigen_grid: np.ndarray,
    q: float,
    lambda_minus: float,
    lambda_plus: float,
) -> np.ndarray:
    density = np.zeros_like(eigen_grid, dtype=float)
    if q <= 0:
        return density

    mask = (eigen_grid >= lambda_minus) & (eigen_grid <= lambda_plus)
    safe_grid = np.clip(eigen_grid[mask], 1e-12, None)
    density[mask] = np.sqrt((lambda_plus - eigen_grid[mask]) * (eigen_grid[mask] - lambda_minus))
    density[mask] /= 2.0 * np.pi * q * safe_grid
    return density


# ---------------------------------------------------------------------------
# Config & data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestConfig:
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS
    top_n_universe: int = DEFAULT_TOP_N_UNIVERSE
    min_estimation_obs_ratio: float = 0.95
    require_complete_case_months: bool = True
    risk_free_annual: float = 0.0
    rolling_sharpe_window_months: int = DEFAULT_ROLLING_SHARPE_WINDOW_MONTHS
    start_month: Optional[str] = None
    end_month: Optional[str] = None

    @property
    def min_estimation_obs(self) -> int:
        return max(2, int(np.ceil(self.lookback_months * self.min_estimation_obs_ratio)))


@dataclass(frozen=True)
class EstimationSnapshot:
    universe_month: pd.Period
    n_universe_assets: int
    permnos: list[int]
    n_common_sample_months: int
    cov: np.ndarray
    correlation: pd.DataFrame


@dataclass(frozen=True)
class CorrelationSnapshot:
    evaluation_month: pd.Period
    universe_month: pd.Period
    n_universe_assets: int
    n_eligible_assets: int
    n_common_sample_months: int
    sorted_permnos: list[int]
    correlation: pd.DataFrame
    correlation_sorted: pd.DataFrame
    diagnostics: dict[str, float | int]


@dataclass(frozen=True)
class BacktestArtifacts:
    backtest_df: pd.DataFrame
    condition_history: pd.DataFrame
    turnover_history: pd.DataFrame
    turnover_summary: pd.DataFrame
    sspw_history: pd.DataFrame
    sspw_summary: pd.DataFrame
    weights_history: pd.DataFrame
    correlation_snapshot: Optional[CorrelationSnapshot]


class ModelTimeoutError(TimeoutError):
    """Raised when a model-specific computation exceeds its time budget."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_crsp_panels(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(col).strip() for col in df.columns]

    if "DLRET" not in df.columns:
        df["DLRET"] = np.nan

    for col in ["PERMNO", "PRC", "SHROUT", "EXCHCD", "SHRCD"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["PERMNO", "date"]).copy()
    df["PERMNO"] = df["PERMNO"].astype(int)
    df = df.sort_values(["PERMNO", "date"]).drop_duplicates(subset=["PERMNO", "date"], keep="first")

    ret = pd.to_numeric(df["RET"], errors="coerce")
    dlret = pd.to_numeric(df["DLRET"], errors="coerce")
    total_ret = (1.0 + ret).fillna(1.0) * (1.0 + dlret).fillna(1.0) - 1.0
    total_ret[ret.isna() & dlret.isna()] = np.nan

    df["PRC"] = pd.to_numeric(df["PRC"], errors="coerce").abs()
    df["SHROUT"] = pd.to_numeric(df["SHROUT"], errors="coerce")
    df["market_cap"] = df["PRC"] * df["SHROUT"]
    df["return"] = total_ret
    df = df[df["EXCHCD"].isin([1, 2, 3])]
    df = df[df["SHRCD"].isin([10, 11])]
    df["month"] = df["date"].dt.to_period("M")
    df = df.dropna(subset=["month"]).copy()

    returns_panel = (
        df.pivot_table(index="month", columns="PERMNO", values="return", aggfunc="first")
        .sort_index()
        .sort_index(axis=1)
    )
    market_cap_panel = (
        df.pivot_table(index="month", columns="PERMNO", values="market_cap", aggfunc="first")
        .sort_index()
        .sort_index(axis=1)
    )

    print(f"Indlaest CRSP-aktiedata fra: {path}")
    print(
        f"Aktiver: {returns_panel.shape[1]} | "
        f"Maneder: {returns_panel.shape[0]} | "
        f"Andel manglende afkast: {returns_panel.isna().mean().mean():.4f}"
    )

    return returns_panel, market_cap_panel


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def empty_summary() -> dict[str, float]:
    return {
        "Annual Return": np.nan,
        "Annual Volatility": np.nan,
        "Sharpe Ratio": np.nan,
        "Cumulative Return": np.nan,
        "Used Months": 0,
    }


def summarize_returns(monthly_returns: pd.Series, risk_free_annual: float) -> dict[str, float]:
    valid = monthly_returns.dropna()
    n_months = len(valid)
    if n_months == 0:
        return empty_summary()

    cumulative_return = float(np.prod(1.0 + valid.to_numpy()) - 1.0)
    annual_return = float((1.0 + cumulative_return) ** (12.0 / n_months) - 1.0)
    annual_volatility = float(valid.std(ddof=1) * np.sqrt(12.0)) if n_months > 1 else np.nan
    sharpe_ratio = (
        float((annual_return - risk_free_annual) / annual_volatility)
        if np.isfinite(annual_volatility) and annual_volatility > 0
        else np.nan
    )

    return {
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Cumulative Return": cumulative_return,
        "Used Months": n_months,
    }


def rolling_sharpe_ratio(
    monthly_returns: pd.Series,
    risk_free_annual: float,
    window_months: int,
) -> pd.Series:
    rolling_values = pd.Series(np.nan, index=monthly_returns.index, dtype=float)
    for end_idx in range(window_months - 1, len(monthly_returns)):
        window = monthly_returns.iloc[end_idx - window_months + 1 : end_idx + 1]
        rolling_values.iloc[end_idx] = summarize_returns(window, risk_free_annual)["Sharpe Ratio"]
    return rolling_values


def one_month_portfolio_return(return_row: pd.Series, weights: np.ndarray) -> float:
    values = return_row.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return np.nan
    return float(np.dot(weights, values))


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def selected_months(all_months: pd.Index, cfg: BacktestConfig) -> list[pd.Period]:
    start_month = pd.Period(cfg.start_month, freq="M") if cfg.start_month else all_months.min()
    end_month = pd.Period(cfg.end_month, freq="M") if cfg.end_month else all_months.max()
    return [month for month in all_months if start_month <= month <= end_month]


def build_estimation_snapshot(
    returns_panel: pd.DataFrame,
    market_cap_panel: pd.DataFrame,
    cfg: BacktestConfig,
    eval_month: pd.Period,
) -> Optional[EstimationSnapshot]:
    universe_month = eval_month - 1
    if universe_month not in market_cap_panel.index:
        return None

    ranked_market_caps = market_cap_panel.loc[universe_month].dropna().sort_values(ascending=False)
    if len(ranked_market_caps) < 2:
        return None

    est_months = pd.period_range(eval_month - cfg.lookback_months, periods=cfg.lookback_months, freq="M")
    if len(returns_panel.index.intersection(est_months)) < cfg.lookback_months:
        return None

    selected_permnos: list[int] = []
    batch_size = max(cfg.top_n_universe, 250)

    for start_idx in range(0, len(ranked_market_caps), batch_size):
        if len(selected_permnos) >= cfg.top_n_universe:
            break

        batch_permnos = ranked_market_caps.index[start_idx : start_idx + batch_size].tolist()
        returns_est_batch = returns_panel.reindex(index=est_months, columns=batch_permnos)
        valid_share = returns_est_batch.notna().mean(axis=0)

        batch_selected = [
            permno for permno in batch_permnos if valid_share.loc[permno] >= cfg.min_estimation_obs_ratio
        ]
        slots_left = cfg.top_n_universe - len(selected_permnos)
        selected_permnos.extend(batch_selected[:slots_left])

    if len(selected_permnos) < 2:
        return None

    returns_est = returns_panel.reindex(index=est_months, columns=selected_permnos)
    common_sample_months = int(returns_est.dropna(axis=0, how="any").shape[0])

    if cfg.require_complete_case_months:
        returns_est = returns_est.dropna(axis=0, how="any")

    if returns_est.shape[1] < 2:
        return None

    if cfg.require_complete_case_months:
        if returns_est.shape[0] < cfg.min_estimation_obs:
            return None
    else:
        min_asset_obs = int(returns_est.notna().sum(axis=0).min())
        if min_asset_obs < cfg.min_estimation_obs:
            return None

    if returns_est.empty:
        return None

    return EstimationSnapshot(
        universe_month=universe_month,
        n_universe_assets=len(ranked_market_caps),
        permnos=returns_est.columns.tolist(),
        n_common_sample_months=common_sample_months,
        cov=returns_est.cov().to_numpy(dtype=float),
        correlation=returns_est.corr(),
    )


def iter_estimation_snapshots(
    returns_panel: pd.DataFrame,
    market_cap_panel: pd.DataFrame,
    cfg: BacktestConfig,
):
    for eval_month in selected_months(returns_panel.index.sort_values(), cfg):
        snapshot = build_estimation_snapshot(returns_panel, market_cap_panel, cfg, eval_month)
        if snapshot is not None:
            yield eval_month, snapshot


# ---------------------------------------------------------------------------
# Portfolio weights
# ---------------------------------------------------------------------------

def covariance_rank(cov: np.ndarray) -> int:
    try:
        return int(np.linalg.matrix_rank(cov))
    except np.linalg.LinAlgError:
        return 0


def covariance_condition_number(cov: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(cov))
    except np.linalg.LinAlgError:
        return np.nan


def min_variance_long_only_weights(cov: np.ndarray) -> Optional[np.ndarray]:
    cov = _prepare_cov(cov)
    if cov is None:
        return None

    n_assets = cov.shape[0]
    result = minimize(
        fun=lambda w: float(w @ cov @ w),
        x0=np.full(n_assets, 1.0 / n_assets),
        method="SLSQP",
        jac=lambda w: 2.0 * (cov @ w),
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-12, "maxiter": DEFAULT_LONG_ONLY_MV_MAXITER},
    )
    if not result.success:
        return None

    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    if not np.isfinite(weights).all():
        return None
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
        return None

    return weights / weight_sum


def compute_long_only_mv_with_timeout(
    cov: np.ndarray,
    timeout_seconds: Optional[float],
) -> tuple[Optional[np.ndarray], bool, float]:
    start_time = time.monotonic()
    if timeout_seconds is None or timeout_seconds <= 0 or not hasattr(signal, "setitimer"):
        return min_variance_long_only_weights(cov), False, time.monotonic() - start_time

    def _timeout_handler(signum, frame):
        raise ModelTimeoutError

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        weights = min_variance_long_only_weights(cov)
        timed_out = False
    except ModelTimeoutError:
        weights = None
        timed_out = True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)

    return weights, timed_out, time.monotonic() - start_time


def min_variance_classical_weights(cov: np.ndarray) -> Optional[np.ndarray]:
    cov = _prepare_cov(cov)
    if cov is None:
        return None
    if covariance_rank(cov) < cov.shape[0]:
        return None

    try:
        raw_weights = np.linalg.solve(cov, np.ones(cov.shape[0], dtype=float))
    except np.linalg.LinAlgError:
        return None

    if not np.isfinite(raw_weights).all():
        return None
    weight_sum = raw_weights.sum()
    if not np.isfinite(weight_sum) or abs(weight_sum) <= 1e-12:
        return None

    return raw_weights / weight_sum


# ---------------------------------------------------------------------------
# HRP
# ---------------------------------------------------------------------------

def cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    sub_cov = cov[np.ix_(indices, indices)]
    inv_diag = 1.0 / np.clip(np.diag(sub_cov), 1e-12, None)
    ivp = inv_diag / inv_diag.sum()
    return float(ivp @ sub_cov @ ivp)


def get_quasi_diag(linkage_matrix: np.ndarray) -> list[int]:
    link = linkage_matrix.astype(int)
    sort_idx = pd.Series([link[-1, 0], link[-1, 1]], dtype=int)
    n_items = int(link[-1, 3])

    while sort_idx.max() >= n_items:
        sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
        cluster_items = sort_idx[sort_idx >= n_items]
        left_idx = cluster_items.index.to_numpy()
        right_idx = cluster_items.to_numpy() - n_items
        sort_idx.loc[left_idx] = link[right_idx, 0]
        sort_idx = pd.concat([sort_idx, pd.Series(link[right_idx, 1], index=left_idx + 1)])
        sort_idx = sort_idx.sort_index()
        sort_idx.index = range(sort_idx.shape[0])

    return sort_idx.astype(int).tolist()


def compute_single_linkage_matrix(correlation: pd.DataFrame) -> Optional[np.ndarray]:
    corr = correlation.to_numpy(dtype=float, copy=True)
    if corr.shape[0] < 2:
        return None

    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    condensed_dist = squareform(dist, checks=False)
    if not np.isfinite(condensed_dist).all():
        return None

    return linkage(condensed_dist, method="single", optimal_ordering=True)


def hrp_sorted_indices(correlation: pd.DataFrame) -> list[int]:
    linkage_matrix = compute_single_linkage_matrix(correlation)
    if linkage_matrix is None:
        return list(range(correlation.shape[0]))
    return get_quasi_diag(linkage_matrix)


def hrp_weights(cov: np.ndarray, sorted_idx: list[int]) -> np.ndarray:
    def recursive_bisection(indices: list[int]) -> dict[int, float]:
        if len(indices) == 1:
            return {indices[0]: 1.0}

        split = len(indices) // 2
        left = indices[:split]
        right = indices[split:]
        var_left = cluster_variance(cov, left)
        var_right = cluster_variance(cov, right)

        alpha = 0.5 if (var_left + var_right) <= 0 else 1.0 - var_left / (var_left + var_right)
        left_weights = recursive_bisection(left)
        right_weights = recursive_bisection(right)

        weights: dict[int, float] = {}
        for idx, value in left_weights.items():
            weights[idx] = value * alpha
        for idx, value in right_weights.items():
            weights[idx] = value * (1.0 - alpha)
        return weights

    raw_weights = recursive_bisection(sorted_idx)
    weights = np.zeros(cov.shape[0], dtype=float)
    for idx, value in raw_weights.items():
        weights[idx] = value

    weights = np.clip(weights, 0.0, None)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Backtest row & loop
# ---------------------------------------------------------------------------

def compute_model_weight_series(
    snapshot: EstimationSnapshot,
    long_only_mv_timeout_seconds: Optional[float] = None,
    long_only_mv_disabled: bool = False,
) -> tuple[dict[str, Optional[pd.Series]], dict[str, float | bool | int]]:
    sorted_idx = hrp_sorted_indices(snapshot.correlation)
    n_assets = len(snapshot.permnos)
    permno_index = pd.Index(snapshot.permnos)

    classical_mv_weights = min_variance_classical_weights(snapshot.cov)
    cov_rank_value = covariance_rank(snapshot.cov)
    cov_is_singular = cov_rank_value < snapshot.cov.shape[0]

    if long_only_mv_disabled:
        mv_weights, long_only_mv_timed_out = None, True
    else:
        mv_weights, long_only_mv_timed_out, _ = compute_long_only_mv_with_timeout(
            snapshot.cov, timeout_seconds=long_only_mv_timeout_seconds,
        )

    model_weights: dict[str, Optional[pd.Series]] = {
        "Equal Weight": pd.Series(np.full(n_assets, 1.0 / n_assets), index=permno_index, dtype=float),
        "Minimum Variance Classical": (
            pd.Series(classical_mv_weights, index=permno_index, dtype=float)
            if classical_mv_weights is not None else None
        ),
        "Minimum Variance Long Only": (
            pd.Series(mv_weights, index=permno_index, dtype=float)
            if mv_weights is not None else None
        ),
        "HRP": pd.Series(hrp_weights(snapshot.cov, sorted_idx), index=permno_index, dtype=float),
    }
    diagnostics: dict[str, float | bool | int] = {
        "n_assets": n_assets,
        "covariance_rank": cov_rank_value,
        "covariance_is_singular": cov_is_singular,
        "classical_mv_feasible": classical_mv_weights is not None,
        "long_only_mv_feasible": mv_weights is not None,
        "long_only_mv_timed_out": long_only_mv_timed_out,
    }
    return model_weights, diagnostics


def compute_turnover(
    previous_weights: Optional[pd.Series],
    asset_returns: Optional[pd.Series],
    current_weights: Optional[pd.Series],
) -> float:
    if previous_weights is None or asset_returns is None or current_weights is None:
        return np.nan

    previous_weights = previous_weights.astype(float)
    aligned_returns = asset_returns.reindex(previous_weights.index)
    if aligned_returns.isna().any():
        return np.nan

    gross_portfolio_return = 1.0 + one_month_portfolio_return(
        aligned_returns, previous_weights.to_numpy(dtype=float),
    )
    if not np.isfinite(gross_portfolio_return) or abs(gross_portfolio_return) <= 1e-12:
        return np.nan

    # Turnover follows the user's definition: sum_i |w_tilde_i - w_target_i|.
    drifted_weights = previous_weights * (1.0 + aligned_returns) / gross_portfolio_return
    if not np.isfinite(drifted_weights.to_numpy(dtype=float)).all():
        return np.nan

    all_assets = drifted_weights.index.union(current_weights.index)
    drifted_aligned = drifted_weights.reindex(all_assets, fill_value=0.0)
    current_aligned = current_weights.reindex(all_assets, fill_value=0.0).astype(float)
    return float(np.abs(drifted_aligned - current_aligned).sum())


def compute_sspw(weights: Optional[pd.Series]) -> float:
    if weights is None or len(weights) == 0:
        return np.nan

    weight_values = weights.to_numpy(dtype=float)
    if not np.isfinite(weight_values).all():
        return np.nan

    mean_weight = 1.0 / len(weight_values)
    return float(np.sum((weight_values - mean_weight) ** 2))


def build_backtest_row_components(
    eval_month: pd.Period,
    snapshot: EstimationSnapshot,
    returns_panel: pd.DataFrame,
    long_only_mv_timeout_seconds: Optional[float] = None,
    long_only_mv_disabled: bool = False,
) -> Optional[tuple[dict, dict[str, Optional[pd.Series]], pd.Series]]:
    eval_row = returns_panel.reindex(index=[eval_month], columns=snapshot.permnos).iloc[0]
    if eval_row.isna().any():
        return None

    model_weights, diagnostics = compute_model_weight_series(
        snapshot,
        long_only_mv_timeout_seconds=long_only_mv_timeout_seconds,
        long_only_mv_disabled=long_only_mv_disabled,
    )
    n_assets = int(diagnostics["n_assets"])

    row = {
        "evaluation_month": eval_month,
        "universe_month": snapshot.universe_month,
        "n_universe_assets": snapshot.n_universe_assets,
        "n_eligible_assets": n_assets,
        "n_common_sample_months": snapshot.n_common_sample_months,
        "covariance_rank": diagnostics["covariance_rank"],
        "covariance_is_singular": diagnostics["covariance_is_singular"],
        "classical_mv_feasible": diagnostics["classical_mv_feasible"],
        "long_only_mv_feasible": diagnostics["long_only_mv_feasible"],
        "long_only_mv_timed_out": diagnostics["long_only_mv_timed_out"],
    }
    for model_name, return_column in MODEL_COLUMNS.items():
        weights = model_weights[model_name]
        row[return_column] = (
            one_month_portfolio_return(eval_row, weights.to_numpy(dtype=float))
            if weights is not None else np.nan
        )

    return row, model_weights, eval_row


def build_backtest_row(
    eval_month: pd.Period,
    snapshot: EstimationSnapshot,
    returns_panel: pd.DataFrame,
    long_only_mv_timeout_seconds: Optional[float] = None,
    long_only_mv_disabled: bool = False,
) -> Optional[dict]:
    components = build_backtest_row_components(
        eval_month,
        snapshot,
        returns_panel,
        long_only_mv_timeout_seconds=long_only_mv_timeout_seconds,
        long_only_mv_disabled=long_only_mv_disabled,
    )
    return None if components is None else components[0]


def run_rolling_backtest(
    returns_panel: pd.DataFrame,
    market_cap_panel: pd.DataFrame,
    cfg: BacktestConfig,
    long_only_mv_timeout_seconds: Optional[float] = None,
) -> pd.DataFrame:
    rows = []
    remaining_long_only_mv_seconds = long_only_mv_timeout_seconds
    long_only_mv_disabled = False

    for eval_month, snapshot in iter_estimation_snapshots(returns_panel, market_cap_panel, cfg):
        timeout_for_this_month = None if long_only_mv_disabled else remaining_long_only_mv_seconds
        month_start = time.monotonic()
        row = build_backtest_row(
            eval_month, snapshot, returns_panel,
            long_only_mv_timeout_seconds=timeout_for_this_month,
            long_only_mv_disabled=long_only_mv_disabled,
        )
        elapsed = time.monotonic() - month_start
        if remaining_long_only_mv_seconds is not None and not long_only_mv_disabled:
            remaining_long_only_mv_seconds = max(0.0, remaining_long_only_mv_seconds - elapsed)
        if row is not None:
            if row["long_only_mv_timed_out"]:
                long_only_mv_disabled = True
            elif remaining_long_only_mv_seconds is not None and remaining_long_only_mv_seconds <= 0:
                long_only_mv_disabled = True
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation diagnostics
# ---------------------------------------------------------------------------

def build_correlation_diagnostics(snapshot: EstimationSnapshot) -> dict[str, float | int]:
    corr_values = snapshot.correlation.to_numpy(dtype=float)
    corr_off_diag = corr_values[~np.eye(corr_values.shape[0], dtype=bool)] if corr_values.shape[0] > 1 else np.array([])
    eigvals = np.linalg.eigvalsh(snapshot.cov) if snapshot.cov.size else np.array([])
    corr_eigvals = np.linalg.eigvalsh(corr_values) if corr_values.size else np.array([])
    q, lambda_minus, lambda_plus = marchenko_pastur_bounds(len(snapshot.permnos), snapshot.n_common_sample_months)

    return {
        "time_to_assets_ratio": float(snapshot.n_common_sample_months / max(len(snapshot.permnos), 1)),
        "median_offdiag_corr": float(np.median(corr_off_diag)) if corr_off_diag.size else 0.0,
        "min_eigenvalue": float(eigvals.min()) if eigvals.size else np.nan,
        "max_correlation_eigenvalue": float(corr_eigvals.max()) if corr_eigvals.size else np.nan,
        "mp_lambda_minus": lambda_minus,
        "mp_lambda_plus": lambda_plus,
        "eigenvalues_above_mp": int((corr_eigvals > lambda_plus).sum()) if corr_eigvals.size else 0,
        "mp_q_ratio": q,
        "condition_number": covariance_condition_number(snapshot.cov),
    }


def correlation_snapshot_from_estimation(
    eval_month: pd.Period,
    snapshot: EstimationSnapshot,
) -> CorrelationSnapshot:
    sorted_idx = hrp_sorted_indices(snapshot.correlation)
    sorted_permnos = [snapshot.permnos[idx] for idx in sorted_idx]
    correlation_sorted = snapshot.correlation.reindex(index=sorted_permnos, columns=sorted_permnos)

    return CorrelationSnapshot(
        evaluation_month=eval_month,
        universe_month=snapshot.universe_month,
        n_universe_assets=snapshot.n_universe_assets,
        n_eligible_assets=len(snapshot.permnos),
        n_common_sample_months=snapshot.n_common_sample_months,
        sorted_permnos=sorted_permnos,
        correlation=snapshot.correlation,
        correlation_sorted=correlation_sorted,
        diagnostics=build_correlation_diagnostics(snapshot),
    )


# ---------------------------------------------------------------------------
# Main backtest (with artifacts)
# ---------------------------------------------------------------------------

def build_backtest_artifacts(
    returns_panel: pd.DataFrame,
    market_cap_panel: pd.DataFrame,
    cfg: BacktestConfig,
) -> BacktestArtifacts:
    backtest_rows = []
    condition_rows = []
    weight_rows = []
    latest_snapshot: Optional[CorrelationSnapshot] = None
    previous_weights_by_model: dict[str, Optional[pd.Series]] = {
        model_name: None for model_name in MODEL_COLUMNS
    }
    previous_eval_row: Optional[pd.Series] = None

    for eval_month, snapshot in iter_estimation_snapshots(returns_panel, market_cap_panel, cfg):
        latest_snapshot = correlation_snapshot_from_estimation(eval_month, snapshot)
        condition_rows.append({
            "evaluation_month": eval_month,
            "universe_month": snapshot.universe_month,
            "n_universe_assets": snapshot.n_universe_assets,
            "n_eligible_assets": len(snapshot.permnos),
            "n_common_sample_months": snapshot.n_common_sample_months,
            "condition_number": covariance_condition_number(snapshot.cov),
        })

        components = build_backtest_row_components(eval_month, snapshot, returns_panel)
        if components is None:
            previous_weights_by_model = {model_name: None for model_name in MODEL_COLUMNS}
            previous_eval_row = None
            continue

        backtest_row, model_weights, eval_row = components
        for model_name, weights in model_weights.items():
            if weights is None:
                continue
            for permno, weight in weights.items():
                weight_rows.append({
                    "evaluation_month": eval_month,
                    "n_common_sample_months": snapshot.n_common_sample_months,
                    "model": model_name,
                    "permno": int(permno),
                    "weight": float(weight),
                })
        for model_name, turnover_column in MODEL_TURNOVER_COLUMNS.items():
            backtest_row[turnover_column] = compute_turnover(
                previous_weights_by_model[model_name],
                previous_eval_row,
                model_weights[model_name],
            )
        for model_name, sspw_column in MODEL_SSPW_COLUMNS.items():
            backtest_row[sspw_column] = compute_sspw(model_weights[model_name])
        backtest_rows.append(backtest_row)
        previous_weights_by_model = {
            model_name: (weights.copy() if weights is not None else None)
            for model_name, weights in model_weights.items()
        }
        previous_eval_row = eval_row.copy()

    backtest_df = pd.DataFrame(backtest_rows)
    turnover_history = (
        pd.DataFrame({
            "evaluation_month": backtest_df["evaluation_month"],
            "n_common_sample_months": backtest_df["n_common_sample_months"],
        })
        if not backtest_df.empty else
        pd.DataFrame(columns=["evaluation_month", "n_common_sample_months"])
    )
    for model_name, turnover_column in MODEL_TURNOVER_COLUMNS.items():
        turnover_history[model_name] = backtest_df[turnover_column] if turnover_column in backtest_df else pd.Series(dtype=float)

    turnover_summary = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Average Monthly Turnover": float(backtest_df[turnover_column].dropna().mean()) if turnover_column in backtest_df and backtest_df[turnover_column].notna().any() else np.nan,
                "Total Turnover": float(backtest_df[turnover_column].dropna().sum()) if turnover_column in backtest_df and backtest_df[turnover_column].notna().any() else np.nan,
                "Turnover Months": int(backtest_df[turnover_column].notna().sum()) if turnover_column in backtest_df else 0,
            }
            for model_name, turnover_column in MODEL_TURNOVER_COLUMNS.items()
        ]
    )
    sspw_history = (
        pd.DataFrame({
            "evaluation_month": backtest_df["evaluation_month"],
            "n_common_sample_months": backtest_df["n_common_sample_months"],
        })
        if not backtest_df.empty else
        pd.DataFrame(columns=["evaluation_month", "n_common_sample_months"])
    )
    for model_name, sspw_column in MODEL_SSPW_COLUMNS.items():
        sspw_history[model_name] = backtest_df[sspw_column] if sspw_column in backtest_df else pd.Series(dtype=float)

    sspw_summary = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Average SSPW": float(backtest_df[sspw_column].dropna().mean()) if sspw_column in backtest_df and backtest_df[sspw_column].notna().any() else np.nan,
                "SSPW Months": int(backtest_df[sspw_column].notna().sum()) if sspw_column in backtest_df else 0,
            }
            for model_name, sspw_column in MODEL_SSPW_COLUMNS.items()
        ]
    )
    weights_history = pd.DataFrame(weight_rows)

    return BacktestArtifacts(
        backtest_df=backtest_df,
        condition_history=pd.DataFrame(condition_rows),
        turnover_history=turnover_history,
        turnover_summary=turnover_summary,
        sspw_history=sspw_history,
        sspw_summary=sspw_summary,
        weights_history=weights_history,
        correlation_snapshot=latest_snapshot,
    )


# ---------------------------------------------------------------------------
# Summary tables & reporting
# ---------------------------------------------------------------------------

def build_summary_tables(backtest_df: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance_df = pd.DataFrame(
        [
            {"Model": model_name, **summarize_returns(backtest_df[column], cfg.risk_free_annual)}
            for model_name, column in MODEL_COLUMNS.items()
        ]
    )

    perf = performance_df.set_index("Model")
    diff_rows = []
    for benchmark_name in MODEL_COLUMNS:
        if benchmark_name == "HRP":
            continue
        diff_rows.append({
            "Model": "HRP",
            "Vs Model": benchmark_name,
            "Delta Annual Return (Model - benchmark)": perf.loc["HRP", "Annual Return"] - perf.loc[benchmark_name, "Annual Return"],
            "Delta Annual Volatility (Model - benchmark)": perf.loc["HRP", "Annual Volatility"] - perf.loc[benchmark_name, "Annual Volatility"],
            "Delta Sharpe (Model - benchmark)": perf.loc["HRP", "Sharpe Ratio"] - perf.loc[benchmark_name, "Sharpe Ratio"],
            "Delta Cumulative Return (Model - benchmark)": perf.loc["HRP", "Cumulative Return"] - perf.loc[benchmark_name, "Cumulative Return"],
        })

    return performance_df, pd.DataFrame(diff_rows)


def _valid_value_months(history: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if history.empty or value_column not in history or "evaluation_month" not in history:
        return pd.DataFrame(columns=["evaluation_month", value_column])

    values = history[["evaluation_month", value_column]].copy()
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_column])


def _extreme_value_and_month(
    history: pd.DataFrame,
    value_column: str,
    use_max: bool,
) -> tuple[float, str]:
    values = _valid_value_months(history, value_column)
    if values.empty:
        return np.nan, ""

    idx = values[value_column].idxmax() if use_max else values[value_column].idxmin()
    row = values.loc[idx]
    return float(row[value_column]), str(row["evaluation_month"])


def _series_mean(history: pd.DataFrame, value_column: str) -> float:
    values = _valid_value_months(history, value_column)
    return float(values[value_column].mean()) if not values.empty else np.nan


def _series_median(history: pd.DataFrame, value_column: str) -> float:
    values = _valid_value_months(history, value_column)
    return float(values[value_column].median()) if not values.empty else np.nan


def _max_drawdown(monthly_returns: pd.Series, evaluation_months: pd.Series) -> tuple[float, str]:
    valid = pd.DataFrame({
        "evaluation_month": evaluation_months,
        "return": pd.to_numeric(monthly_returns, errors="coerce"),
    }).dropna(subset=["return"])
    if valid.empty:
        return np.nan, ""

    wealth = (1.0 + valid["return"]).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    if drawdown.empty:
        return np.nan, ""

    idx = drawdown.idxmin()
    return float(drawdown.loc[idx]), str(valid.loc[idx, "evaluation_month"])


def _rolling_annualized_volatility(monthly_returns: pd.Series, window_months: int) -> pd.Series:
    return pd.to_numeric(monthly_returns, errors="coerce").rolling(window_months).std(ddof=1) * np.sqrt(12.0)


def export_comprehensive_results_csv(
    cfg: BacktestConfig,
    backtest_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    sspw_summary: pd.DataFrame,
    condition_history: pd.DataFrame,
) -> None:
    """Save one broad CSV with the main model results and timing of extrema."""
    if backtest_df.empty:
        print("Springer samlet resultat-CSV over, fordi backtesten er tom.")
        return

    performance_by_model = performance_df.set_index("Model")
    turnover_by_model = turnover_summary.set_index("Model") if not turnover_summary.empty else pd.DataFrame()
    sspw_by_model = sspw_summary.set_index("Model") if not sspw_summary.empty else pd.DataFrame()

    common_t = pd.to_numeric(backtest_df["n_common_sample_months"], errors="coerce")
    common_context = {
        "Første evalueringsmåned": str(backtest_df["evaluation_month"].iloc[0]),
        "Sidste evalueringsmåned": str(backtest_df["evaluation_month"].iloc[-1]),
        "Lookback-måneder": cfg.lookback_months,
        "Top N-univers": cfg.top_n_universe,
        "Kovariansestimering": estimation_mode_label(cfg),
        "Gennemsnitligt antal market cap-kandidater": float(backtest_df["n_universe_assets"].mean()),
        "Gennemsnitligt antal aktiver i porteføljen": float(backtest_df["n_eligible_assets"].mean()),
        "T_common minimum": float(common_t.min()),
        "T_common median": float(common_t.median()),
        "T_common gennemsnit": float(common_t.mean()),
        "T_common maksimum": float(common_t.max()),
        "Andel måneder med fuldt tilbagebliksvindue": float(common_t.eq(cfg.lookback_months).mean()),
    }

    condition_context = {}
    if not condition_history.empty and "condition_number" in condition_history:
        condition_values = _valid_value_months(condition_history, "condition_number")
        max_condition, max_condition_month = _extreme_value_and_month(condition_history, "condition_number", use_max=True)
        min_condition, min_condition_month = _extreme_value_and_month(condition_history, "condition_number", use_max=False)
        condition_context = {
            "Konditionstal minimum": min_condition,
            "Måned for laveste konditionstal": min_condition_month,
            "Konditionstal gennemsnit": float(condition_values["condition_number"].mean()) if not condition_values.empty else np.nan,
            "Konditionstal median": float(condition_values["condition_number"].median()) if not condition_values.empty else np.nan,
            "Konditionstal maksimum": max_condition,
            "Måned for højeste konditionstal": max_condition_month,
        }

    rows = []
    hrp_performance = performance_by_model.loc["HRP"] if "HRP" in performance_by_model.index else pd.Series(dtype=float)
    for model_name, return_column in MODEL_COLUMNS.items():
        returns = pd.to_numeric(backtest_df[return_column], errors="coerce")
        used_returns = returns.dropna()
        best_return, best_month = _extreme_value_and_month(backtest_df, return_column, use_max=True)
        worst_return, worst_month = _extreme_value_and_month(backtest_df, return_column, use_max=False)
        max_drawdown, max_drawdown_month = _max_drawdown(returns, backtest_df["evaluation_month"])

        rolling_sharpe_col = "rolling_sharpe"
        rolling_sharpe_history = pd.DataFrame({
            "evaluation_month": backtest_df["evaluation_month"],
            rolling_sharpe_col: rolling_sharpe_ratio(
                returns,
                risk_free_annual=cfg.risk_free_annual,
                window_months=cfg.rolling_sharpe_window_months,
            ),
        })
        max_rolling_sharpe, max_rolling_sharpe_month = _extreme_value_and_month(
            rolling_sharpe_history, rolling_sharpe_col, use_max=True,
        )
        min_rolling_sharpe, min_rolling_sharpe_month = _extreme_value_and_month(
            rolling_sharpe_history, rolling_sharpe_col, use_max=False,
        )

        rolling_vol_col = "rolling_volatility"
        rolling_vol_history = pd.DataFrame({
            "evaluation_month": backtest_df["evaluation_month"],
            rolling_vol_col: _rolling_annualized_volatility(returns, window_months=12),
        })
        max_rolling_vol, max_rolling_vol_month = _extreme_value_and_month(
            rolling_vol_history, rolling_vol_col, use_max=True,
        )
        min_rolling_vol, min_rolling_vol_month = _extreme_value_and_month(
            rolling_vol_history, rolling_vol_col, use_max=False,
        )

        turnover_column = MODEL_TURNOVER_COLUMNS[model_name]
        max_turnover, max_turnover_month = _extreme_value_and_month(backtest_df, turnover_column, use_max=True)
        min_turnover, min_turnover_month = _extreme_value_and_month(backtest_df, turnover_column, use_max=False)

        sspw_column = MODEL_SSPW_COLUMNS[model_name]
        max_sspw, max_sspw_month = _extreme_value_and_month(backtest_df, sspw_column, use_max=True)
        min_sspw, min_sspw_month = _extreme_value_and_month(backtest_df, sspw_column, use_max=False)

        row = {
            "Model": model_name,
            "Modelnavn": MODEL_PLOT_LABELS.get(model_name, model_name),
            **common_context,
            **condition_context,
            "Antal brugte måneder": int(performance_by_model.loc[model_name, "Used Months"]),
            "Kumulativt afkast": float(performance_by_model.loc[model_name, "Cumulative Return"]),
            "Annualiseret afkast": float(performance_by_model.loc[model_name, "Annual Return"]),
            "Annualiseret volatilitet": float(performance_by_model.loc[model_name, "Annual Volatility"]),
            "Sharpe-ratio": float(performance_by_model.loc[model_name, "Sharpe Ratio"]),
            "HRP minus modellen: annualiseret afkast": (
                float(hrp_performance["Annual Return"] - performance_by_model.loc[model_name, "Annual Return"])
                if not hrp_performance.empty else np.nan
            ),
            "HRP minus modellen: annualiseret volatilitet": (
                float(hrp_performance["Annual Volatility"] - performance_by_model.loc[model_name, "Annual Volatility"])
                if not hrp_performance.empty else np.nan
            ),
            "HRP minus modellen: Sharpe-ratio": (
                float(hrp_performance["Sharpe Ratio"] - performance_by_model.loc[model_name, "Sharpe Ratio"])
                if not hrp_performance.empty else np.nan
            ),
            "HRP minus modellen: kumulativt afkast": (
                float(hrp_performance["Cumulative Return"] - performance_by_model.loc[model_name, "Cumulative Return"])
                if not hrp_performance.empty else np.nan
            ),
            "Månedligt gennemsnitsafkast": float(used_returns.mean()) if not used_returns.empty else np.nan,
            "Månedligt medianafkast": float(used_returns.median()) if not used_returns.empty else np.nan,
            "Månedlig standardafvigelse": float(used_returns.std(ddof=1)) if len(used_returns) > 1 else np.nan,
            "Månedligt minimumsafkast": worst_return,
            "Måned for minimumsafkast": worst_month,
            "Månedligt maksimumsafkast": best_return,
            "Måned for maksimumsafkast": best_month,
            "Andel positive måneder": float((used_returns > 0).mean()) if not used_returns.empty else np.nan,
            "Maksimalt drawdown": max_drawdown,
            "Måned for maksimalt drawdown": max_drawdown_month,
            f"Rullende {cfg.rolling_sharpe_window_months}-måneders Sharpe gennemsnit": _series_mean(rolling_sharpe_history, rolling_sharpe_col),
            f"Rullende {cfg.rolling_sharpe_window_months}-måneders Sharpe minimum": min_rolling_sharpe,
            f"Måned for laveste rullende {cfg.rolling_sharpe_window_months}-måneders Sharpe": min_rolling_sharpe_month,
            f"Rullende {cfg.rolling_sharpe_window_months}-måneders Sharpe maksimum": max_rolling_sharpe,
            f"Måned for højeste rullende {cfg.rolling_sharpe_window_months}-måneders Sharpe": max_rolling_sharpe_month,
            "Rullende 12-måneders volatilitet gennemsnit": _series_mean(rolling_vol_history, rolling_vol_col),
            "Rullende 12-måneders volatilitet minimum": min_rolling_vol,
            "Måned for laveste rullende 12-måneders volatilitet": min_rolling_vol_month,
            "Rullende 12-måneders volatilitet maksimum": max_rolling_vol,
            "Måned for højeste rullende 12-måneders volatilitet": max_rolling_vol_month,
            "Gennemsnitlig månedlig omsætning": _series_mean(backtest_df, turnover_column),
            "Median månedlig omsætning": _series_median(backtest_df, turnover_column),
            "Minimum månedlig omsætning": min_turnover,
            "Måned for minimum omsætning": min_turnover_month,
            "Maksimum månedlig omsætning": max_turnover,
            "Måned for maksimum omsætning": max_turnover_month,
            "Samlet omsætning": (
                float(turnover_by_model.loc[model_name, "Total Turnover"])
                if not turnover_by_model.empty and model_name in turnover_by_model.index else np.nan
            ),
            "Gennemsnitlig SSPW": _series_mean(backtest_df, sspw_column),
            "Median SSPW": _series_median(backtest_df, sspw_column),
            "Minimum SSPW": min_sspw,
            "Måned for minimum SSPW": min_sspw_month,
            "Maksimum SSPW": max_sspw,
            "Måned for maksimum SSPW": max_sspw_month,
            "SSPW-måneder": (
                int(sspw_by_model.loc[model_name, "SSPW Months"])
                if not sspw_by_model.empty and model_name in sspw_by_model.index else 0
            ),
        }

        if model_name == "Minimum Variance Classical" and "classical_mv_feasible" in backtest_df:
            row["Mulige måneder for klassisk minimumsvarians"] = int(backtest_df["classical_mv_feasible"].sum())
        if model_name == "Minimum Variance Long Only" and "long_only_mv_feasible" in backtest_df:
            row["Mulige måneder for minimumsvarians uden korte positioner"] = int(backtest_df["long_only_mv_feasible"].sum())
        rows.append(row)

    results = pd.DataFrame(rows)
    output_path = Path(__file__).with_name("samlede_modelresultater.csv")
    results.to_csv(output_path, index=False)
    print(f"Gemt: {output_path}")


def print_report(
    cfg: BacktestConfig,
    backtest_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    sspw_summary: pd.DataFrame,
) -> None:
    print("Fuld out-of-sample rullende backtest")
    print(
        f"Lookback: {cfg.lookback_months} mdr | "
        f"Portefoljeunivers: top {cfg.top_n_universe} aktiver, der bestar filtrene | "
        f"Min obs i estimering: {cfg.min_estimation_obs} | "
        f"Kovariansestimering: {estimation_mode_label(cfg)}"
    )

    if backtest_df.empty:
        print("Ingen gyldige out-of-sample maneder kunne beregnes.")
        return

    print(
        f"Maneder med out-of-sample afkast: {len(backtest_df)} "
        f"({backtest_df['evaluation_month'].iloc[0]} til {backtest_df['evaluation_month'].iloc[-1]})"
    )
    print(f"Gns. antal market cap-kandidater i universmaneden: {backtest_df['n_universe_assets'].mean():.1f}")
    print(f"Gns. antal aktiver brugt i portefoljen: {backtest_df['n_eligible_assets'].mean():.1f}")
    print(
        f"T_common i estimeringen: min/median/gennemsnit/max = "
        f"{backtest_df['n_common_sample_months'].min():.0f}/"
        f"{backtest_df['n_common_sample_months'].median():.0f}/"
        f"{backtest_df['n_common_sample_months'].mean():.1f}/"
        f"{backtest_df['n_common_sample_months'].max():.0f} "
        f"ud af lookback {cfg.lookback_months}"
    )
    print(
        f"Andel maaneder med fuldt estimeringsvindue: "
        f"{(backtest_df['n_common_sample_months'].eq(cfg.lookback_months).mean() * 100):.1f}%"
    )
    if "classical_mv_feasible" in backtest_df.columns:
        print(f"Klassisk MV kunne beregnes i {int(backtest_df['classical_mv_feasible'].sum())} af {len(backtest_df)} maneder.")
    if "long_only_mv_feasible" in backtest_df.columns:
        print(f"Long-only MV kunne beregnes i {int(backtest_df['long_only_mv_feasible'].sum())} af {len(backtest_df)} maneder.")
    if "covariance_is_singular" in backtest_df.columns:
        singular_months = int(backtest_df["covariance_is_singular"].sum())
        singular_and_classical_mv = int((backtest_df["covariance_is_singular"] & backtest_df["classical_mv_feasible"]).sum())
        singular_and_long_only_mv = int((backtest_df["covariance_is_singular"] & backtest_df["long_only_mv_feasible"]).sum())
        print(f"Singulaer kovariansmatrix i {singular_months} af {len(backtest_df)} maneder.")
        print(
            "Klassisk MV bruger den lukkede inverse-formel og kan derfor ikke bruges ved singulaer covariance "
            f"({singular_and_classical_mv} måneder her)."
        )
        print(
            "Long-only MV loeses som et constrained simplex-problem og kan derfor godt eksistere, "
            f"selv når kovariansmatricen er singulaer ({singular_and_long_only_mv} måneder her)."
        )
    print()

    print("Performance (out-of-sample):")
    print(performance_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()

    print("Forskel mellem modeller")
    print(diff_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()

    print("Turnover-summary")
    print(turnover_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()

    print("SSPW-summary")
    print(sspw_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()

    print("Seneste 5 out-of-sample manedsafkast:")
    preview_cols = [
        "evaluation_month",
        "n_eligible_assets",
        "ret_equal_weight",
        "ret_min_variance_classical",
        "ret_min_variance_long_only",
        "ret_hrp",
    ]
    print(backtest_df[preview_cols].tail(5).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def add_evaluation_dates(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    history["evaluation_date"] = pd.PeriodIndex(history["evaluation_month"], freq="M").to_timestamp()
    return history


def select_weight_snapshot_month(
    weights_history: pd.DataFrame,
    requested_month: str,
) -> tuple[Optional[pd.Period], bool]:
    if weights_history.empty or "evaluation_month" not in weights_history:
        return None, False

    available_periods = pd.PeriodIndex(weights_history["evaluation_month"].astype(str).unique(), freq="M").sort_values()
    if len(available_periods) == 0:
        return None, False

    requested_period = pd.Period(requested_month, freq="M")
    if requested_period in available_periods:
        return requested_period, False

    distance = np.abs(available_periods.astype("int64") - requested_period.ordinal)
    nearest_idx = int(np.argmin(distance))
    return available_periods[nearest_idx], True


def plot_effective_estimation_window(history: pd.DataFrame, cfg: BacktestConfig) -> None:
    if history.empty or "n_common_sample_months" not in history:
        print("Ingen T_common-historik at gemme.")
        return

    plot_history = add_evaluation_dates(history)
    output_path = Path(__file__).with_name("effective_estimation_window_over_time.png")
    history_path = Path(__file__).with_name("effective_estimation_window_over_time.csv")
    plot_history[["evaluation_month", "n_common_sample_months"]].to_csv(history_path, index=False)
    print(f"Gemt: {history_path}")

    if plt is None:
        print("Springer T_common-plot over, fordi matplotlib ikke er installeret.")
        return

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(
        plot_history["evaluation_date"],
        plot_history["n_common_sample_months"],
        color="#9ecae1",
        alpha=0.45,
        label="Faktisk T_common",
    )
    ax.plot(
        plot_history["evaluation_date"],
        plot_history["n_common_sample_months"],
        color="#24507a",
        linewidth=2.0,
    )
    ax.axhline(
        cfg.lookback_months,
        color="#bf4d24",
        linewidth=1.4,
        linestyle="--",
        label=f"Tilbagebliksvindue = {cfg.lookback_months}",
    )
    ax.set_title("Faktisk antal måneder brugt i kovariansestimeringen", pad=18)
    add_plot_subtitle(ax, effective_t_caption(history, cfg))
    ax.set_xlabel("Evalueringsmåned")
    ax.set_ylabel("T_common")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_fig(fig, output_path)


def plot_average_top_weight_distribution(
    weights_history: pd.DataFrame,
    cfg: BacktestConfig,
    top_n: int = DEFAULT_WEIGHT_DISTRIBUTION_TOP_N,
) -> None:
    if weights_history.empty:
        print("Ingen vaegthistorik at gemme.")
        return

    meta_history = weights_history[["evaluation_month", "n_common_sample_months"]].drop_duplicates()
    ranked = weights_history.copy()
    ranked["evaluation_month"] = ranked["evaluation_month"].astype(str)
    ranked = ranked.sort_values(["model", "evaluation_month", "weight"], ascending=[True, True, False])
    ranked["rank"] = ranked.groupby(["model", "evaluation_month"]).cumcount() + 1
    ranked = ranked[ranked["rank"] <= top_n].copy()
    if ranked.empty:
        print("Ingen rangerede vaegte at plotte.")
        return

    rank_index = pd.MultiIndex.from_product([list(MODEL_COLUMNS.keys()), range(1, top_n + 1)], names=["model", "rank"])
    rank_summary = (
        ranked.groupby(["model", "rank"], as_index=True)["weight"].mean()
        .reindex(rank_index)
        .reset_index()
    )

    csv_path = Path(__file__).with_name("average_top_weight_distribution.csv")
    rank_summary.to_csv(csv_path, index=False)
    print(f"Gemt: {csv_path}")

    if plt is None:
        print("Springer bubble-plot over, fordi matplotlib ikke er installeret.")
        return

    plot_df = rank_summary.dropna(subset=["weight"]).copy()
    if plot_df.empty:
        print("Springer bubble-plot over, fordi alle gennemsnitsvaegte er tomme.")
        return

    max_weight = float(plot_df["weight"].max())
    if not np.isfinite(max_weight) or max_weight <= 0:
        print("Springer bubble-plot over, fordi gennemsnitsvaegtene ikke er positive.")
        return

    model_order = list(MODEL_COLUMNS.keys())
    y_positions = {model: len(model_order) - idx for idx, model in enumerate(model_order)}
    plot_df["y"] = plot_df["model"].map(y_positions)
    plot_df["size"] = 220.0 + 2200.0 * (plot_df["weight"] / max_weight)

    output_path = Path(__file__).with_name("average_top_weight_distribution.png")
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    for model_name in model_order:
        model_slice = plot_df[plot_df["model"] == model_name]
        if model_slice.empty:
            continue
        ax.scatter(
            model_slice["rank"],
            model_slice["y"],
            s=model_slice["size"],
            color=MODEL_COLORS[model_name],
            alpha=0.72,
            edgecolors="white",
            linewidths=1.0,
        )

    ax.set_title(f"Gennemsnitlig vægtfordeling for de {top_n} højest vægtede aktiver", pad=18)
    add_plot_subtitle(ax, effective_t_caption(meta_history, cfg))
    ax.set_xlabel("Rang efter vægt i porteføljen")
    ax.set_ylabel("Porteføljemetode")
    ax.set_xlim(0.5, top_n + 0.5)
    ax.set_xticks(range(1, top_n + 1))
    ax.set_yticks([y_positions[model] for model in model_order])
    ax.set_yticklabels([MODEL_SHORT_LABELS.get(model, model) for model in model_order])
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", alpha=0.10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_fig(fig, output_path)


def plot_allocation_method_returns(backtest_df: pd.DataFrame, cfg: BacktestConfig) -> None:
    if backtest_df.empty:
        print("Ingen backtest-historik at plotte.")
        return

    history = add_evaluation_dates(backtest_df)

    cumulative_returns = pd.DataFrame({
        "evaluation_date": history["evaluation_date"],
        "n_common_sample_months": history["n_common_sample_months"],
    })
    for label, column in MODEL_COLUMNS.items():
        cumulative_returns[label] = (1.0 + history[column].fillna(0.0)).cumprod()

    output_path = Path(__file__).with_name("allocation_method_cumulative_returns.png")
    cumulative_path = Path(__file__).with_name("allocation_method_cumulative_returns.csv")
    cumulative_returns.to_csv(cumulative_path, index=False)
    print(f"Gemt: {cumulative_path}")

    if plt is None:
        print("Springer afkastplot over, fordi matplotlib ikke er installeret.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    n_total = len(history)
    for label, column in MODEL_COLUMNS.items():
        n_feasible = int(history[column].notna().sum())
        display_label = MODEL_PLOT_LABELS.get(label, label)
        legend_label = f"{display_label} ({n_feasible}/{n_total} mdr.)" if n_feasible < n_total else display_label
        ax.plot(
            cumulative_returns["evaluation_date"],
            cumulative_returns[label],
            linewidth=1.8,
            label=legend_label,
            color=MODEL_COLORS[label],
        )

    ax.set_title("Kumulative afkast efter allokeringsmetode", pad=18)
    add_plot_subtitle(ax, effective_t_caption(backtest_df, cfg))
    ax.set_xlabel("Evalueringsmåned")
    ax.set_ylabel("Værdi af 1 investeret")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_weight_snapshot_pies(
    weights_history: pd.DataFrame,
    cfg: BacktestConfig,
    snapshot_month: str = DEFAULT_WEIGHT_SNAPSHOT_MONTH,
) -> None:
    if weights_history.empty:
        print("Ingen vaegthistorik at gemme.")
        return

    meta_history = weights_history[["evaluation_month", "n_common_sample_months"]].drop_duplicates()
    selected_period, used_fallback = select_weight_snapshot_month(weights_history, snapshot_month)
    if selected_period is None:
        print("Ingen gyldig snapshot-maaned til pie charts.")
        return

    snapshot = weights_history[weights_history["evaluation_month"].astype(str) == str(selected_period)].copy()
    if snapshot.empty:
        print("Ingen vaegte fundet for snapshot-maaneden.")
        return

    snapshot = snapshot.sort_values(["model", "weight"], ascending=[True, False])
    csv_path = Path(__file__).with_name(f"weight_snapshot_{selected_period}.csv")
    snapshot.to_csv(csv_path, index=False)
    print(f"Gemt: {csv_path}")

    if plt is None:
        print("Springer pie charts over, fordi matplotlib ikke er installeret.")
        return

    model_order = list(MODEL_COLUMNS.keys())
    n_models = len(model_order)
    ncols = min(2, n_models)
    nrows = int(np.ceil(n_models / ncols))
    output_path = Path(__file__).with_name(f"weight_snapshot_{selected_period}.png")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11.5, 5.3 * nrows), subplot_kw={"aspect": "equal"})
    axes = np.atleast_1d(axes).ravel()

    color_pool = plt.cm.tab20(np.linspace(0.0, 1.0, 20))
    title = f"Vægtfordelinger den {format_danish_month_start(selected_period)}"
    if used_fallback:
        title += f" (nærmeste tilgængelige måned til {snapshot_month})"
    fig.suptitle(title, fontsize=15, y=0.98)

    for ax, model_name in zip(axes, model_order):
        model_slice = snapshot[snapshot["model"] == model_name].copy()
        model_slice = model_slice[model_slice["weight"] > 0].sort_values("weight", ascending=False)
        if model_slice.empty:
            ax.set_axis_off()
            continue
        colors = [color_pool[idx % len(color_pool)] for idx in range(len(model_slice))]
        ax.pie(
            model_slice["weight"],
            startangle=90,
            colors=colors,
            wedgeprops={"linewidth": 0.15, "edgecolor": "white"},
        )
        ax.set_title(MODEL_PLOT_LABELS.get(model_name, model_name), fontsize=12, pad=10)

    for ax in axes[n_models:]:
        ax.set_axis_off()

    fig.text(
        0.02,
        0.02,
        effective_t_caption(meta_history, cfg),
        ha="left",
        va="bottom",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save_fig(fig, output_path)


def plot_rolling_sharpe_history(backtest_df: pd.DataFrame, cfg: BacktestConfig) -> None:
    if backtest_df.empty:
        print("Ingen Sharpe-historik at plotte.")
        return

    history = add_evaluation_dates(backtest_df)

    sharpe_history = pd.DataFrame({
        "evaluation_date": history["evaluation_date"],
        "n_common_sample_months": history["n_common_sample_months"],
    })
    for label, column in MODEL_COLUMNS.items():
        sharpe_history[label] = rolling_sharpe_ratio(
            history[column],
            risk_free_annual=cfg.risk_free_annual,
            window_months=cfg.rolling_sharpe_window_months,
        )

    output_path = Path(__file__).with_name("rolling_sharpe_ratio_over_time.png")
    history_path = Path(__file__).with_name("rolling_sharpe_ratio_over_time.csv")
    sharpe_history.to_csv(history_path, index=False)
    print(f"Gemt: {history_path}")

    if plt is None:
        print("Springer Sharpe-plot over, fordi matplotlib ikke er installeret.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for label in MODEL_COLUMNS:
        ax.plot(
            sharpe_history["evaluation_date"],
            sharpe_history[label],
            linewidth=1.8,
            label=MODEL_PLOT_LABELS.get(label, label),
            color=MODEL_COLORS[label],
        )

    ax.axhline(0.0, color="#4c4c4c", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_title(f"Rullende {cfg.rolling_sharpe_window_months}-måneders Sharpe-ratio", pad=18)
    add_plot_subtitle(ax, effective_t_caption(backtest_df, cfg))
    ax.set_xlabel("Evalueringsmåned")
    ax.set_ylabel("Sharpe-ratio")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_rolling_annualized_volatility(backtest_df: pd.DataFrame, cfg: BacktestConfig, window_months: int = 12) -> None:
    if backtest_df.empty:
        print("Ingen volatilitetshistorik at plotte.")
        return

    history = add_evaluation_dates(backtest_df)
    volatility_history = pd.DataFrame({
        "evaluation_date": history["evaluation_date"],
        "n_common_sample_months": history["n_common_sample_months"],
    })
    for label, column in MODEL_COLUMNS.items():
        volatility_history[label] = history[column].rolling(window_months).std(ddof=1) * np.sqrt(12.0)

    history_path = Path(__file__).with_name("rolling_annualized_volatility_over_time.csv")
    volatility_history.to_csv(history_path, index=False)
    print(f"Gemt: {history_path}")

    if plt is None:
        print("Springer volatilitetsplot over, fordi matplotlib ikke er installeret.")
        return

    output_path = Path(__file__).with_name("rolling_annualized_volatility_over_time.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in MODEL_COLUMNS:
        ax.plot(
            volatility_history["evaluation_date"],
            volatility_history[label],
            linewidth=1.8,
            label=MODEL_SHORT_LABELS.get(label, label),
            color=MODEL_COLORS[label],
        )

    ax.set_title(f"Rullende {window_months}-måneders annualiseret volatilitet", pad=18)
    add_plot_subtitle(ax, effective_t_caption(backtest_df, cfg))
    ax.set_xlabel("Evalueringsmåned")
    ax.set_ylabel("Annualiseret volatilitet")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2 if len(MODEL_COLUMNS) > 3 else 1)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_fig(fig, output_path)


def print_turnover_summary(turnover_summary: pd.DataFrame) -> None:
    if turnover_summary.empty:
        print("\nIngen turnover-summary kunne beregnes.")
        return

    output_path = Path(__file__).with_name("turnover_summary.csv")
    turnover_summary.to_csv(output_path, index=False)
    print(f"Gemt: {output_path}")


def plot_turnover_history(turnover_history: pd.DataFrame, cfg: BacktestConfig) -> None:
    if turnover_history.empty:
        print("Ingen turnover-historik at gemme.")
        return

    history_path = Path(__file__).with_name("turnover_history.csv")
    turnover_history.to_csv(history_path, index=False)
    print(f"Gemt: {history_path}")

    if plt is None:
        print("Springer turnover-plot over, fordi matplotlib ikke er installeret.")
        return

    plot_history = add_evaluation_dates(turnover_history)
    output_path = Path(__file__).with_name("turnover_over_time.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in MODEL_COLUMNS:
        ax.plot(
            plot_history["evaluation_date"],
            plot_history[label],
            linewidth=1.8,
            label=MODEL_PLOT_LABELS.get(label, label),
            color=MODEL_COLORS[label],
        )

    ax.set_title("Porteføljeomsætning over tid", pad=18)
    add_plot_subtitle(ax, effective_t_caption(turnover_history, cfg))
    ax.set_xlabel("Evalueringsmåned")
    ax.set_ylabel("Omsætning")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_fig(fig, output_path)


def print_sspw_summary(sspw_summary: pd.DataFrame) -> None:
    if sspw_summary.empty:
        print("\nIngen SSPW-summary kunne beregnes.")
        return

    output_path = Path(__file__).with_name("sspw_summary.csv")
    sspw_summary.to_csv(output_path, index=False)
    print(f"Gemt: {output_path}")


def plot_sspw_history(sspw_history: pd.DataFrame, cfg: BacktestConfig) -> None:
    if sspw_history.empty:
        print("Ingen SSPW-historik at gemme.")
        return

    history_path = Path(__file__).with_name("sspw_history.csv")
    sspw_history.to_csv(history_path, index=False)
    print(f"Gemt: {history_path}")

    if plt is None:
        print("Springer SSPW-plot over, fordi matplotlib ikke er installeret.")
        return

    plot_history = add_evaluation_dates(sspw_history)
    output_path = Path(__file__).with_name("sspw_over_time.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in MODEL_COLUMNS:
        ax.plot(
            plot_history["evaluation_date"],
            plot_history[label],
            linewidth=1.8,
            label=MODEL_PLOT_LABELS.get(label, label),
            color=MODEL_COLORS[label],
        )

    ax.set_title("Sum af kvadrerede porteføljevægte (SSPW) over tid", pad=18)
    add_plot_subtitle(ax, effective_t_caption(sspw_history, cfg))
    ax.set_xlabel("Evalueringsmåned")
    ax.set_ylabel("SSPW")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_fig(fig, output_path)


def print_condition_number_summary(history: pd.DataFrame) -> None:
    if history.empty:
        print("\nIngen konditionstal-historik kunne beregnes.")
        return

    valid = history["condition_number"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        print("\nkonditionstal kunne ikke beregnes stabilt for perioden.")
        return

    print()
    print("konditionstal for kovariansmatricen (walk-forward estimering):")
    print(
        f"Min / median / max: "
        f"{valid.min():.2f} / {valid.median():.2f} / {valid.max():.2f}"
    )
    if "n_common_sample_months" in history:
        print(
            f"T_common bag konditionstallene: "
            f"{history['n_common_sample_months'].min():.0f} / "
            f"{history['n_common_sample_months'].median():.0f} / "
            f"{history['n_common_sample_months'].max():.0f}"
        )


def plot_condition_number_history(history: pd.DataFrame, cfg: BacktestConfig) -> None:
    if history.empty:
        print("Ingen konditionstal-historik at gemme.")
        return

    history_path = Path(__file__).with_name("condition_number_history.csv")
    history.to_csv(history_path, index=False)
    print(f"Gemt: {history_path}")

    if plt is None:
        print("Springer konditionstal-plot over, fordi matplotlib ikke er installeret.")
        return

    plot_history = add_evaluation_dates(history)
    finite = np.isfinite(plot_history["condition_number"])
    if not finite.any():
        print("Springer konditionstal-plot over, fordi alle værdier er ikke-finite.")
        return

    output_path = Path(__file__).with_name("condition_number_over_time.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        plot_history.loc[finite, "evaluation_date"],
        plot_history.loc[finite, "condition_number"],
        color="#bf4d24",
        linewidth=1.5,
    )
    ax.set_yscale("log")
    ax.set_title("Kovariansmatricens konditionstal over tid", pad=18)
    add_plot_subtitle(ax, effective_t_caption(history, cfg))
    ax.set_xlabel("Evalueringsmåned")
    ax.set_ylabel("Konditionstal (log-skala)")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_fig(fig, output_path)


def save_correlation_heatmap(correlation: pd.DataFrame, title: str, output_path: Path, subtitle: Optional[str] = None) -> None:
    corr_values = np.ma.masked_invalid(np.clip(correlation.to_numpy(dtype=float), -1.0, 1.0))
    n_assets = correlation.shape[0]
    tick_step = max(1, int(np.ceil(n_assets / 20)))
    tick_positions = np.arange(0, n_assets, tick_step)
    tick_labels = [str(pos + 1) for pos in tick_positions]
    cmap = LinearSegmentedColormap.from_list(
        "GnBu_continuous", plt.get_cmap("GnBu")(np.linspace(0.0, 1.0, 256))
    )
    norm = mcolors.Normalize(vmin=0.2, vmax=1.0)
    cmap.set_bad("#d9d9d9")

    fig, ax = plt.subplots(figsize=(8, 8))
    heatmap = ax.imshow(
        corr_values,
        cmap=cmap, norm=norm, interpolation="none", aspect="equal",
    )
    ax.set_title(title, fontsize=14, pad=18)
    if subtitle:
        add_plot_subtitle(ax, subtitle)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Aktivindeks")
    ax.set_ylabel("Aktivindeks")
    ax.plot(
        [-0.5, n_assets - 0.5],
        [-0.5, n_assets - 0.5],
        color="#0f172a",
        linewidth=1.0,
        alpha=0.8,
    )

    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_ticks([])
    colorbar.ax.tick_params(length=0)
    colorbar.set_label("Korrelation", rotation=270, labelpad=14)

    fig.tight_layout()
    _save_fig(fig, output_path)


def save_correlation_dendrogram(snapshot: CorrelationSnapshot, cfg: BacktestConfig) -> None:
    linkage_matrix = compute_single_linkage_matrix(snapshot.correlation)
    if linkage_matrix is None:
        print("Springer dendrogram over, fordi linkage-matricen ikke kunne beregnes.")
        return

    output_path = Path(__file__).with_name(f"corr_dendrogram_{snapshot.evaluation_month}.png")
    labels = [str(label) for label in snapshot.correlation.index]
    show_labels = len(labels) <= 60

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        labels=labels if show_labels else None,
        leaf_rotation=90, leaf_font_size=6,
        above_threshold_color="#24507a", ax=ax,
    )
    ax.set_title("Dendrogram med enkeltkobling (samme som HRP)", pad=18)
    add_plot_subtitle(ax, snapshot_estimation_caption(snapshot, cfg))
    ax.set_ylabel("Afstand")
    if not show_labels:
        ax.set_xlabel("Aktiver")

    fig.tight_layout()
    _save_fig(fig, output_path)
    if not show_labels:
        print("Dendrogram labels er skjult pga. mange aktiver.")


def save_marchenko_pastur_plot(snapshot: CorrelationSnapshot, cfg: BacktestConfig) -> None:
    corr_values = snapshot.correlation.to_numpy(dtype=float)
    if corr_values.size == 0:
        print("Springer Marchenko-Pastur-plot over, fordi korrelationsmatricen er tom.")
        return

    corr_eigvals = np.linalg.eigvalsh(corr_values)
    corr_eigvals = corr_eigvals[np.isfinite(corr_eigvals)]
    if corr_eigvals.size == 0:
        print("Springer Marchenko-Pastur-plot over, fordi egenværdierne ikke er finite.")
        return

    q, lambda_minus, lambda_plus = marchenko_pastur_bounds(
        snapshot.n_eligible_assets,
        snapshot.n_common_sample_months,
    )
    n_signal = int((corr_eigvals > lambda_plus).sum())
    eig_desc = np.sort(corr_eigvals)[::-1]
    top_k = min(len(eig_desc), max(15, n_signal + 10))

    bulk_candidates = corr_eigvals[corr_eigvals <= lambda_plus * 1.6]
    if bulk_candidates.size:
        bulk_cap = float(max(lambda_plus * 1.08, np.quantile(bulk_candidates, 0.98) * 1.05))
    else:
        bulk_cap = float(lambda_plus * 1.15)
    bulk_cap = min(bulk_cap, float(max(lambda_plus * 1.2, corr_eigvals.max() * 0.35)))
    bulk_cap = max(bulk_cap, float(lambda_plus * 1.08))

    eigen_grid = np.linspace(max(1e-4, lambda_minus * 0.6), bulk_cap, 500)
    mp_density = marchenko_pastur_density(eigen_grid, q, lambda_minus, lambda_plus)
    bulk_eigvals = corr_eigvals[corr_eigvals <= bulk_cap]
    hidden_outliers = int((corr_eigvals > bulk_cap).sum())

    output_path = Path(__file__).with_name("covariance_vs_mp_snapshot.png")
    fig = plt.figure(figsize=(13.2, 6.6), facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.34, 1.0],
        width_ratios=[1.15, 0.95],
        hspace=0.28,
        wspace=0.12,
    )
    ax_info = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_rank = fig.add_subplot(gs[1, 1])

    ax_info.axis("off")
    ax_info.text(
        0.0,
        0.86,
        "Korrelations-egenværdier vs. Marchenko-Pastur",
        fontsize=17,
        fontweight="bold",
        color=CBS_DARK_BLUE,
        ha="left",
        va="top",
    )
    ax_info.text(
        0.0,
        0.18,
        (
            f"{snapshot_estimation_caption(snapshot, cfg)}\n"
            f"q = N/T = {q:.3f}   |   lambda- = {lambda_minus:.3f}   |   "
            f"lambda+ = {lambda_plus:.3f}   |   signal-egenværdier = {n_signal}"
        ),
        fontsize=10.2,
        color="#425466",
        ha="left",
        va="bottom",
        linespacing=1.55,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F7FAFD",
            "edgecolor": CBS_LIGHT_BLUE,
            "linewidth": 1.2,
        },
    )

    ax_hist.hist(
        bulk_eigvals,
        bins=min(16, max(8, int(np.sqrt(max(len(bulk_eigvals), 1))))),
        density=True,
        color=CBS_LIGHT_BLUE,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.95,
    )
    ax_hist.axvspan(lambda_minus, lambda_plus, color=CBS_MIDDLE_BLUE, alpha=0.18, label="MP-støjbånd")
    ax_hist.plot(eigen_grid, mp_density, color=CBS_RED, linewidth=2.6, label="Marchenko-Pastur-tæthed")
    ax_hist.axvline(lambda_minus, color=CBS_CHESTNUT, linestyle=":", linewidth=1.4)
    ax_hist.axvline(lambda_plus, color=CBS_DARK_BLUE, linestyle="--", linewidth=1.6, label=f"lambda+ = {lambda_plus:.2f}")
    ax_hist.set_xlim(0.0, bulk_cap)
    ax_hist.set_title("Hovedområde: empirisk fordeling mod MP-støjbånd", fontsize=12.5, color=CBS_DARK_BLUE, pad=12)
    ax_hist.set_xlabel("Egenværdi")
    ax_hist.set_ylabel("Tæthed")
    ax_hist.grid(alpha=0.18, color=CBS_MIDDLE_BLUE)
    ax_hist.legend(loc="upper right", fontsize=9)
    if hidden_outliers:
        ax_hist.text(
            0.98,
            0.72,
            f"{hidden_outliers} store afvigere skjult\nfor at zoome ind på hovedområdet",
            transform=ax_hist.transAxes,
            ha="right",
            va="top",
            fontsize=9.2,
            color=CBS_CHESTNUT,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#FFF7F5",
                "edgecolor": CBS_RED,
                "linewidth": 1.0,
            },
        )

    top_ranks = np.arange(1, top_k + 1)
    top_vals = eig_desc[:top_k]
    signal_mask = top_vals > lambda_plus
    ax_rank.fill_between([0.5, top_k + 0.5], 0, lambda_plus, color=CBS_LIGHT_BLUE, alpha=0.22)
    ax_rank.vlines(top_ranks, 0, top_vals, color=CBS_MIDDLE_BLUE, linewidth=2.0, alpha=0.65)
    ax_rank.scatter(top_ranks[~signal_mask], top_vals[~signal_mask], color="#A8B4C8", s=34, zorder=3, label="Inden for MP")
    if signal_mask.any():
        ax_rank.scatter(top_ranks[signal_mask], top_vals[signal_mask], color=CBS_DARK_BLUE, s=42, zorder=4, label="Over lambda+")
    ax_rank.axhline(lambda_plus, color=CBS_RED, linestyle="--", linewidth=1.6, label=f"lambda+ = {lambda_plus:.2f}")
    ax_rank.set_title(f"Top {top_k} sorterede egenværdier", fontsize=12.5, color=CBS_DARK_BLUE, pad=12)
    ax_rank.set_xlabel("Rang")
    ax_rank.set_ylabel("Egenværdi")
    ax_rank.grid(alpha=0.18, color=CBS_MIDDLE_BLUE)
    ax_rank.legend(loc="upper right", fontsize=9)
    if top_vals.size:
        ax_rank.text(
            0.98,
            0.08,
            f"Største egenværdi = {top_vals[0]:.2f}",
            transform=ax_rank.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.2,
            color=CBS_DARK_GREEN,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "#F4FBF9",
                "edgecolor": CBS_DARK_GREEN,
                "linewidth": 1.0,
            },
        )

    for ax in (ax_hist, ax_rank):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#A8B4C8")
        ax.spines["bottom"].set_color("#A8B4C8")

    fig.subplots_adjust(top=0.90, left=0.07, right=0.98, bottom=0.11)
    _save_fig(fig, output_path)


def print_correlation_snapshot(snapshot: CorrelationSnapshot, cfg: BacktestConfig, preview_size: int = 12) -> None:
    preview_size = max(2, min(preview_size, snapshot.n_eligible_assets))
    corr_preview = snapshot.correlation.iloc[:preview_size, :preview_size].round(4)
    corr_sorted_preview = snapshot.correlation_sorted.iloc[:preview_size, :preview_size].round(4)

    print()
    print(f"Korrelationsdiagnostik for {snapshot.evaluation_month}:")
    print(
        f"Universmaned: {snapshot.universe_month} | "
        f"Market cap-kandidater: {snapshot.n_universe_assets} | "
        f"Aktiver brugt: {snapshot.n_eligible_assets} | "
        f"Common-sample maneder: {snapshot.n_common_sample_months} | "
        f"Lookback: {cfg.lookback_months} mdr"
    )
    if cfg.require_complete_case_months:
        print(f"Common-sample maneder efter strict dropna-filter: {snapshot.n_common_sample_months} ud af {cfg.lookback_months}")
    else:
        print(
            f"Common-sample maneder i vinduet uden strict dropna-filter: "
            f"{snapshot.n_common_sample_months} ud af {cfg.lookback_months}"
        )
    print(
        "Plot-ordning: single linkage + quasi-diagonal ordering (samme som HRP) | "
        f"T/N-forhold: {snapshot.diagnostics['time_to_assets_ratio']:.3f} | "
        f"Median off-diagonal korrelation: {snapshot.diagnostics['median_offdiag_corr']:.4f} | "
        f"Mindste egenvaerdi: {snapshot.diagnostics['min_eigenvalue']:.4e} | "
        f"konditionstal: {snapshot.diagnostics['condition_number']:.2f}"
    )
    print(
        f"Marchenko-Pastur: q={snapshot.diagnostics['mp_q_ratio']:.3f} | "
        f"lambda-={snapshot.diagnostics['mp_lambda_minus']:.3f} | "
        f"lambda+={snapshot.diagnostics['mp_lambda_plus']:.3f} | "
        f"egenvaerdier over lambda+={int(snapshot.diagnostics['eigenvalues_above_mp'])}"
    )
    if snapshot.diagnostics["time_to_assets_ratio"] < 1.0:
        print("Advarsel: Faerre common-sample maneder end aktiver. Korrelationen kan vaere ustabil.")

    print()
    print(f"Usorterede signed korrelationer for {snapshot.evaluation_month}:")
    print(corr_preview.to_string())
    print()
    print(f"Sorterede signed korrelationer for {snapshot.evaluation_month}:")
    print(corr_sorted_preview.to_string())

    outputs = {
        f"corr_common_sample_{snapshot.evaluation_month}.csv": snapshot.correlation,
        f"corr_common_sample_sorted_{snapshot.evaluation_month}.csv": snapshot.correlation_sorted,
        f"corr_diagnostics_{snapshot.evaluation_month}.csv": pd.DataFrame([snapshot.diagnostics]),
    }
    for filename, frame in outputs.items():
        output_path = Path(__file__).with_name(filename)
        frame.to_csv(output_path, index="diagnostics" not in filename)
        print(f"Gemt: {output_path}")

    if plt is None or mcolors is None:
        print("Springer korrelationsplot over, fordi matplotlib ikke er installeret.")
        return

    heatmaps = {
        f"corr_unsorted_{snapshot.evaluation_month}.png": (snapshot.correlation, "Korrelationsmatrix (før HRP-sortering)"),
        f"corr_sorted_{snapshot.evaluation_month}.png": (snapshot.correlation_sorted, "Sorteret korrelationsmatrix (efter HRP-sortering)"),
    }
    for filename, (correlation, title) in heatmaps.items():
        save_correlation_heatmap(
            correlation,
            title,
            Path(__file__).with_name(filename),
            subtitle=snapshot_estimation_caption(snapshot, cfg),
        )

    save_correlation_dendrogram(snapshot, cfg)
    save_marchenko_pastur_plot(snapshot, cfg)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = BacktestConfig(
        min_estimation_obs_ratio=0.95,
        require_complete_case_months=True,
        risk_free_annual=0.0,
        start_month="1990-01",
        end_month="2021-12",
    )

    returns_panel, market_cap_panel = load_crsp_panels(DATA_PATH)

    print(f"\nDatafil: {DATA_PATH}")
    print("\nPaneldiagnostik:")
    print(f"Returns panel shape: {returns_panel.shape}")
    print(f"Market cap panel shape: {market_cap_panel.shape}")
    print(f"Forste maned: {returns_panel.index.min()}, sidste maned: {returns_panel.index.max()}")
    print(f"Andel observerede afkast i panel: {returns_panel.notna().mean().mean():.4f}")
    print(f"Andel observerede market cap-vaerdier i panel: {market_cap_panel.notna().mean().mean():.4f}")

    artifacts = build_backtest_artifacts(returns_panel, market_cap_panel, cfg)
    backtest_df = artifacts.backtest_df
    if backtest_df.empty:
        raise ValueError("Ingen gyldige out-of-sample maneder. Prov lavere minimumskrav eller kortere lookback.")

    performance_df, diff_df = build_summary_tables(backtest_df, cfg)
    print_report(cfg, backtest_df, performance_df, diff_df, artifacts.turnover_summary, artifacts.sspw_summary)
    export_comprehensive_results_csv(
        cfg,
        backtest_df,
        performance_df,
        artifacts.turnover_summary,
        artifacts.sspw_summary,
        artifacts.condition_history,
    )
    plot_allocation_method_returns(backtest_df, cfg)
    plot_rolling_sharpe_history(backtest_df, cfg)
    plot_rolling_annualized_volatility(backtest_df, cfg)
    print_turnover_summary(artifacts.turnover_summary)
    plot_turnover_history(artifacts.turnover_history, cfg)
    print_sspw_summary(artifacts.sspw_summary)
    plot_sspw_history(artifacts.sspw_history, cfg)
    if not artifacts.weights_history.empty:
        weights_history_path = Path(__file__).with_name("portfolio_weights_history.csv")
        artifacts.weights_history.to_csv(weights_history_path, index=False)
        print(f"Gemt: {weights_history_path}")
    plot_average_top_weight_distribution(artifacts.weights_history, cfg)
    plot_weight_snapshot_pies(artifacts.weights_history, cfg)

    condition_history = artifacts.condition_history
    print_condition_number_summary(condition_history)
    plot_condition_number_history(condition_history, cfg)
    plot_effective_estimation_window(condition_history, cfg)

    corr_snapshot = artifacts.correlation_snapshot
    if corr_snapshot is None:
        print("\nKunne ikke bygge et korrelationssnapshot.")
    else:
        print_correlation_snapshot(corr_snapshot, cfg)


if __name__ == "__main__":
    main()
