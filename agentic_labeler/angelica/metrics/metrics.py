import json
from typing import Any, Iterable, Callable, Optional, List, Tuple
from sklearn.metrics import cohen_kappa_score

from angelica.storage.sqlite.store_sqlite import SQLiteStore
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

COLORS = [
    "#0F62FE",  # Blue 60
    "#42BE65",  # Green 50
    "#FF832B",  # Orange 40
    "#BE95FF",  # Purple 40
    "#EE5396",  # Magenta 40
    "#33B1FF",  # Cyan 40
    "#FA4D56",  # Red 50
    "#8A3FFC",  # Purple 60
]

def _parse_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}

def bucket_confidence(x):
    """
    Bucket a numeric confidence score into categorical bins
    suitable for Cohen's kappa.
    """
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None

    if x < 0.6:
        return "low"
    if x < 0.85:
        return "mid"
    return "high"

DEFAULT_FIELD_TRANSFORMS = {
    "confidence_score": bucket_confidence,
}

def _get_path(d: dict, path: str) -> Any:
    """Get nested field from dict using dotted path, e.g. 'a.b.c'."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _matches_target_combinations(
    label_dict: dict,
    target_combinations: Optional[List[Tuple[bool, bool, bool]]] = None
) -> bool:
    """
    Check if a label matches any of the target combinations.
    
    Args:
        label_dict: The label dictionary to check
        target_combinations: List of tuples (is_integration_test, is_self_contained, is_deployed)
                           If None, all tests pass (no filtering)
    
    Returns:
        True if the label matches any target combination or if no filtering is applied
    """
    if target_combinations is None:
        return True
    
    is_integration = label_dict.get('is_integration_test', None)
    is_self_contained = label_dict.get('is_self_contained', None)
    is_deployed = label_dict.get('is_test_executed_against_deployed_services', None)
    
    combination = (is_integration, is_self_contained, is_deployed)
    return combination in target_combinations


# Default target combinations: TFT, TTT, TTF
DEFAULT_TARGET_COMBINATIONS = [
    (True, False, True),   # TFT: is_integration_test=True, is_self_contained=False, is_deployed=True
    (True, True, True),    # TTT: is_integration_test=True, is_self_contained=True, is_deployed=True
    (True, True, False)    # TTF: is_integration_test=True, is_self_contained=True, is_deployed=False
]


def rolling_kappa_for_field(
    store: SQLiteStore,
    agent_a: str,
    agent_b: str,
    field: str,
    window: int = 50,
    transform: Optional[Callable[[Any], Any]] = None,
    drop_missing: bool = True,
    target_combinations: Optional[List[Tuple[bool, bool, bool]]] = DEFAULT_TARGET_COMBINATIONS,
) -> pd.DataFrame:
    """
    Rolling Cohen's kappa for a *single* field path from label JSON.

    - field: dotted path in label JSON, e.g. "pattern_name", "fit_assessment", "is_self_contained"
    - transform: optional function to normalize/bucket values (e.g., confidence into bins)
    - drop_missing: if True, remove rows where either agent value is missing
    - target_combinations: List of tuples (is_integration_test, is_self_contained, is_deployed) to filter by.
                          Defaults to TFT, TTT, TTF combinations. Set to None to disable filtering.
    """
    df = store.fetch_agent_pairwise_json(agent_a, agent_b)
    if df.empty:
        return pd.DataFrame(columns=["created_at", "kappa", "n", "field"])

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    a_vals = []
    b_vals = []
    keep_indices = []
    
    for idx, (a_json, b_json) in enumerate(zip(df["a_label_json"], df["b_label_json"])):
        a_dict = _parse_json(a_json)
        b_dict = _parse_json(b_json)
        
        # Filter by target combinations if specified
        if target_combinations is not None:
            # Check both agent labels - only include if both match target combinations
            if not (_matches_target_combinations(a_dict, target_combinations) and
                    _matches_target_combinations(b_dict, target_combinations)):
                continue
        
        av = _get_path(a_dict, field)
        bv = _get_path(b_dict, field)
        if transform:
            av = transform(av)
            bv = transform(bv)
        a_vals.append(av)
        b_vals.append(bv)
        keep_indices.append(idx)

    if not a_vals:
        return pd.DataFrame(columns=["created_at", "kappa", "n", "field"])

    df2 = pd.DataFrame({
        "created_at": df["created_at"].iloc[keep_indices].values,
        "a": a_vals,
        "b": b_vals
    })

    if drop_missing:
        df2 = df2.dropna(subset=["a", "b"])

    if df2.empty:
        return pd.DataFrame(columns=["created_at", "kappa", "n", "field"])

    out = []
    for i in range(len(df2)):
        start = max(0, i - window + 1)
        sub = df2.iloc[start : i + 1]
        kappa = cohen_kappa_score(sub["a"], sub["b"])
        out.append((df2.iloc[i]["created_at"], float(kappa), len(sub), field))

    return pd.DataFrame(out, columns=["created_at", "kappa", "n", "field"])


def rolling_kappa_for_fields(
    store: SQLiteStore,
    agent_a: str,
    agent_b: str,
    fields: Iterable[str],
    window: int = 50,
    transforms: Optional[dict[str, Callable[[Any], Any]]] = None,
    drop_missing: bool = True,
    target_combinations: Optional[List[Tuple[bool, bool, bool]]] = DEFAULT_TARGET_COMBINATIONS,
) -> pd.DataFrame:
    """
    Compute rolling kappa for multiple fields and return a concatenated dataframe.
    
    Args:
        store: SQLite store containing the data
        agent_a: First agent name
        agent_b: Second agent name
        fields: List of field paths to compute kappa for
        window: Rolling window size
        transforms: Optional dict of field-specific transform functions
        drop_missing: Whether to drop rows with missing values
        target_combinations: List of tuples (is_integration_test, is_self_contained, is_deployed) to filter by.
                           Defaults to TFT, TTT, TTF combinations. Set to None to disable filtering.
    """
    transforms = transforms or {}
    frames = []
    for f in fields:
        frames.append(
            rolling_kappa_for_field(
                store=store,
                agent_a=agent_a,
                agent_b=agent_b,
                field=f,
                window=window,
                transform=transforms.get(f) or DEFAULT_FIELD_TRANSFORMS.get(f),
                drop_missing=drop_missing,
                target_combinations=target_combinations,
            )
        )
    if not frames:
        return pd.DataFrame(columns=["created_at", "kappa", "n", "field"])
    return pd.concat(frames, ignore_index=True)




def plot_kappa(df: pd.DataFrame, title: str) -> None:
    """
    Plots rolling Cohen's kappa using Carbon-style colors.
    If df contains multiple fields, plots each field as a separate line.
    """
    if df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 5))

    ax = plt.gca()
    ax.set_facecolor("#F4F4F4")  # Carbon gray-10 background

    if "field" in df.columns and df["field"].nunique() > 1:
        for (field, sub), color in zip(df.groupby("field"), COLORS):
            ax.plot(
                sub["created_at"],
                sub["kappa"],
                label=str(field),
                color=color,
                linewidth=2.2,
                alpha=0.95,
            )
        ax.legend(
            title="Field",
            frameon=False,
            loc="best"
        )
    else:
        ax.plot(
            df["created_at"],
            df["kappa"],
            color=COLORS[0],
            linewidth=2.5,
        )

    # Titles & labels
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Cohen’s κ (rolling)")

    # Grid styling (Carbon-like)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.grid(False, axis="x")

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.show()

