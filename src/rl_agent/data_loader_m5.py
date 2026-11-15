"""M5 Forecasting Dataset Loader for Inventory Replenishment Environments.

Adapted from notebooks/baselines/m5_baseline.ipynb Cell 5 to provide clean,
reusable access to M5 Walmart sales, prices, and calendar splits without notebook dependency.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

KEYS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


@dataclass
class M5Dataset:
    """Container for processed M5 data subsets."""
    demand: np.ndarray
    prices: np.ndarray
    calendar: pd.DataFrame
    items: pd.DataFrame
    audit: Dict[str, Any]


def _header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def _calendar(path: Path, day_columns: List[str], state_id: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"d", "date", "wm_yr_wk", "wday", "month", f"snap_{state_id}"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Calendar missing columns: {required - set(frame.columns)}")
    if frame["d"].isna().any() or frame["d"].duplicated().any():
        raise ValueError("Calendar d must be unique and nonmissing")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("Calendar dates must be unique and nonmissing")
    indexed = frame.set_index("d")
    missing = set(day_columns) - set(indexed.index)
    if missing:
        raise ValueError(f"Calendar missing sales days: {sorted(missing)[:5]}")
    selected = indexed.loc[day_columns].reset_index()
    selected["date"] = pd.to_datetime(selected["date"], errors="raise")
    if not selected["date"].diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("Selected sales calendar is not consecutive daily data")
    selected["weekday"] = selected["wday"].astype("int8")
    selected["event"] = (
        selected.get("event_name_1", pd.Series(index=selected.index, dtype=object)).notna().astype("int8")
    )
    selected["snap"] = selected[f"snap_{state_id}"].astype("int8")
    if selected[["wm_yr_wk", "weekday", "month", "snap"]].isna().any().any():
        raise ValueError("Calendar join keys and known features must be nonmissing")
    return selected


def load_m5(
    data_dir: Union[str, Path],
    store_id: str = "CA_1",
    n_skus: Optional[int] = None,
    seed: int = 2024,
    validation_days: int = 56,
    test_days: int = 112,
    min_train_days: int = 365,
    item_ids: Optional[Sequence[str]] = None,
    sales_file: Optional[str] = None,
) -> M5Dataset:
    """Load one store and a seeded subset, with chronological train/val/test splits."""
    root = Path(data_dir)
    if sales_file is None:
        sales_file = next(
            (
                name
                for name in ("sales_train_evaluation.csv", "sales_train_validation.csv")
                if (root / name).exists()
            ),
            None,
        )
    if sales_file is None:
        raise FileNotFoundError(f"No M5 sales CSV found in {root}")

    sales_path = root / sales_file
    columns = _header(sales_path)
    if not set(KEYS).issubset(columns):
        raise ValueError(f"Sales table missing keys: {set(KEYS) - set(columns)}")
    days = sorted(
        (name for name in columns if name.startswith("d_") and name[2:].isdigit()),
        key=lambda name: int(name[2:]),
    )
    if not days or [int(name[2:]) for name in days] != list(range(1, len(days) + 1)):
        raise ValueError("Sales d_* columns must be consecutive from d_1")
    n_days = len(days)
    train_end = n_days - int(validation_days) - int(test_days)
    if min(validation_days, test_days, min_train_days) < 1 or train_end < min_train_days:
        raise ValueError("Not enough actual labeled days for configured chronological split")
    if n_skus is not None and n_skus < 1:
        raise ValueError("n_skus must be positive or None for all eligible SKUs")

    dtype = {name: "string" for name in KEYS}
    dtype.update({name: "float32" for name in days})
    seen: set = set()
    seen_ids: set = set()
    store_items: Dict[str, set] = {}
    candidates: List[Dict[str, Any]] = []
    row_count = 0

    for chunk in pd.read_csv(sales_path, usecols=KEYS + days, dtype=dtype, chunksize=512):
        if chunk[KEYS].isna().any().any():
            raise ValueError("Missing sales identifier")
        values = chunk[days].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("M5 simulated demand must be finite nonnegative integer observations")
        for row in chunk[KEYS].itertuples(index=False, name=None):
            identifier, item, _, _, store, _ = row
            if (store, item) in seen:
                raise ValueError(f"Duplicate (store_id, item_id): {(store, item)}")
            if identifier in seen_ids:
                raise ValueError(f"Duplicate sales series id: {identifier}")
            seen.add((store, item))
            seen_ids.add(identifier)
            store_items.setdefault(store, set()).add(item)
        row_count += len(chunk)
        selected = chunk.loc[chunk["store_id"].eq(store_id)]
        for _, row in selected.iterrows():
            observed = row[days].to_numpy(dtype=np.float32)
            first_positive = int(np.argmax(observed > 0) + 1) if (observed > 0).any() else None
            candidates.append(
                {
                    "item_id": str(row["item_id"]),
                    "dept_id": str(row["dept_id"]),
                    "cat_id": str(row["cat_id"]),
                    "store_id": str(row["store_id"]),
                    "state_id": str(row["state_id"]),
                    "first_positive_day": first_positive,
                    "first_positive_index": first_positive - 1 if first_positive else None,
                    "eligible_from_train": bool(
                        first_positive is not None and first_positive - 1 <= train_end - min_train_days
                    ),
                    "sales": observed,
                }
            )

    if store_id not in store_items:
        raise ValueError(f"Store {store_id!r} absent; available stores: {sorted(store_items)}")
    by_item = {entry["item_id"]: entry for entry in candidates}
    eligible = sorted(entry["item_id"] for entry in candidates if entry["eligible_from_train"])

    if item_ids is None:
        if n_skus is None:
            chosen = eligible
        else:
            rng = np.random.default_rng(seed)
            sample_size = min(int(n_skus), len(eligible))
            chosen = sorted(rng.choice(eligible, size=sample_size, replace=False).tolist())
    else:
        chosen = list(dict.fromkeys(item_ids))
        absent = set(chosen) - set(by_item)
        if absent:
            raise ValueError(f"Requested item_ids not found in store {store_id!r}: {sorted(absent)}")

    if not chosen:
        raise ValueError(f"No eligible items found for store {store_id!r}")

    demand = np.stack([by_item[item]["sales"] for item in chosen], axis=0).astype(np.float32)
    items = pd.DataFrame(
        [
            {
                "item_id": item,
                "dept_id": by_item[item]["dept_id"],
                "cat_id": by_item[item]["cat_id"],
                "store_id": store_id,
                "state_id": by_item[item]["state_id"],
                "first_positive_day": by_item[item]["first_positive_day"],
                "first_positive_index": by_item[item]["first_positive_index"],
                "eligible_from_train": by_item[item]["eligible_from_train"],
            }
            for item in chosen
        ]
    )

    calendar = _calendar(root / "calendar.csv", days, items["state_id"].iloc[0])
    price_rows = []
    price_count = 0
    price_columns = ["store_id", "item_id", "wm_yr_wk", "sell_price"]

    for chunk in pd.read_csv(
        root / "sell_prices.csv",
        usecols=price_columns,
        dtype={"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float32"},
        chunksize=250000,
    ):
        price_count += len(chunk)
        subset = chunk.loc[chunk["store_id"].eq(store_id) & chunk["item_id"].isin(chosen)]
        if len(subset):
            price_rows.append(subset)

    if not price_rows:
        raise ValueError("No retail price observations for selected items")
    prices_table = pd.concat(price_rows, ignore_index=True)
    if prices_table[price_columns].isna().any().any():
        raise ValueError("Missing selected price key/value")
    if prices_table.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("Duplicate selected store/item/week retail prices")
    if not np.isfinite(prices_table["sell_price"]).all() or (prices_table["sell_price"] <= 0).any():
        raise ValueError("Retail prices must be finite and positive")

    prices = np.full_like(demand, np.nan)
    price_audit = []
    for index, item in enumerate(chosen):
        item_prices = prices_table.loc[prices_table["item_id"].eq(item)].set_index("wm_yr_wk")["sell_price"]
        prices[index] = calendar["wm_yr_wk"].map(item_prices).to_numpy(dtype=np.float32)
        available = np.flatnonzero(np.isfinite(prices[index]))
        first_priced = int(available[0] + 1) if len(available) else None
        first_priced_week = int(item_prices.index.min())
        if ((demand[index] > 0) & ~np.isfinite(prices[index])).any():
            raise ValueError(f"Positive sales without observed same-week retail price: {item}")
        after_launch_missing = (
            int(np.isnan(prices[index, first_priced - 1:]).sum()) if first_priced else n_days
        )
        if after_launch_missing:
            raise ValueError(
                f"Missing retail prices after launch for {item}: {after_launch_missing} days"
            )
        items.loc[index, "first_priced_day"] = first_priced
        items.loc[index, "first_priced_index"] = first_priced - 1 if first_priced else None
        items.loc[index, "first_priced_week"] = first_priced_week
        price_audit.append(
            {
                "item_id": item,
                "first_positive_day": by_item[item]["first_positive_day"],
                "first_priced_day": first_priced,
                "first_priced_week": first_priced_week,
                "missing_price_days": int(np.isnan(prices[index]).sum()),
                "missing_price_days_after_launch": after_launch_missing,
            }
        )

    def split(start: int, stop: int) -> Dict[str, Any]:
        return {
            "first_day": days[start],
            "last_day": days[stop - 1],
            "n_days": stop - start,
            "first_date": calendar["date"].iloc[start].date().isoformat(),
            "last_date": calendar["date"].iloc[stop - 1].date().isoformat(),
        }

    audit = {
        "sales_file": str(sales_path),
        "source_sales_rows": row_count,
        "stores": {store: len(item_set) for store, item_set in sorted(store_items.items())},
        "sales_days": n_days,
        "first_day": days[0],
        "last_day": days[-1],
        "first_date": calendar["date"].iloc[0].date().isoformat(),
        "last_date": calendar["date"].iloc[-1].date().isoformat(),
        "selected_store": store_id,
        "selected_skus": chosen,
        "selected_sku_count": len(chosen),
        "selection_seed": seed,
        "eligible_sku_count_from_train": len(eligible),
        "min_train_days": min_train_days,
        "splits": {
            "train": split(0, train_end),
            "validation": split(train_end, n_days - test_days),
            "test": split(n_days - test_days, n_days),
        },
        "price_source_rows": price_count,
        "selected_price_rows": len(prices_table),
    }

    return M5Dataset(demand, prices, calendar, items, audit)
