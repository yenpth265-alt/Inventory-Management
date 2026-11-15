"""Reference implementation of baseline simulation and classical policies.

Extracted directly from notebooks/baselines/m5_baseline.ipynb to serve as
a ground-truth reference for unit testing, consistency verification, and evaluation.
"""

from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def classical_policy(S: float, s: Optional[float] = None) -> Callable[[Dict[str, Any]], float]:
    """Create a classical (R, S) or (R, s, S) inventory policy.

    If s is None: Order up to S whenever review occurs: max(0, S - IP).
    If s is specified: Order up to S only if IP < s: max(0, S - IP) if IP < s else 0.
    """
    def policy(state: Dict[str, Any]) -> float:
        ip = state["stock"] + sum(state["pipeline"])
        return max(0.0, S - ip) if s is None or ip < s else 0.0
    return policy


def simulate(
    demand: Union[List[float], np.ndarray],
    policy: Callable[[Dict[str, Any]], float],
    *,
    known: Optional[List[Dict[str, Any]]] = None,
    prices: Optional[Union[List[float], np.ndarray]] = None,
    unit_cost: float = 1.0,
    history: Tuple[float, ...] = (),
    warmup: int = 0,
    k: float = 0.0,
    lead_time: int = 2,
    review: int = 1,
    h: float = 0.1,
    p: float = 5.0,
    initial_stock: float = 0.0,
    initial_pipeline: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Simulate inventory replenishment matching m5_baseline.ipynb Cell 9."""
    demand = np.asarray(demand, dtype=float)
    assert demand.ndim == 1 and len(demand) > warmup >= 0
    assert np.isfinite(demand).all() and (demand >= 0).all()
    assert lead_time >= 1 and review >= 1
    known = [{} for _ in demand] if known is None else known
    prices = np.full(len(demand), np.nan) if prices is None else np.asarray(prices)
    assert len(known) == len(demand) == len(prices)
    pipeline = dict({} if initial_pipeline is None else initial_pipeline)
    observed, stock, rows = list(history), float(initial_stock), []
    for t, d in enumerate(demand):
        received = pipeline.pop(t, 0.0)
        stock += received
        outstanding = tuple(pipeline.get(t + j, 0.0) for j in range(1, lead_time + 1))
        state = {
            "stock": stock,
            "pipeline": outstanding,
            "history": tuple(observed),
            "known": dict(known[t]),
        }
        q, seconds = 0.0, 0.0
        if t % review == 0:
            started = perf_counter()
            proposed = float(policy(state))
            seconds = perf_counter() - started
            assert np.isfinite(proposed) and proposed >= 0
            q = float(np.rint(proposed))
            if q > 0:
                pipeline[t + lead_time] = pipeline.get(t + lead_time, 0.0) + q
        sales = min(stock, d)
        lost = d - sales
        end = stock - sales
        holding, penalty, ordering = h * end, p * lost, k if q > 0 else 0.0
        margin = sales * (prices[t] - unit_cost) if np.isfinite(prices[t]) else (0 if sales == 0 else np.nan)
        rows.append(
            dict(
                day=t,
                review=t % review == 0,
                cycle=t // review,
                review_period=review,
                received=received,
                on_hand=stock,
                ip=stock + sum(outstanding),
                pipeline=outstanding,
                order=q,
                arrival_day=t + lead_time if q else np.nan,
                demand=d,
                sales=sales,
                lost=lost,
                end_inventory=end,
                on_order=sum(pipeline.values()),
                pipeline_due=tuple(sorted(pipeline.items())),
                holding_cost=holding,
                lost_sales_cost=penalty,
                ordering_cost=ordering,
                total_cost=holding + penalty + ordering,
                inventory_value=end * unit_cost,
                gross_margin=margin,
                inference_seconds=seconds,
                scored=t >= warmup,
            )
        )
        observed.append(sales)
        stock = end
    return pd.DataFrame(rows)


def ratio(numerator: float, denominator: float) -> float:
    """Calculate ratio with nan safe division."""
    return float(numerator / denominator) if denominator > 0 else np.nan


def summarize(log: pd.DataFrame) -> Dict[str, Any]:
    """Summarize simulation metrics matching m5_baseline.ipynb Cell 9."""
    scored = log[log.scored]
    n = len(scored)
    cycles = scored.groupby("cycle").agg(
        size=("day", "size"),
        begins=("review", "first"),
        lost=("lost", "sum"),
    )
    complete = cycles[(cycles["size"] == int(log.review_period.iloc[0])) & cycles.begins]
    total = lambda col: float(scored[col].sum())
    inventory_value = total("inventory_value") / n if n > 0 else 0.0
    margin = float(scored.gross_margin.sum(min_count=n))
    result = {
        name: total(name)
        for name in ["holding_cost", "lost_sales_cost", "ordering_cost", "total_cost", "inference_seconds"]
    }
    result.update(
        days=n,
        total_demand=total("demand"),
        total_sales=total("sales"),
        total_lost=total("lost"),
        fill_rate=ratio(total("sales"), total("demand")),
        complete_cycles=len(complete),
        successful_cycles=int((complete.lost == 0).sum()),
        csl=ratio(int((complete.lost == 0).sum()), len(complete)),
        average_end_inventory=total("end_inventory") / n if n > 0 else 0.0,
        average_inventory_value=inventory_value,
        gross_margin=margin,
        gmroi=ratio(margin, inventory_value),
        order_count=int((scored.order > 0).sum()),
        total_ordered=total("order"),
        terminal_inventory=float(log.end_inventory.iloc[-1]) if len(log) > 0 else 0.0,
        terminal_pipeline=float(log.on_order.iloc[-1]) if len(log) > 0 else 0.0,
    )
    return result
