"""Benchmark Evaluation Script for M5 Walmart Inventory Replenishment.

Evaluates Classical (R, S), Classical (R, s, S), and Deep RL (PPO) policies
on the exact same 112-day independent M5 test period.
Outputs standardized benchmark metrics matching the team's approved plan.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.rl_agent.data_loader_m5 import load_m5
from src.rl_agent.envs._simulate_ref import classical_policy, ratio, simulate, summarize
from src.rl_agent.envs.m5_inventory_env import M5SingleItemEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate M5 Inventory Policies.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/m5-forecasting-accuracy",
        help="Path to M5 data directory.",
    )
    parser.add_argument("--store_id", type=str, default="CA_1", help="Store ID.")
    parser.add_argument("--item_id", type=str, default=None, help="Explicit SKU item_id.")
    parser.add_argument(
        "--k_values",
        type=float,
        nargs="+",
        default=[0.0, 5.0],
        help="List of fixed ordering costs K to benchmark.",
    )
    parser.add_argument("--h", type=float, default=0.1, help="Holding cost per unit/day.")
    parser.add_argument("--p", type=float, default=5.0, help="Lost sales penalty per unit.")
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models/m5_ppo",
        help="Directory containing trained PPO models.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="reports/tables/m5_benchmark_results.csv",
        help="Path to save benchmark results CSV.",
    )
    return parser.parse_args()


def tune_classical_policies_on_val(
    val_demand: np.ndarray,
    h: float,
    p: float,
    k: float,
    lead_time: int = 2,
    max_search: int = 25,
) -> Tuple[float, float, float]:
    """Find optimal S for (R, S) and optimal (s, S) for (R, s, S) on validation split.

    Returns:
        Tuple[best_S_for_RS, best_s_for_sS, best_S_for_sS]
    """
    # 1. Tune Base-Stock S
    best_S_RS = 1.0
    best_cost_RS = float("inf")
    for S in range(1, max_search + 1):
        res = summarize(
            simulate(val_demand, classical_policy(S=float(S)), h=h, p=p, k=k, lead_time=lead_time)
        )
        if res["total_cost"] < best_cost_RS:
            best_cost_RS = res["total_cost"]
            best_S_RS = float(S)

    # 2. Tune (s, S)
    best_s_sS = 0.0
    best_S_sS = best_S_RS
    best_cost_sS = float("inf")
    for s_val in range(0, max_search):
        for S_val in range(s_val + 1, max_search + 1):
            res = summarize(
                simulate(
                    val_demand,
                    classical_policy(S=float(S_val), s=float(s_val)),
                    h=h,
                    p=p,
                    k=k,
                    lead_time=lead_time,
                )
            )
            if res["total_cost"] < best_cost_sS:
                best_cost_sS = res["total_cost"]
                best_s_sS = float(s_val)
                best_S_sS = float(S_val)

    return best_S_RS, best_s_sS, best_S_sS


def evaluate_policy_on_gym_env(
    env: M5SingleItemEnv,
    predict_fn: Any,
) -> Dict[str, Any]:
    """Evaluate an arbitrary policy on M5SingleItemEnv across test period."""
    obs, info = env.reset()
    done = False
    while not done:
        action = predict_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    df = env.get_history_df()
    return summarize(df)


def run_benchmark(
    data_dir: str = "data/m5-forecasting-accuracy",
    store_id: str = "CA_1",
    item_id: Optional[str] = None,
    k_values: List[float] = [0.0, 5.0],
    h: float = 0.1,
    p: float = 5.0,
    models_dir: str = "models/m5_ppo",
    output_csv: str = "reports/tables/m5_benchmark_results.csv",
) -> pd.DataFrame:
    """Run full benchmark across classical baselines and PPO on M5 test split."""
    print("=" * 75)
    print("M5 Walmart Inventory Replenishment Benchmark")
    print(f"Store: {store_id} | Holding Cost h: {h} | Lost Sales Penalty p: {p}")
    print(f"Configurations K: {k_values}")
    print("=" * 75)

    # 1. Load M5 Data
    item_ids = [item_id] if item_id else None
    dataset = load_m5(
        data_dir=data_dir,
        store_id=store_id,
        n_skus=1 if item_ids is None else None,
        item_ids=item_ids,
        seed=2024,
        validation_days=56,
        test_days=112,
        min_train_days=365,
    )

    sku = dataset.items["item_id"].iloc[0]
    first_priced_idx = int(dataset.items["first_priced_index"].iloc[0] or 0)
    print(f"Evaluating SKU: {sku}")

    demand = dataset.demand[0]
    prices = dataset.prices[0]
    n_days = len(demand)
    train_end = n_days - 56 - 112
    val_end = n_days - 112

    train_prices = prices[first_priced_idx:train_end]
    val_demand = demand[train_end:val_end]
    val_prices = prices[train_end:val_end]
    test_demand = demand[val_end:]
    test_prices = prices[val_end:]

    v_i = float(np.nanmedian(train_prices))
    unit_cost = float(0.7 * v_i)

    print(f"Test period: {len(test_demand)} days | Total test demand: {test_demand.sum():.1f} units")

    records = []

    for k in k_values:
        print(f"\n--- Running Benchmark for Setup Cost K = {k} ---")

        # 2. Tune Classical Baselines on Validation Split
        best_S_RS, best_s_sS, best_S_sS = tune_classical_policies_on_val(
            val_demand, h=h, p=p, k=k, lead_time=2
        )
        print(f"Tuned (R, S): S* = {best_S_RS}")
        print(f"Tuned (R, s, S): s* = {best_s_sS}, S* = {best_S_sS}")

        # 3. Evaluate Classical (R, S) on Test
        env_rs = M5SingleItemEnv(
            test_demand, prices=test_prices, h=h, p=p, k=k, unit_cost=unit_cost, lead_time=2, warmup_days=0
        )
        rs_fn = lambda obs, info, S=best_S_RS: classical_policy(S)(info["state"])
        res_rs = evaluate_policy_on_gym_env(env_rs, rs_fn)

        records.append(
            {
                "item_id": sku,
                "store_id": store_id,
                "K": k,
                "policy": "Classical (R, S)",
                "params": f"S={best_S_RS:.0f}",
                "total_cost": res_rs["total_cost"],
                "holding_cost": res_rs["holding_cost"],
                "lost_sales_cost": res_rs["lost_sales_cost"],
                "ordering_cost": res_rs["ordering_cost"],
                "fill_rate": res_rs["fill_rate"],
                "csl": res_rs["csl"],
                "average_end_inventory": res_rs["average_end_inventory"],
                "order_count": res_rs["order_count"],
                "total_ordered": res_rs["total_ordered"],
                "gmroi": res_rs["gmroi"],
            }
        )

        # 4. Evaluate Classical (R, s, S) on Test
        env_rss = M5SingleItemEnv(
            test_demand, prices=test_prices, h=h, p=p, k=k, unit_cost=unit_cost, lead_time=2, warmup_days=0
        )
        rss_fn = lambda obs, info, S=best_S_sS, s=best_s_sS: classical_policy(S, s)(info["state"])
        res_rss = evaluate_policy_on_gym_env(env_rss, rss_fn)

        records.append(
            {
                "item_id": sku,
                "store_id": store_id,
                "K": k,
                "policy": "Classical (R, s, S)",
                "params": f"s={best_s_sS:.0f}, S={best_S_sS:.0f}",
                "total_cost": res_rss["total_cost"],
                "holding_cost": res_rss["holding_cost"],
                "lost_sales_cost": res_rss["lost_sales_cost"],
                "ordering_cost": res_rss["ordering_cost"],
                "fill_rate": res_rss["fill_rate"],
                "csl": res_rss["csl"],
                "average_end_inventory": res_rss["average_end_inventory"],
                "order_count": res_rss["order_count"],
                "total_ordered": res_rss["total_ordered"],
                "gmroi": res_rss["gmroi"],
            }
        )

        # 5. Evaluate PPO Agent on Test
        k_tag = f"k{int(k)}" if k.is_integer() else f"k{k}"
        model_path = Path(models_dir) / f"{sku}_{k_tag}" / "best_model" / "best_model.zip"

        if not model_path.exists():
            # Fallback to final model if best model not found
            model_path = Path(models_dir) / f"{sku}_{k_tag}" / "final_model.zip"

        if model_path.exists():
            print(f"Loading PPO checkpoint from: {model_path}")
            ppo_model = PPO.load(str(model_path))

            # Inspect if model was trained with discrete actions
            summary_path = model_path.parent / "training_summary.json"
            if not summary_path.exists():
                summary_path = model_path.parent.parent / "training_summary.json"

            discrete_actions = False
            action_levels = None
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        saved_cfg = json.load(f)
                        discrete_actions = bool(saved_cfg.get("discrete_actions", False))
                        action_levels = saved_cfg.get("action_levels", None)
                except Exception as e:
                    print(f"Notice: Failed to read training summary: {e}")

            if hasattr(ppo_model.action_space, "n"):
                discrete_actions = True

            env_ppo = M5SingleItemEnv(
                test_demand,
                prices=test_prices,
                h=h,
                p=p,
                k=k,
                unit_cost=unit_cost,
                lead_time=2,
                warmup_days=0,
                discrete_actions=discrete_actions,
                action_levels=action_levels,
            )
            ppo_fn = lambda obs, info, m=ppo_model: m.predict(obs, deterministic=True)[0]
            res_ppo = evaluate_policy_on_gym_env(env_ppo, ppo_fn)

            param_desc = (
                f"PPO (Discrete, {len(action_levels or env_ppo.action_levels)} lvls)"
                if discrete_actions
                else "Neural Actor-Critic (Continuous)"
            )

            records.append(
                {
                    "item_id": sku,
                    "store_id": store_id,
                    "K": k,
                    "policy": "Deep RL (PPO Agent)",
                    "params": param_desc,
                    "total_cost": res_ppo["total_cost"],
                    "holding_cost": res_ppo["holding_cost"],
                    "lost_sales_cost": res_ppo["lost_sales_cost"],
                    "ordering_cost": res_ppo["ordering_cost"],
                    "fill_rate": res_ppo["fill_rate"],
                    "csl": res_ppo["csl"],
                    "average_end_inventory": res_ppo["average_end_inventory"],
                    "order_count": res_ppo["order_count"],
                    "total_ordered": res_ppo["total_ordered"],
                    "gmroi": res_ppo["gmroi"],
                }
            )
        else:
            print(f"Warning: No PPO model found at {model_path}, skipping RL for K={k}")

    # Build DataFrame
    df_results = pd.DataFrame(records)

    # Calculate Savings % vs (R, S) within each K configuration
    for k in k_values:
        mask = df_results["K"] == k
        rs_mask = mask & (df_results["policy"] == "Classical (R, S)")
        if rs_mask.any():
            base_cost = df_results.loc[rs_mask, "total_cost"].values[0]
            if base_cost > 0:
                savings = (base_cost - df_results.loc[mask, "total_cost"]) / base_cost * 100.0
                df_results.loc[mask, "savings_pct_vs_RS"] = savings
            else:
                df_results.loc[mask, "savings_pct_vs_RS"] = np.nan

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\nBenchmark results successfully exported to: {output_path}")

    # Display Pretty Table
    display_cols = [
        "K",
        "policy",
        "params",
        "total_cost",
        "holding_cost",
        "lost_sales_cost",
        "ordering_cost",
        "fill_rate",
        "csl",
        "savings_pct_vs_RS",
    ]
    print("\n" + "=" * 90)
    print("FINAL BENCHMARK COMPARISON TABLE (112 TEST DAYS)")
    print("=" * 90)
    try:
        print(df_results[display_cols].to_markdown(index=False, floatfmt=".2f"))
    except Exception:
        print(df_results[display_cols].to_string(index=False))

    return df_results


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        data_dir=args.data_dir,
        store_id=args.store_id,
        item_id=args.item_id,
        k_values=args.k_values,
        h=args.h,
        p=args.p,
        models_dir=args.models_dir,
        output_csv=args.output_csv,
    )
