"""Benchmark Evaluation Script for RL Inventory Replenishment.

Compares 5 policies on identical test episodes (deterministic seeds):
1. Random Policy (Lower Bound)
2. Base-Stock Policy (Order-up-to S=60)
3. (s, S) Policy (s=30, S=70)
4. DQN Agent
5. PPO Agent

Calculates standard economic metrics:
- Fill Rate (%)
- Cycle Service Level - CSL (%)
- Total Operating Cost ($)
- Cost Breakdown: Holding, Stockout, Ordering
- GMROI (Gross Margin Return on Inventory Investment)
- Stockout Count (per 52-week episode)
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import DQN, PPO

from src.rl_agent.envs.single_item_env import SingleItemInventoryEnv
from src.rl_agent.policies.base_stock import BaseStockPolicy, RandomPolicy, sSPolicy


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_policy_on_episodes(
    policy: Any,
    env: SingleItemInventoryEnv,
    num_episodes: int = 100,
    seed_start: int = 1000,
) -> Dict[str, float]:
    """Evaluate a single policy across multiple independent test episodes.

    Args:
        policy: Policy instance with predict(obs, deterministic=True) method.
        env: SingleItemInventoryEnv instance.
        num_episodes: Number of test years/episodes to evaluate.
        seed_start: Base seed for test episodes (ensures reproducibility).

    Returns:
        Dict[str, float]: Aggregated metrics over all episodes.
    """
    episode_total_costs = []
    episode_holding_costs = []
    episode_stockout_costs = []
    episode_ordering_costs = []
    episode_fill_rates = []
    episode_csls = []
    episode_gmrois = []
    episode_stockouts = []

    for ep in range(num_episodes):
        ep_seed = seed_start + ep
        obs, _ = env.reset(seed=ep_seed)

        done = False
        total_demand = 0.0
        total_sales = 0.0
        no_stockout_periods = 0
        stockout_periods = 0

        ep_holding = 0.0
        ep_stockout = 0.0
        ep_ordering = 0.0
        ep_total_cost = 0.0
        ep_gross_profit = 0.0
        inventory_levels = []

        while not done:
            # Predict action
            action_raw = policy.predict(obs, deterministic=True)
            if isinstance(action_raw, tuple):
                action = action_raw[0]
            else:
                action = action_raw

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Accumulate step metrics
            demand = info["demand"]
            sales = info["sales"]
            lost_sales = info["lost_sales"]

            total_demand += demand
            total_sales += sales

            if lost_sales == 0.0:
                no_stockout_periods += 1
            else:
                stockout_periods += 1

            ep_holding += info["holding_cost"]
            ep_stockout += info["stockout_cost"]
            ep_ordering += info["ordering_cost"]
            ep_total_cost += info["total_cost"]
            ep_gross_profit += info["gross_profit"]
            inventory_levels.append(info["ending_inventory"])

        # Episode-level summary metrics
        ep_fill_rate = (total_sales / total_demand * 100.0) if total_demand > 0 else 100.0
        ep_csl = (no_stockout_periods / env.episode_length) * 100.0

        # GMROI = Total Gross Profit / Average Inventory Investment
        avg_inv_value = np.mean(inventory_levels) * env.unit_cost
        ep_gmroi = (ep_gross_profit / avg_inv_value) if avg_inv_value > 0 else 0.0

        episode_total_costs.append(ep_total_cost)
        episode_holding_costs.append(ep_holding)
        episode_stockout_costs.append(ep_stockout)
        episode_ordering_costs.append(ep_ordering)
        episode_fill_rates.append(ep_fill_rate)
        episode_csls.append(ep_csl)
        episode_gmrois.append(ep_gmroi)
        episode_stockouts.append(stockout_periods)

    return {
        "Total Cost ($)": float(np.mean(episode_total_costs)),
        "Total Cost Std": float(np.std(episode_total_costs)),
        "Holding Cost ($)": float(np.mean(episode_holding_costs)),
        "Stockout Cost ($)": float(np.mean(episode_stockout_costs)),
        "Ordering Cost ($)": float(np.mean(episode_ordering_costs)),
        "Fill Rate (%)": float(np.mean(episode_fill_rates)),
        "CSL (%)": float(np.mean(episode_csls)),
        "GMROI": float(np.mean(episode_gmrois)),
        "Stockout Count": float(np.mean(episode_stockouts)),
    }


def run_benchmark(
    config_path: str = "experiments/configs/rl_demo_config.yaml",
    num_episodes: int = 100,
    save_csv: bool = True,
    models_dir: Optional[Any] = None,
) -> pd.DataFrame:
    """Run full benchmarking across all 5 models and return comparison DataFrame."""
    config = load_config(config_path)
    env = SingleItemInventoryEnv(config)

    action_levels = config.get("environment", {}).get("action_levels")
    eval_cfg = config.get("evaluation", {})
    seed_start = int(eval_cfg.get("eval_seed_start", 1000))

    if models_dir is None:
        resolved_models_dir = PROJECT_ROOT / "experiments" / "results" / "rl_models"
    else:
        resolved_models_dir = Path(models_dir)

    results = {}

    print("=" * 80)
    print(f"BẮT ĐẦU ĐỐI CHUẨN 5 CHÍNH SÁCH TỒN KHO TRÊN {num_episodes} EPISODES (TEST SET)")
    print(f"- Seed bắt đầu : {seed_start} -> {seed_start + num_episodes - 1}")
    print("=" * 80)

    # 1. Random Policy
    print("[1/5] Đánh giá Random Policy...")
    random_policy = RandomPolicy(action_levels, seed=seed_start)
    results["Random Policy"] = evaluate_policy_on_episodes(
        random_policy, env, num_episodes, seed_start
    )

    # 2. Base-Stock Policy (S=60)
    print("[2/5] Đánh giá Base-Stock Policy (S=60)...")
    base_stock_policy = BaseStockPolicy(
        target_order_up_to_S=60.0, action_levels=action_levels
    )
    results["Base-Stock (S=60)"] = evaluate_policy_on_episodes(
        base_stock_policy, env, num_episodes, seed_start
    )

    # 3. (s, S) Policy (s=30, S=70)
    print("[3/5] Đánh giá Classical (s, S) Policy (s=30, S=70)...")
    s_s_policy = sSPolicy(
        reorder_point_s=30.0, order_up_to_S=70.0, action_levels=action_levels
    )
    results["Classical (s, S)"] = evaluate_policy_on_episodes(
        s_s_policy, env, num_episodes, seed_start
    )

    # 4. DQN Agent
    dqn_path = resolved_models_dir / "dqn_best_model.zip"
    if dqn_path.exists():
        print(f"[4/5] Đánh giá DQN Agent (Load từ {dqn_path})...")
        dqn_model = DQN.load(str(dqn_path), device="cpu")
        results["DQN Agent"] = evaluate_policy_on_episodes(
            dqn_model, env, num_episodes, seed_start
        )
    else:
        print(f"[4/5] Bỏ qua DQN (Không tìm thấy {dqn_path})")

    # 5. PPO Agent
    ppo_path = resolved_models_dir / "ppo_best_model.zip"
    if ppo_path.exists():
        print(f"[5/5] Đánh giá PPO Agent (Load từ {ppo_path})...")
        ppo_model = PPO.load(str(ppo_path), device="cpu")
        results["PPO Agent"] = evaluate_policy_on_episodes(
            ppo_model, env, num_episodes, seed_start
        )
    else:
        print(f"[5/5] Bỏ qua PPO (Không tìm thấy {ppo_path})")

    df_results = pd.DataFrame(results).T

    # Calculate cost reduction relative to Random baseline
    baseline_cost = df_results.loc["Random Policy", "Total Cost ($)"]
    df_results["Cost Reduction (%)"] = (
        (baseline_cost - df_results["Total Cost ($)"]) / baseline_cost * 100.0
    )

    # Reorder columns logically
    cols = [
        "Total Cost ($)",
        "Cost Reduction (%)",
        "Fill Rate (%)",
        "CSL (%)",
        "Holding Cost ($)",
        "Stockout Cost ($)",
        "Ordering Cost ($)",
        "GMROI",
        "Stockout Count",
    ]
    df_results = df_results[cols]

    print("\n" + "=" * 90)
    print("BẢNG ĐỐI CHUẨN HIỆU NĂNG TỔNG HỢP (BENCHMARK RESULTS)")
    print("=" * 90)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(df_results)
    print("=" * 90 + "\n")

    if save_csv:
        results_dir = PROJECT_ROOT / "experiments" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "evaluation_benchmark.csv"
        df_results.to_csv(csv_path)
        print(f"Đã lưu kết quả đối chuẩn vào: {csv_path}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Inventory Replenishment Policies")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/rl_demo_config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of test episodes",
    )

    args = parser.parse_args()
    run_benchmark(config_path=args.config, num_episodes=args.episodes)
