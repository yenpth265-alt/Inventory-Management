"""PPO Training Script for M5 Walmart Inventory Replenishment.

Trains Stable-Baselines3 PPO agents on empirical M5 demand series.
Uses M5 Validation split (56 days) for hyperparameter tuning and model checkpointing.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.rl_agent.data_loader_m5 import load_m5
from src.rl_agent.envs.m5_inventory_env import M5SingleItemEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on M5 Inventory Data.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/m5-forecasting-accuracy",
        help="Path to unzipped M5 data directory.",
    )
    parser.add_argument("--store_id", type=str, default="CA_1", help="M5 store ID.")
    parser.add_argument("--n_skus", type=int, default=1, help="Number of SKUs to sample.")
    parser.add_argument("--item_id", type=str, default=None, help="Explicit SKU item_id.")
    parser.add_argument("--k", type=float, default=0.0, help="Fixed ordering cost (0.0 or 5.0).")
    parser.add_argument("--h", type=float, default=0.1, help="Unit holding cost.")
    parser.add_argument("--p", type=float, default=5.0, help="Unit lost sales penalty.")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps.")
    parser.add_argument("--eval_freq", type=int, default=5000, help="Evaluation frequency in steps.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--discrete", action="store_true", default=False, help="Use discrete replenishment action space.")
    parser.add_argument(
        "--action_levels",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 5, 8, 10, 15, 20],
        help="Order quantity levels when discrete mode is enabled.",
    )
    parser.add_argument("--ent_coef", type=float, default=None, help="PPO entropy coefficient.")
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/m5_ppo",
        help="Directory to save trained model checkpoints.",
    )
    return parser.parse_args()


def train_m5_ppo(
    data_dir: str = "data/m5-forecasting-accuracy",
    store_id: str = "CA_1",
    n_skus: int = 1,
    item_id: Optional[str] = None,
    k: float = 0.0,
    h: float = 0.1,
    p: float = 5.0,
    timesteps: int = 100000,
    eval_freq: int = 5000,
    seed: int = 2024,
    output_dir: str = "models/m5_ppo",
    discrete: bool = False,
    action_levels: Optional[List[int]] = None,
    ent_coef: Optional[float] = None,
    learning_rate: float = 3e-4,
) -> Dict[str, Any]:
    """Train PPO agent on M5 data and evaluate on validation split."""
    if action_levels is None:
        action_levels = [0, 1, 2, 3, 5, 8, 10, 15, 20]
    if ent_coef is None:
        ent_coef = 0.05 if discrete else 0.01

    print("=" * 70)
    print(f"Starting PPO Training for M5 Inventory Control (K={k})")
    print(f"Data Dir: {data_dir} | Store: {store_id} | Seed: {seed}")
    print(f"Action Space: {'Discrete ' + str(action_levels) if discrete else 'Continuous Box [0, 200]'}")
    print(f"Exploration ent_coef: {ent_coef} | Learning rate: {learning_rate}")
    print("=" * 70)

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load M5 Dataset
    item_ids = [item_id] if item_id else None
    dataset = load_m5(
        data_dir=data_dir,
        store_id=store_id,
        n_skus=n_skus if item_ids is None else None,
        item_ids=item_ids,
        seed=seed,
        validation_days=56,
        test_days=112,
        min_train_days=365,
    )

    chosen_sku = dataset.items["item_id"].iloc[0]
    first_priced_idx = int(dataset.items["first_priced_index"].iloc[0] or 0)
    print(f"Selected SKU: {chosen_sku} (first priced day index: {first_priced_idx})")

    # Time split boundaries
    n_total_days = dataset.demand.shape[1]
    train_end = n_total_days - 56 - 112
    val_end = n_total_days - 112

    # Demand and price slices
    demand_series = dataset.demand[0]
    prices_series = dataset.prices[0]

    train_demand = demand_series[first_priced_idx:train_end]
    train_prices = prices_series[first_priced_idx:train_end]

    val_demand = demand_series[train_end:val_end]
    val_prices = prices_series[train_end:val_end]

    test_demand = demand_series[val_end:]
    test_prices = prices_series[val_end:]

    print(f"Train days: {len(train_demand)} | Val days: {len(val_demand)} | Test days: {len(test_demand)}")

    # Reference economic parameters
    v_i = float(np.nanmedian(train_prices))
    unit_cost = float(0.7 * v_i)  # PURCHASE_RATIO = 0.7
    print(f"Median train selling price v_i: ${v_i:.2f} | Unit cost c_i: ${unit_cost:.2f}")
    print(f"Holding cost h: ${h:.2f} | Shortage penalty p: ${p:.2f} | Setup cost K: ${k:.2f}")

    # 2. Setup Environments
    train_env = M5SingleItemEnv(
        demand_series=train_demand,
        prices=train_prices,
        h=h,
        p=p,
        k=k,
        unit_cost=unit_cost,
        lead_time=2,
        warmup_days=28,
        initial_stock=0.0,
        discrete_actions=discrete,
        action_levels=action_levels if discrete else None,
    )
    train_env = Monitor(train_env)

    val_env = M5SingleItemEnv(
        demand_series=val_demand,
        prices=val_prices,
        h=h,
        p=p,
        k=k,
        unit_cost=unit_cost,
        lead_time=2,
        warmup_days=0,  # Full 56 days scored on validation
        initial_stock=0.0,
        discrete_actions=discrete,
        action_levels=action_levels if discrete else None,
    )
    val_env = Monitor(val_env)

    # 3. Setup Checkpoint Directories
    k_tag = f"k{int(k)}" if k.is_integer() else f"k{k}"
    model_save_dir = Path(output_dir) / f"{chosen_sku}_{k_tag}"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = model_save_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = model_save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 4. Setup Evaluation Callback
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )

    # 5. Initialize PPO Agent
    # Policy gradient actor-critic with continuous Box or Discrete action space
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
    )

    print("\nTraining PPO Agent...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)

    # 6. Save Final Model and Config
    final_model_path = model_save_dir / "final_model.zip"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_dir / 'best_model.zip'}")

    summary_info = {
        "sku": chosen_sku,
        "k": k,
        "h": h,
        "p": p,
        "v_i": v_i,
        "unit_cost": unit_cost,
        "timesteps": timesteps,
        "train_days": len(train_demand),
        "val_days": len(val_demand),
        "test_days": len(test_demand),
        "discrete_actions": discrete,
        "action_levels": action_levels if discrete else None,
        "ent_coef": ent_coef,
        "learning_rate": learning_rate,
        "best_model_path": str(best_model_dir / "best_model.zip"),
        "final_model_path": str(final_model_path),
    }

    with open(model_save_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_info, f, indent=2)

    return summary_info


if __name__ == "__main__":
    args = parse_args()
    train_m5_ppo(
        data_dir=args.data_dir,
        store_id=args.store_id,
        n_skus=args.n_skus,
        item_id=args.item_id,
        k=args.k,
        h=args.h,
        p=args.p,
        timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        seed=args.seed,
        output_dir=args.output_dir,
        discrete=args.discrete,
        action_levels=args.action_levels,
        ent_coef=args.ent_coef,
        learning_rate=args.lr,
    )
