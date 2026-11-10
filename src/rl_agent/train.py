"""Training Script for Reinforcement Learning Inventory Agents (PPO & DQN).

Loads configuration from YAML, sets up Gymnasium environments with Monitor wrappers,
configures EvalCallback for tracking the best model, logs metrics to TensorBoard,
and trains using Stable-Baselines3.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.rl_agent.envs.single_item_env import SingleItemInventoryEnv


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration dictionary from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_agent(
    config_path: str = "experiments/configs/rl_demo_config.yaml",
    algo: str = "ppo",
    total_timesteps: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Train an RL inventory replenishment agent.

    Args:
        config_path (str): Path to YAML config file.
        algo (str): Algorithm choice ('ppo' or 'dqn').
        total_timesteps (Optional[int]): Override total timesteps if provided.
        device (str): Computing device ('auto', 'cuda', 'cpu').
        seed (Optional[int]): Random seed override.

    Returns:
        Dict[str, Any]: Summary metrics and artifact paths.
    """
    config = load_config(config_path)
    algo_name = algo.lower()

    # Determine seed
    train_seed = seed if seed is not None else int(config.get("training", {}).get("seed", 42))

    # Output directories
    results_dir = Path("experiments/results/rl_models")
    logs_dir = Path("experiments/logs/tensorboard")
    eval_log_dir = Path(f"experiments/logs/eval_{algo_name}")

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate environments
    train_env = Monitor(
        SingleItemInventoryEnv(config),
        filename=str(eval_log_dir / "train_monitor.csv"),
    )
    eval_env = Monitor(
        SingleItemInventoryEnv(config),
        filename=str(eval_log_dir / "eval_monitor.csv"),
    )

    train_env.reset(seed=train_seed)
    eval_env.reset(seed=train_seed + 1000)

    # Resolve timesteps
    train_cfg = config.get("training", {})
    if total_timesteps is not None:
        timesteps = total_timesteps
    else:
        timesteps = int(train_cfg.get("total_timesteps", 200_000))
        if algo_name == "dqn":
            # Default DQN demo runs 100k steps unless overridden
            timesteps = min(timesteps, 100_000)

    # Configure EvalCallback
    eval_freq = max(2048, timesteps // 20)  # Evaluate ~20 times during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(results_dir),
        log_path=str(eval_log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Initialize Agent
    gamma = float(train_cfg.get("gamma", 0.99))
    lr = float(train_cfg.get("learning_rate", 3e-4))

    print("=" * 70)
    print(f"BẮT ĐẦU HUẤN LUYỆN RL INVENTORY AGENT [{algo_name.upper()}]")
    print(f"- Config File      : {config_path}")
    print(f"- Total Timesteps  : {timesteps:,}")
    print(f"- Device           : {device} (CUDA Available: {torch.cuda.is_available()})")
    print(f"- Learning Rate    : {lr}")
    print(f"- Gamma (Discount) : {gamma}")
    print(f"- Random Seed      : {train_seed}")
    print(f"- TensorBoard Log  : {logs_dir}")
    print("=" * 70)

    start_time = time.time()

    if algo_name == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=lr,
            n_steps=int(train_cfg.get("n_steps", 2048)),
            batch_size=int(train_cfg.get("batch_size", 64)),
            n_epochs=int(train_cfg.get("n_epochs", 10)),
            gamma=gamma,
            gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
            clip_range=float(train_cfg.get("clip_range", 0.2)),
            tensorboard_log=str(logs_dir),
            seed=train_seed,
            device=device,
            verbose=1,
        )
    elif algo_name == "dqn":
        dqn_cfg = train_cfg.get("dqn", {})
        model = DQN(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=lr,
            buffer_size=int(dqn_cfg.get("buffer_size", 50_000)),
            learning_starts=int(dqn_cfg.get("learning_starts", 1_000)),
            batch_size=int(dqn_cfg.get("batch_size", 64)),
            gamma=gamma,
            target_update_interval=int(dqn_cfg.get("target_update_interval", 1_000)),
            exploration_fraction=float(dqn_cfg.get("exploration_fraction", 0.2)),
            exploration_final_eps=float(dqn_cfg.get("exploration_final_eps", 0.02)),
            tensorboard_log=str(logs_dir),
            seed=train_seed,
            device=device,
            verbose=1,
        )
    else:
        raise ValueError(f"Thuật toán không hỗ trợ: '{algo_name}'. Chọn 'ppo' hoặc 'dqn'.")

    # Execute training
    model.learn(total_timesteps=timesteps, callback=eval_callback, tb_log_name=f"{algo_name}_run")

    elapsed_time = time.time() - start_time

    # Save final model
    final_model_path = results_dir / f"{algo_name}_final_model.zip"
    model.save(str(final_model_path))

    # Rename best model to standard convention if created
    default_best = results_dir / "best_model.zip"
    best_model_path = results_dir / f"{algo_name}_best_model.zip"
    if default_best.exists():
        if best_model_path.exists():
            best_model_path.unlink()
        default_best.rename(best_model_path)

    print("\n" + "=" * 70)
    print(f"HUẤN LUYỆN HOÀN TẤT TRONG: {elapsed_time:.2f} giây ({elapsed_time/60:.2f} phút)!")
    print(f"- Best Model Saved : {best_model_path}")
    print(f"- Final Model Saved: {final_model_path}")
    print("=" * 70 + "\n")

    train_env.close()
    eval_env.close()

    return {
        "algo": algo_name,
        "timesteps": timesteps,
        "elapsed_seconds": elapsed_time,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Inventory Replenishment Agent")
    parser.add_argument("--config", type=str, default="experiments/configs/rl_demo_config.yaml", help="Path to config YAML")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"], help="RL Algorithm")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()
    train_agent(
        config_path=args.config,
        algo=args.algo,
        total_timesteps=args.timesteps,
        device=args.device,
        seed=args.seed,
    )
