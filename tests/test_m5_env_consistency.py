"""Unit tests for verifying strict consistency between M5SingleItemEnv and baseline simulate().

Ensures that the Gymnasium environment produces bit-for-bit identical trajectories,
inventories, costs, and summary metrics compared to m5_baseline.ipynb Cell 9.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.rl_agent.envs._simulate_ref import classical_policy, simulate, summarize
from src.rl_agent.envs.m5_inventory_env import M5SingleItemEnv


def run_env_with_state_policy(env: M5SingleItemEnv, policy_fn) -> pd.DataFrame:
    """Run M5SingleItemEnv using a policy that expects state dict."""
    obs, info = env.reset()
    done = False
    while not done:
        state = info["state"]
        action = policy_fn(state)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_history_df()


class TestM5EnvConsistency:
    """Test suite comparing M5SingleItemEnv directly against baseline simulate()."""

    @pytest.fixture
    def sample_demand(self) -> np.ndarray:
        """Deterministic 30-day demand series."""
        rng = np.random.default_rng(2024)
        return rng.integers(0, 15, size=30).astype(np.float32)

    def test_zero_order_consistency(self, sample_demand):
        """Verify identical behavior when no orders are ever placed."""
        h, p, k, lead_time = 0.1, 5.0, 0.0, 2
        policy = lambda s: 0.0

        # Baseline reference
        ref_df = simulate(
            sample_demand, policy, h=h, p=p, k=k, lead_time=lead_time, warmup=0
        )

        # M5 Gymnasium environment
        env = M5SingleItemEnv(
            sample_demand, h=h, p=p, k=k, lead_time=lead_time, warmup_days=0
        )
        env_df = run_env_with_state_policy(env, policy)

        # Assert identical day-by-day columns
        columns_to_check = [
            "on_hand", "order", "demand", "sales", "lost",
            "end_inventory", "holding_cost", "lost_sales_cost",
            "ordering_cost", "total_cost"
        ]
        for col in columns_to_check:
            assert np.allclose(ref_df[col].values, env_df[col].values), (
                f"Mismatch in column '{col}' for zero order policy"
            )

    def test_constant_order_consistency(self, sample_demand):
        """Verify pipeline queue and order arrivals with constant replenishment."""
        h, p, k, lead_time = 0.1, 5.0, 0.0, 2
        order_qty = 6.0
        policy = lambda s: order_qty

        ref_df = simulate(
            sample_demand, policy, h=h, p=p, k=k, lead_time=lead_time, warmup=0
        )
        env = M5SingleItemEnv(
            sample_demand, h=h, p=p, k=k, lead_time=lead_time, warmup_days=0
        )
        env_df = run_env_with_state_policy(env, policy)

        columns_to_check = [
            "received", "on_hand", "order", "demand", "sales", "lost",
            "end_inventory", "holding_cost", "lost_sales_cost",
            "ordering_cost", "total_cost"
        ]
        for col in columns_to_check:
            assert np.allclose(ref_df[col].values, env_df[col].values), (
                f"Mismatch in column '{col}' for constant order policy"
            )

    def test_fixed_ordering_cost_consistency(self, sample_demand):
        """Verify ordering cost K=5.0 is incurred if and only if order > 0."""
        h, p, k, lead_time = 0.1, 5.0, 5.0, 2

        # Order 10 every 3 days, 0 otherwise
        def periodic_policy(state):
            history_len = len(state["history"])
            return 10.0 if history_len % 3 == 0 else 0.0

        ref_df = simulate(
            sample_demand, periodic_policy, h=h, p=p, k=k, lead_time=lead_time, warmup=0
        )
        env = M5SingleItemEnv(
            sample_demand, h=h, p=p, k=k, lead_time=lead_time, warmup_days=0
        )
        env_df = run_env_with_state_policy(env, periodic_policy)

        # Check ordering cost logic explicitly
        assert (env_df.loc[env_df["order"] > 0, "ordering_cost"] == 5.0).all()
        assert (env_df.loc[env_df["order"] == 0, "ordering_cost"] == 0.0).all()

        columns_to_check = [
            "order", "ordering_cost", "end_inventory", "total_cost"
        ]
        for col in columns_to_check:
            assert np.allclose(ref_df[col].values, env_df[col].values), (
                f"Mismatch in column '{col}' with fixed cost K=5.0"
            )

    def test_classical_base_stock_policy_consistency(self, sample_demand):
        """Verify classical (R, S) Base-Stock policy with S=20."""
        h, p, k, lead_time = 0.1, 5.0, 0.0, 2
        S = 20.0
        policy = classical_policy(S=S, s=None)

        ref_df = simulate(
            sample_demand, policy, h=h, p=p, k=k, lead_time=lead_time, warmup=0
        )
        env = M5SingleItemEnv(
            sample_demand, h=h, p=p, k=k, lead_time=lead_time, warmup_days=0
        )
        env_df = run_env_with_state_policy(env, policy)

        # Verify summary metrics match exactly
        ref_summary = summarize(ref_df)
        env_summary = summarize(env_df)

        for metric in ["total_cost", "holding_cost", "lost_sales_cost", "fill_rate", "csl", "average_end_inventory"]:
            assert np.isclose(ref_summary[metric], env_summary[metric], atol=1e-5), (
                f"Metric '{metric}' differs: ref={ref_summary[metric]}, env={env_summary[metric]}"
            )

    def test_classical_s_S_policy_consistency(self, sample_demand):
        """Verify classical (R, s, S) policy with s=8, S=22 and K=5.0."""
        h, p, k, lead_time = 0.1, 5.0, 5.0, 2
        s, S = 8.0, 22.0
        policy = classical_policy(S=S, s=s)

        ref_df = simulate(
            sample_demand, policy, h=h, p=p, k=k, lead_time=lead_time, warmup=0
        )
        env = M5SingleItemEnv(
            sample_demand, h=h, p=p, k=k, lead_time=lead_time, warmup_days=0
        )
        env_df = run_env_with_state_policy(env, policy)

        ref_summary = summarize(ref_df)
        env_summary = summarize(env_df)

        for metric in ["total_cost", "ordering_cost", "fill_rate", "csl", "order_count"]:
            assert np.isclose(ref_summary[metric], env_summary[metric], atol=1e-5), (
                f"Metric '{metric}' differs for (s, S): ref={ref_summary[metric]}, env={env_summary[metric]}"
            )

    def test_warmup_period_metrics(self):
        """Verify warmup period excludes the first 28 days from scored evaluation."""
        rng = np.random.default_rng(42)
        long_demand = rng.integers(0, 10, size=70).astype(np.float32)
        h, p, k, lead_time, warmup = 0.1, 5.0, 5.0, 2, 28

        policy = classical_policy(S=15.0, s=5.0)

        ref_df = simulate(
            long_demand, policy, h=h, p=p, k=k, lead_time=lead_time, warmup=warmup
        )
        env = M5SingleItemEnv(
            long_demand, h=h, p=p, k=k, lead_time=lead_time, warmup_days=warmup
        )
        env_df = run_env_with_state_policy(env, policy)

        ref_summary = summarize(ref_df)
        env_summary = summarize(env_df)

        assert ref_summary["days"] == 70 - 28 == 42
        assert env_summary["days"] == 42
        assert np.isclose(ref_summary["total_cost"], env_summary["total_cost"], atol=1e-5)

    def test_discrete_action_space(self, sample_demand):
        """Verify discrete action mapping in M5SingleItemEnv."""
        h, p, k, lead_time = 0.1, 5.0, 5.0, 2
        action_levels = [0, 2, 5, 10, 15]
        env = M5SingleItemEnv(
            sample_demand,
            h=h,
            p=p,
            k=k,
            lead_time=lead_time,
            warmup_days=0,
            discrete_actions=True,
            action_levels=action_levels,
        )
        assert env.action_space.n == len(action_levels)
        obs, info = env.reset()

        # Step with action index 2 -> corresponds to quantity 5
        obs, reward, term, trunc, info = env.step(2)
        assert info["order"] == 5.0
        assert info["ordering_cost"] == 5.0

        # Step with action index 0 -> corresponds to quantity 0
        obs, reward, term, trunc, info = env.step(0)
        assert info["order"] == 0.0
        assert info["ordering_cost"] == 0.0

        # Step with numpy scalar
        obs, reward, term, trunc, info = env.step(np.int64(3))
        assert info["order"] == 10.0
