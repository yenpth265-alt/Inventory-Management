"""Unit Tests for SingleItemInventoryEnv and Baseline Policies.

Verifies:
1. Environment initialization and Gymnasium API compliance.
2. Observation and Action space shapes.
3. Lost Sales mechanics (Inventory strictly non-negative).
4. Lead Time delay (Orders arrive exactly after L periods).
5. Cost and Reward calculations.
6. Deterministic seed reproducibility.
7. Baseline policies execution over a full horizon.
"""

import pytest
import numpy as np

from src.rl_agent.envs.single_item_env import SingleItemInventoryEnv
from src.rl_agent.policies.base_stock import BaseStockPolicy, RandomPolicy, sSPolicy


@pytest.fixture
def test_config():
    """Standard test configuration fixture."""
    return {
        "environment": {
            "item_name": "TEST_SKU_001",
            "max_inventory": 200.0,
            "max_order_quantity": 100.0,
            "lead_time": 2,
            "episode_length": 52,
            "lost_sales": True,
            "holding_cost": 1.0,
            "lost_sales_penalty": 10.0,
            "fixed_ordering_cost": 50.0,
            "unit_cost": 5.0,
            "selling_price": 8.0,
            "demand_distribution": "poisson",
            "demand_mean": 20.0,
            "action_levels": [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100],
        }
    }


class TestSingleItemInventoryEnv:
    """Test suite for SingleItemInventoryEnv."""

    def test_initialization(self, test_config):
        """Test proper initialization and Gymnasium space definitions."""
        env = SingleItemInventoryEnv(test_config)
        assert env.observation_space.shape == (4,)
        assert env.action_space.n == 12
        assert env.lead_time == 2
        assert env.episode_length == 52

    def test_reset(self, test_config):
        """Test reset logic and return types."""
        env = SingleItemInventoryEnv(test_config)
        obs, info = env.reset(seed=42)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert isinstance(info, dict)
        assert "inventory" in info
        assert "pipeline" in info
        assert info["step"] == 0
        assert obs[0] == env.inventory

    def test_step_contract(self, test_config):
        """Test standard Gymnasium 5-tuple return contract from step()."""
        env = SingleItemInventoryEnv(test_config)
        env.reset(seed=42)

        action = 2  # Order quantity 10
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Check required info keys
        required_keys = [
            "step",
            "demand",
            "sales",
            "lost_sales",
            "order_quantity",
            "arrived_quantity",
            "ending_inventory",
            "holding_cost",
            "stockout_cost",
            "ordering_cost",
            "total_cost",
        ]
        for key in required_keys:
            assert key in info, f"Missing key in info dict: {key}"

        # Verify reward is negative of total cost
        assert reward == pytest.approx(-info["total_cost"])

    def test_lost_sales_mechanics(self, test_config):
        """Test that inventory is strictly non-negative under extreme demand."""
        # Set initial inventory to 5 and demand to 50
        env = SingleItemInventoryEnv(test_config)
        env.reset(seed=42, options={"initial_inventory": 5.0})

        # Step with zero order
        action_zero = 0
        obs, reward, terminated, truncated, info = env.step(action_zero)

        # On-hand inventory MUST NOT be negative
        assert info["ending_inventory"] >= 0.0
        assert env.inventory >= 0.0
        # If demand > initial available, lost_sales must be positive
        if info["demand"] > 5.0:
            assert info["lost_sales"] == pytest.approx(info["demand"] - 5.0)

    def test_lead_time_delay(self, test_config):
        """Test that order placed at t=0 arrives exactly at t=2 (L=2)."""
        env = SingleItemInventoryEnv(test_config)
        env.reset(seed=42, options={"initial_inventory": 0.0})

        # Period 0: Order 40 units (action index 7 = 40)
        action_order_40 = 7
        _, _, _, _, info_0 = env.step(action_order_40)
        assert info_0["arrived_quantity"] == 0.0
        assert info_0["order_quantity"] == 40.0

        # Period 1: Order 0 units
        action_order_0 = 0
        _, _, _, _, info_1 = env.step(action_order_0)
        assert info_1["arrived_quantity"] == 0.0  # Not yet arrived

        # Period 2: Order 0 units -> Order from t=0 MUST arrive now!
        _, _, _, _, info_2 = env.step(action_order_0)
        assert info_2["arrived_quantity"] == 40.0

    def test_seed_reproducibility(self, test_config):
        """Test that identical seeds generate bit-exact identical trajectories."""
        env1 = SingleItemInventoryEnv(test_config)
        env2 = SingleItemInventoryEnv(test_config)

        obs1, _ = env1.reset(seed=12345)
        obs2, _ = env2.reset(seed=12345)
        np.testing.assert_array_equal(obs1, obs2)

        for _ in range(10):
            action = env1.action_space.sample()
            obs1, r1, _, _, info1 = env1.step(action)
            obs2, r2, _, _, info2 = env2.step(action)

            np.testing.assert_array_equal(obs1, obs2)
            assert r1 == pytest.approx(r2)
            assert info1["demand"] == pytest.approx(info2["demand"])


class TestBaselinePolicies:
    """Test suite for Classical Baseline Policies."""

    def test_random_policy_execution(self, test_config):
        """Test RandomPolicy runs for full episode without error."""
        env = SingleItemInventoryEnv(test_config)
        obs, _ = env.reset(seed=42)

        policy = RandomPolicy(test_config["environment"]["action_levels"], seed=42)
        done = False
        steps = 0

        while not done:
            action = policy.predict(obs)
            assert 0 <= action < len(test_config["environment"]["action_levels"])
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps == 52

    def test_base_stock_policy_execution(self, test_config):
        """Test BaseStockPolicy runs correctly."""
        env = SingleItemInventoryEnv(test_config)
        obs, _ = env.reset(seed=42)

        policy = BaseStockPolicy(
            target_order_up_to_S=60.0,
            action_levels=test_config["environment"]["action_levels"],
        )
        done = False
        steps = 0

        while not done:
            action = policy.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps == 52

    def test_s_S_policy_execution(self, test_config):
        """Test sSPolicy reorder logic."""
        env = SingleItemInventoryEnv(test_config)
        obs, _ = env.reset(seed=42)

        policy = sSPolicy(
            reorder_point_s=30.0,
            order_up_to_S=70.0,
            action_levels=test_config["environment"]["action_levels"],
        )
        done = False
        steps = 0

        while not done:
            action = policy.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps == 52
