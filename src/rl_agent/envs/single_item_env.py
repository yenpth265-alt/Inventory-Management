"""Single-Item Inventory Management Environment with Lost Sales and Lead Time.

Compliant with Farama Gymnasium API v0.29+ / v1.0.
Formulated as a Markov Decision Process (MDP) for retail inventory replenishment
under realistic operational constraints (Lead time L >= 1, Lost Sales, Fixed Ordering Cost).
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SingleItemInventoryEnv(gym.Env):
    """Single-item multi-period periodic review inventory environment.

    Mathematical Formulation:
        State: s_t = [I_t, PL_t, D_{t-1}, t mod T]
        Action: a_t in {0, 1, ..., |action_levels| - 1} mapping to order quantity x_t
        Transition:
            I_t^avail = I_t + Q_{t-L}
            Sales_t = min(I_t^avail, D_t)
            LostSales_t = max(0, D_t - I_t^avail)
            I_{t+1} = I_t^avail - Sales_t >= 0
        Reward:
            R_t = -(h * I_{t+1} + p * LostSales_t + k * I(x_t > 0) + C * x_t)

    Attributes:
        config (Dict[str, Any]): Dictionary containing environment parameters.
        lead_time (int): Order delivery lag (L >= 1).
        holding_cost (float): Unit holding cost per period (h).
        lost_sales_penalty (float): Unit stockout penalty cost per period (p).
        fixed_ordering_cost (float): Fixed setup cost incurred when an order is placed (k).
        unit_cost (float): Unit purchase cost (C).
        selling_price (float): Unit retail price (P).
        demand_mean (float): Mean demand per period (Poisson lambda).
        action_levels (List[int]): Discrete order quantities corresponding to actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the inventory environment with specified configuration."""
        super().__init__()

        cfg = config or {}
        env_cfg = cfg.get("environment", cfg)

        self.num_items: int = 1
        self.item_name: str = env_cfg.get("item_name", "FOODS_1_001_CA_1")
        self.max_inventory: float = float(env_cfg.get("max_inventory", 200.0))
        self.max_order_quantity: float = float(env_cfg.get("max_order_quantity", 100.0))

        # Time horizon and lead time
        self.lead_time: int = int(env_cfg.get("lead_time", 2))
        self.episode_length: int = int(env_cfg.get("episode_length", 52))
        self.lost_sales: bool = bool(env_cfg.get("lost_sales", True))

        # Economic parameters
        self.holding_cost: float = float(env_cfg.get("holding_cost", 1.0))
        self.lost_sales_penalty: float = float(env_cfg.get("lost_sales_penalty", 10.0))
        self.fixed_ordering_cost: float = float(env_cfg.get("fixed_ordering_cost", 50.0))
        self.unit_cost: float = float(env_cfg.get("unit_cost", 5.0))
        self.selling_price: float = float(env_cfg.get("selling_price", 8.0))

        # Demand parameters
        self.demand_distribution: str = env_cfg.get("demand_distribution", "poisson")
        self.demand_mean: float = float(env_cfg.get("demand_mean", 20.0))

        # Action levels for discrete action space
        default_actions = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
        self.action_levels: List[int] = env_cfg.get("action_levels", default_actions)

        # Action Space: Discrete index selecting from action_levels
        self.action_space = spaces.Discrete(len(self.action_levels))

        # Observation Space:
        # [On-hand Inventory (I_t), Pipeline In-Transit (Q_{t-1}), Last Demand (D_{t-1}), Time Feature (t % T)]
        # Shape: (4,)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([self.max_inventory, self.max_order_quantity, np.inf, self.episode_length], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state variables
        self.inventory: float = 0.0
        self.pipeline: deque = deque([0.0] * self.lead_time, maxlen=self.lead_time)
        self.last_demand: float = self.demand_mean
        self.current_step: int = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the initial state.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[Dict[str, Any]]): Additional options (e.g. initial_inventory).

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info dictionary.
        """
        super().reset(seed=seed)

        options = options or {}
        # Initial inventory defaults to 1.5x expected period demand
        default_init_inv = self.demand_mean * 1.5
        self.inventory = float(options.get("initial_inventory", default_init_inv))
        self.inventory = min(self.inventory, self.max_inventory)

        # Pipeline is initialized with empty in-transit orders
        self.pipeline = deque([0.0] * self.lead_time, maxlen=self.lead_time)
        self.last_demand = float(self.demand_mean)
        self.current_step = 0

        obs = self._get_obs()
        info = {
            "inventory": self.inventory,
            "pipeline": list(self.pipeline),
            "step": self.current_step,
        }
        return obs, info

    def step(
        self, action: Union[int, np.integer, float, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one simulation step of the inventory replenishment process.

        Timeline of events in period t:
            1. Arriving order Q_{t-L} is received into available inventory.
            2. New order x_t is placed and appended to the in-transit pipeline.
            3. Stochastic customer demand D_t is realized.
            4. Demand is satisfied from available stock; excess demand is Lost Sales.
            5. Ending inventory I_{t+1} and operational costs are computed.

        Args:
            action (Union[int, float]): Discrete action index or direct quantity.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                (observation, reward, terminated, truncated, info)
        """
        # Determine actual order quantity x_t
        if isinstance(action, (int, np.integer)):
            action_idx = int(action)
            order_quantity = float(self.action_levels[action_idx])
        elif isinstance(action, (float, np.floating)):
            order_quantity = float(np.clip(action, 0.0, self.max_order_quantity))
        elif isinstance(action, np.ndarray):
            order_quantity = float(np.clip(action.item(), 0.0, self.max_order_quantity))
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

        # 1. Receive arriving orders that completed lead_time
        arrived_quantity = float(self.pipeline.popleft())
        available_inventory = self.inventory + arrived_quantity

        # 2. Append newly placed order to the pipeline
        self.pipeline.append(order_quantity)

        # 3. Sample stochastic customer demand D_t
        if self.demand_distribution == "poisson":
            demand = float(self.np_random.poisson(lam=self.demand_mean))
        elif self.demand_distribution == "normal":
            demand = float(max(0.0, self.np_random.normal(loc=self.demand_mean, scale=np.sqrt(self.demand_mean))))
        else:
            demand = float(self.np_random.poisson(lam=self.demand_mean))

        # 4. Sales and Lost Sales calculation (Lost Sales Mechanics)
        sales = min(available_inventory, demand)
        lost_sales = max(0.0, demand - available_inventory)

        # Ending on-hand inventory for period t+1 (strictly non-negative)
        self.inventory = max(0.0, available_inventory - sales)
        # Storage capacity constraint
        self.inventory = min(self.inventory, self.max_inventory)

        # 5. Cost calculations
        holding_cost = self.holding_cost * self.inventory
        stockout_cost = self.lost_sales_penalty * lost_sales
        fixed_order_cost = self.fixed_ordering_cost if order_quantity > 0.0 else 0.0
        purchase_cost = self.unit_cost * order_quantity

        ordering_cost = fixed_order_cost + purchase_cost
        total_cost = holding_cost + stockout_cost + ordering_cost

        # Reward is the negative of total operating cost
        reward = -total_cost

        # Revenue and Gross Profit metrics
        revenue = self.selling_price * sales
        gross_profit = revenue - purchase_cost

        # Advance step counter
        self.current_step += 1
        self.last_demand = demand

        # Termination & Truncation
        terminated = False  # Continuing inventory horizon until episode limit
        truncated = self.current_step >= self.episode_length

        obs = self._get_obs()
        info = {
            "step": self.current_step,
            "demand": demand,
            "sales": sales,
            "lost_sales": lost_sales,
            "order_quantity": order_quantity,
            "arrived_quantity": arrived_quantity,
            "ending_inventory": self.inventory,
            "holding_cost": holding_cost,
            "stockout_cost": stockout_cost,
            "fixed_order_cost": fixed_order_cost,
            "purchase_cost": purchase_cost,
            "ordering_cost": ordering_cost,
            "total_cost": total_cost,
            "revenue": revenue,
            "gross_profit": gross_profit,
            "service_level": 1.0 if lost_sales == 0.0 else 0.0,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct the Markovian observation vector."""
        # For L=2, pipeline has length 2, but popleft() just occurred and new order appended.
        # The oldest in-transit order due next period is pipeline[0].
        in_transit_next = float(self.pipeline[0]) if len(self.pipeline) > 0 else 0.0
        time_feature = float(self.current_step % self.episode_length)

        obs = np.array(
            [self.inventory, in_transit_next, self.last_demand, time_feature],
            dtype=np.float32,
        )
        return obs

    def render(self) -> None:
        """Render the current environment state to stdout."""
        pipeline_str = ", ".join([f"{q:.0f}" for q in self.pipeline])
        print(
            f"Step {self.current_step:02d} | "
            f"Inventory: {self.inventory:5.1f} | "
            f"Pipeline: [{pipeline_str}] | "
            f"Last Demand: {self.last_demand:4.1f}"
        )
