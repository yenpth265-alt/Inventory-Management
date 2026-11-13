"""Classical Inventory Control Policies for Benchmarking.

Includes:
1. RandomPolicy: Uniformly samples feasible discrete ordering actions.
2. BaseStockPolicy: Order-up-to level S policy.
3. sSPolicy: Classical Scarf (s, S) policy with reorder point s and target level S.
"""

from typing import List, Optional, Union
import numpy as np


class RandomPolicy:
    """Random policy that samples actions uniformly from the action space."""

    def __init__(self, action_levels: List[int], seed: Optional[int] = None) -> None:
        """Initialize random policy.

        Args:
            action_levels (List[int]): List of discrete order quantities.
            seed (Optional[int]): Random seed.
        """
        self.action_levels = action_levels
        self.num_actions = len(action_levels)
        self.rng = np.random.default_rng(seed)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Select a random action index.

        Args:
            observation (np.ndarray): Current environment observation vector.
            deterministic (bool): Ignored for random policy.

        Returns:
            int: Action index corresponding to action_levels.
        """
        return int(self.rng.integers(0, self.num_actions))


class BaseStockPolicy:
    """Base-Stock (Order-up-to S) Inventory Control Policy.

    Logic:
        Inventory Position IP_t = On-hand Inventory + Sum(In-transit Pipeline)
        Desired Order Quantity x_t = max(0, S - IP_t)
        Action Index = Nearest level in action_levels.
    """

    def __init__(self, target_order_up_to_S: float, action_levels: List[int]) -> None:
        """Initialize Base-Stock policy.

        Args:
            target_order_up_to_S (float): Target inventory position level S.
            action_levels (List[int]): Feasible discrete order quantities.
        """
        self.S = float(target_order_up_to_S)
        self.action_levels = np.array(action_levels, dtype=np.float32)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Select order quantity action index to bring inventory position to S.

        Args:
            observation (np.ndarray): [On-hand, In-transit, Last Demand, Time]
            deterministic (bool): Deterministic flag (always True for classical policy).

        Returns:
            int: Action index corresponding to action_levels.
        """
        on_hand = float(observation[0])
        in_transit = float(observation[1])
        inventory_position = on_hand + in_transit

        desired_quantity = max(0.0, self.S - inventory_position)

        # Snap to nearest discrete action level
        action_idx = int(np.argmin(np.abs(self.action_levels - desired_quantity)))
        return action_idx


class sSPolicy:
    """Classical (s, S) Inventory Policy with fixed ordering cost.

    Logic:
        Inventory Position IP_t = On-hand Inventory + Sum(In-transit Pipeline)
        If IP_t < s:
            Place order x_t = S - IP_t
        Else:
            x_t = 0
        Action Index = Nearest level in action_levels.
    """

    def __init__(
        self, reorder_point_s: float, order_up_to_S: float, action_levels: List[int]
    ) -> None:
        """Initialize (s, S) policy.

        Args:
            reorder_point_s (float): Reorder threshold level s.
            order_up_to_S (float): Target inventory position level S (S > s).
            action_levels (List[int]): Feasible discrete order quantities.
        """
        if order_up_to_S <= reorder_point_s:
            raise ValueError(f"S ({order_up_to_S}) must be strictly greater than s ({reorder_point_s}).")

        self.s = float(reorder_point_s)
        self.S = float(order_up_to_S)
        self.action_levels = np.array(action_levels, dtype=np.float32)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Select action index based on (s, S) threshold rule.

        Args:
            observation (np.ndarray): [On-hand, In-transit, Last Demand, Time]
            deterministic (bool): Deterministic flag.

        Returns:
            int: Action index corresponding to action_levels.
        """
        on_hand = float(observation[0])
        in_transit = float(observation[1])
        inventory_position = on_hand + in_transit

        if inventory_position < self.s:
            desired_quantity = self.S - inventory_position
        else:
            desired_quantity = 0.0

        # Snap to nearest discrete action level
        action_idx = int(np.argmin(np.abs(self.action_levels - desired_quantity)))
        return action_idx
