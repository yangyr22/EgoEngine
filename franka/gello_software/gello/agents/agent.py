from typing import Any, Dict, Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: action to take on the environment.
        """
        raise NotImplementedError


class DummyAgent(Agent):
    def __init__(self, num_dofs: int):
        self.num_dofs = num_dofs

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.zeros(self.num_dofs)


class BimanualAgent(Agent):
    def __init__(self, agent_left: Agent, agent_right: Agent):
        self.agent_left = agent_left
        self.agent_right = agent_right

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        left_obs = {}
        right_obs = {}
        if obs is not None:
            for key, val in obs.items():
                L = val.shape[0]
                half_dim = L // 2
                assert L == half_dim * 2, f"{key} must be even, something is wrong"
                left_obs[key] = val[:half_dim]
                right_obs[key] = val[half_dim:]
        return np.concatenate(
            [self.agent_left.act(left_obs), self.agent_right.act(right_obs)]
        )

    def step(self, joint_positions):
        L = joint_positions.shape[-1]
        half_dim = L // 2
        assert L == half_dim * 2, f"joint_positions must be even, something is wrong"
        left_cmd = joint_positions[..., :half_dim]
        right_cmd = joint_positions[..., half_dim:]
        self.agent_left.step(left_cmd)
        self.agent_right.step(right_cmd)

    def set_torque_mode(self, mode: bool):
        self.agent_left.set_torque_mode(mode)
        self.agent_right.set_torque_mode(mode)
