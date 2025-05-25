from DSSE import DroneSwarmSearch
import numpy as np
from DSSE.environment.constants import Actions
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple

class EnergyAwareDroneSwarmSearch(DroneSwarmSearch):
    """Energy-aware version of DroneSwarmSearch with modified observation and reward structure"""
    
    def __init__(
        self,
        grid_size=40,
        render_mode="human",
        render_grid=True,
        render_gradient=True,
        vector=(1, 1),
        timestep_limit=300,
        person_amount=1,
        dispersion_inc=0.05,
        person_initial_position=(15, 15),
        drone_amount=2,
        drone_speed=10,
        probability_of_detection=0.9,
        pre_render_time=0,
        energy_penalty_factor=1,  # Reduced from 0.1 to be less punishing
        distance_reward_factor=0.1,  # Increased from 0.05 to encourage exploration
        recharge_reward=10.0,  # Increased significantly to make recharging more attractive
        low_battery_threshold=30,  # Increased to encourage earlier recharging
        battery_emergency_threshold=15,  # Emergency threshold for critical battery level
        base_return_factor=2.0,  # New factor for base return rewards
    ):
        super().__init__(
            grid_size=grid_size,
            render_mode=render_mode,
            render_grid=render_grid,
            render_gradient=render_gradient,
            vector=vector,
            timestep_limit=timestep_limit,
            person_amount=person_amount,
            dispersion_inc=dispersion_inc,
            person_initial_position=person_initial_position,
            drone_amount=drone_amount,
            drone_speed=drone_speed,
            probability_of_detection=probability_of_detection,
            pre_render_time=pre_render_time,
        )
        
        self.energy_penalty_factor = energy_penalty_factor
        self.distance_reward_factor = distance_reward_factor
        self.recharge_reward = recharge_reward
        self.low_battery_threshold = low_battery_threshold
        self.battery_emergency_threshold = battery_emergency_threshold
        self.base_return_factor = base_return_factor
        
        # Additional metrics tracking
        self.episode_metrics = {
            'targets_found': 0,
            'energy_consumed': 0,
            'recharge_count': 0,
            'steps_with_low_battery': 0,
            'successful_searches': 0,
        }
        
    def get_normalized_observation(self, agent_idx):
        """Convert raw observations to normalized form suitable for PPO"""
        position = self.agents_positions[agent_idx]
        prob_matrix = self.probability_matrix.get_matrix()
        battery_level = self.drone.get_battery(agent_idx) / 100.0  # Normalize battery
        
        # Get distance to recharge base
        base_pos = self.recharge_base.get_position()
        distance_to_base = np.array([
            (position[0] - base_pos[0]) / self.grid_size,
            (position[1] - base_pos[1]) / self.grid_size
        ])
        
        # Add battery status flags
        battery_status = np.zeros(3)  # [normal, low, emergency]
        if battery_level <= self.battery_emergency_threshold/100.0:
            battery_status[2] = 1.0  # emergency
        elif battery_level <= self.low_battery_threshold/100.0:
            battery_status[1] = 1.0  # low
        else:
            battery_status[0] = 1.0  # normal
        
        # Flatten and normalize position
        normalized_pos = np.array([
            position[0] / self.grid_size,
            position[1] / self.grid_size
        ])
        
        return {
            'position': normalized_pos,
            'probability_matrix': prob_matrix,
            'battery_level': np.array([battery_level]),
            'battery_status': battery_status,
            'distance_to_base': distance_to_base
        }
        
    def calculate_energy_aware_reward(self, agent, base_reward, action, old_pos, new_pos):
        """Calculate energy-aware reward components with improved battery management"""
        reward = base_reward
        battery_level = self.drone.get_battery(agent)
        base_pos = self.recharge_base.get_position()
        prob_matrix = self.probability_matrix.get_matrix()
        
        # Get probability values
        old_prob = prob_matrix[old_pos[1], old_pos[0]]
        new_prob = prob_matrix[new_pos[1], new_pos[0]]
        prob_improvement = new_prob - old_prob
        
        # Calculate distances
        old_distance_to_base = abs(old_pos[0] - base_pos[0]) + abs(old_pos[1] - base_pos[1])
        new_distance_to_base = abs(new_pos[0] - base_pos[0]) + abs(new_pos[1] - base_pos[1])
        distance_improvement = old_distance_to_base - new_distance_to_base
        
        # Battery management rewards
        if battery_level <= self.battery_emergency_threshold:
            # Emergency battery situation
            if new_pos == base_pos:
                # Big reward for reaching base in emergency
                reward += self.recharge_reward * 3.0
            else:
                # Strong penalty based on distance from base
                reward -= self.base_return_factor * new_distance_to_base
                # Reward for moving towards base
                if distance_improvement > 0:
                    reward += self.base_return_factor * distance_improvement
                
        elif battery_level <= self.low_battery_threshold:
            # Low battery situation
            if new_pos == base_pos:
                # Good reward for reaching base when low
                reward += self.recharge_reward
            else:
                # Moderate penalty based on distance from base
                reward -= self.base_return_factor * 0.5 * new_distance_to_base
                # Reward for moving towards base
                if distance_improvement > 0:
                    reward += self.base_return_factor * 0.5 * distance_improvement
        
        # Movement and exploration rewards (only if battery isn't critical)
        if battery_level > self.battery_emergency_threshold:
            if action != Actions.SEARCH.value:
                # Reward for moving towards higher probability areas
                if prob_improvement > 0:
                    reward += prob_improvement * self.distance_reward_factor * 2.0
                
                # Small reward for being in high probability areas
                if new_prob > 0.5:
                    reward += 0.5
            else:
                # Extra reward for searching in high probability areas
                if new_prob > 0.5:
                    reward += 1.0
        
        # Base energy penalty
        energy_penalty = -self.energy_penalty_factor * (1.0 - battery_level/100.0)
        reward += energy_penalty
        
        # Extra reward for successful search actions
        if action == Actions.SEARCH.value and base_reward > 0:
            reward *= 1.5
        
        return reward
        
    def step(self, actions):
        """Override step to include energy-aware rewards and metrics tracking"""
        old_positions = self.agents_positions.copy()
        
        # Store person positions before movement
        old_person_positions = [(person.x, person.y) for person in self.persons_set]
        
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # Verify person movement is correct (not towards recharge base)
        for person, old_pos in zip(list(self.persons_set), old_person_positions):
            new_pos = (person.x, person.y)
            if new_pos == old_pos:  # If person hasn't moved, ensure they move according to their vector
                movement_map = self.build_movement_matrix(person)
                person.step(movement_map)
        
        # Modify rewards to include energy awareness
        for idx, agent in enumerate(self.agents):
            if agent in rewards:  # Check if agent still active
                old_pos = old_positions[idx]
                new_pos = self.agents_positions[idx]
                rewards[agent] = self.calculate_energy_aware_reward(
                    idx, rewards[agent], actions[agent], old_pos, new_pos
                )
                
                # Update metrics
                self.episode_metrics['energy_consumed'] += 1
                if self.drone.get_battery(idx) <= self.low_battery_threshold:
                    self.episode_metrics['steps_with_low_battery'] += 1
                if new_pos == self.recharge_base.get_position():
                    self.episode_metrics['recharge_count'] += 1
        
        # Update normalized observations
        normalized_obs = {
            agent: self.get_normalized_observation(idx)
            for idx, agent in enumerate(self.agents)
        }
        
        return normalized_obs, rewards, terminations, truncations, infos
    
    def reset(self, seed=None, options=None):
        """Reset environment and metrics, ensuring drones start with full battery and at different positions"""
        # Set random but diverse starting positions for drones
        grid_size = self.grid_size
        positions = []
        
        # Divide grid into regions for each drone
        regions = []
        n_regions = int(np.ceil(np.sqrt(self.drone.amount)))
        region_size = grid_size // n_regions
        
        for i in range(n_regions):
            for j in range(n_regions):
                if len(regions) < self.drone.amount:
                    regions.append((
                        i * region_size,
                        j * region_size,
                        min((i + 1) * region_size, grid_size),
                        min((j + 1) * region_size, grid_size)
                    ))
        
        # Place each drone randomly within its region
        for x1, y1, x2, y2 in regions:
            pos = (
                np.random.randint(x1, x2),
                np.random.randint(y1, y2)
            )
            positions.append(pos)
        
        # Set options for drone positions
        if options is None:
            options = {}
        options['drones_positions'] = positions
        
        # Set person movement vector (random direction but not towards bottom right)
        angle = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
        vector_x = np.cos(angle)
        vector_y = np.sin(angle)
        options['vector'] = (vector_x, vector_y)
        
        observations, info = super().reset(seed=seed, options=options)
        
        # Reset metrics
        self.episode_metrics = {
            'targets_found': 0,
            'energy_consumed': 0,
            'recharge_count': 0,
            'steps_with_low_battery': 0,
            'successful_searches': 0,
        }
        
        # Ensure all drones start with full battery
        for i in range(self.drone.amount):
            self.drone.batteries[i] = self.drone.battery_capacity
        
        # Convert to normalized observations
        normalized_obs = {
            agent: self.get_normalized_observation(idx)
            for idx, agent in enumerate(self.agents)
        }
        
        return normalized_obs, info
