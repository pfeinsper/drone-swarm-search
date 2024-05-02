from gymnasium.spaces import Discrete
from .env_base import DroneSwarmSearchBase
from .simulation.particle_simulation import ParticleSimulation
from .constants import Reward
import numpy as np
import functools


class CoverageDroneSwarmSearch(DroneSwarmSearchBase):
    metadata = {
        "name": "DroneSwarmSearchCPP",
    }
    reward_scheme = Reward(
        default=0,
        leave_grid=-10,
        exceed_timestep=-100,
        drones_collision=-10,
        search_cell=10,
        search_and_find=100,
    )

    def __init__(
        self,
        render_mode="ansi",
        render_grid=True,
        render_gradient=True,
        timestep_limit=100,
        disaster_position=(-24.04, -46.17),
        drone_amount=1,
        drone_speed=10,
        drone_probability_of_detection=0.9,
        pre_render_time=10,
        prob_matrix_path=None,
    ) -> None:

        # Prob matrix
        self.probability_matrix = ParticleSimulation(
            disaster_lat=disaster_position[0],
            disaster_long=disaster_position[1],
            duration_hours=pre_render_time,
        )
        if prob_matrix_path is not None:
            if not isinstance(prob_matrix_path, str):
                raise ValueError("prob_matrix_path must be a string")
            self.probability_matrix.load_state(prob_matrix_path)
        else: 
            self.probability_matrix.run_or_get_simulation()
        grid_size = self.probability_matrix.get_map_size()

        super().__init__(
            grid_size=grid_size,
            render_mode=render_mode,
            render_grid=render_grid,
            render_gradient=render_gradient,
            timestep_limit=timestep_limit,
            drone_amount=drone_amount,
            drone_speed=drone_speed,
            probability_of_detection=drone_probability_of_detection,
        )
        self.disaster_position = disaster_position
        # Sets used to keep track of the seen and not seen states for reward calculation
        self.seen_states = None
        self.not_seen_states = None
        self.all_states = {
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        }
        self.repeated_coverage = 0
        self.cumm_pos = 0

    def reset(self, seed=None, options=None):
        obs, _ = super().reset(seed=seed, options=options)

        self.reset_search_state()

        self.cumm_pos = 0
        self.repeated_coverage = 0
        infos = self.compute_infos(False)
        return obs, infos

    def reset_search_state(self):
        # This is in (x, y)
        self.seen_states = {pos for pos in self.agents_positions}

        mat = self.probability_matrix.get_matrix()
        # (row, col)
        close_to_zero = np.argwhere(np.abs(mat) < 1e-10)

        # Remove the need to visit cells with POC near to 0
        for y, x in close_to_zero:
            self.seen_states.add((x, y))

        self.not_seen_states = self.all_states - self.seen_states

    def create_observations(self):
        observations = {}

        probability_matrix = self.probability_matrix.get_matrix()
        for idx, agent in enumerate(self.agents):
            observation = (
                self.agents_positions[idx],
                probability_matrix,
            )
            observations[agent] = observation

        return observations

    def pre_search_simulate(self):
        self.probability_matrix.run_or_get_simulation()

    def step(self, actions: dict[str, int]) -> tuple:
        if not self._was_reset:
            raise ValueError("Please reset the env before interacting with it")

        terminations = {a: False for a in self.agents}
        rewards = {a: self.reward_scheme.default for a in self.agents}
        truncations = {a: False for a in self.agents}
        self.timestep += 1

        prob_matrix = self.probability_matrix.get_matrix()
        for idx, agent in enumerate(self.agents):
            if agent not in actions:
                raise ValueError("Missing action for " + agent)

            drone_action = actions[agent]
            if drone_action not in self.action_space(agent):
                raise ValueError("Invalid action for " + agent)


            if self.timestep >= self.timestep_limit:
                rewards[agent] = self.reward_scheme.exceed_timestep
                truncations[agent] = True
                continue

            # Action 8 is to stay in the same position, default reward.
            if drone_action == 8:
                continue
        
            drone_x, drone_y = self.agents_positions[idx]
            new_position = self.move_drone((drone_x, drone_y), drone_action)
            if not self.is_valid_position(new_position):
                rewards[agent] = self.reward_scheme.leave_grid
                continue

            self.agents_positions[idx] = new_position
            new_x, new_y = new_position
            if new_position in self.not_seen_states:
                reward_poc = (1 / (self.timestep)) * prob_matrix[new_y, new_x] * 1_000
                rewards[agent] = self.reward_scheme.search_cell + reward_poc
                self.seen_states.add(new_position)
                self.not_seen_states.remove(new_position)
                # Probability of sucess (POS) = POC * POD
                self.cumm_pos += prob_matrix[new_y, new_x] * self.drone.pod
                # Remove the probability of the visited cell.
                prob_matrix[new_y, new_x] = 0.0
            else:
                self.repeated_coverage += 1

        # Get dummy infos
        is_completed = len(self.not_seen_states) == 0
        if self.render_mode == "human":
            self.render()

        if is_completed:
            # TODO: Proper define reward for completing the search (R_done)
            rewards = {
                drone: self.reward_scheme.search_and_find for drone in self.agents
            }
            terminations = {drone: True for drone in self.agents}
        infos = self.compute_infos(is_completed)

        self.compute_drone_collision(terminations, rewards)
        # Get observations
        observations = self.create_observations()
        # If terminted, reset the agents (pettingzoo parallel env requirement)
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def compute_infos(self, is_completed: bool) -> dict[str, dict]:
        # TODO: Is this the best way to inform the coverage rate, Cum_pos and repetitions?
        total_states = len(self.seen_states) + len(self.not_seen_states)
        coverage_rate = len(self.seen_states) / total_states
        infos = {
            "is_completed": is_completed,
            "coverage_rate": coverage_rate,
            "repeated_coverage": self.repeated_coverage / total_states,
            "acumulated_pos": self.cumm_pos,
        }
        return {drone: infos for drone in self.agents}
    
    def save_matrix(self, path: str):
        self.probability_matrix.save_state(path)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(9)
