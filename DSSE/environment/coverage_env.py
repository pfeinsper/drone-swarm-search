from gymnasium.spaces import Discrete
from .env_base import DroneSwarmSearchBase
from .constants import Actions, Reward


# TODO: Match env_base to conv_env -> If using particle sim, redo __init__ and reset.
class CoverageDroneSwarmSearch(DroneSwarmSearchBase):
    metadata = {
        "name": "DroneSwarmSearchCPP",
    }
    reward_scheme = Reward(
        default=0,
        leave_grid=-100,
        exceed_timestep=-100,
        drones_collision=-100,
        search_cell=0,
        search_and_find=10000,
    )

    def __init__(
        self,
        grid_size=7,
        render_mode="ansi",
        render_grid=True,
        render_gradient=True,
        vector=(3.1, 3.2),
        dispersion_inc=0.1,
        dispersion_start=0.5,
        timestep_limit=100,
        disaster_position=(0, 0),
        drone_amount=1,
        drone_speed=10,
        drone_probability_of_detection=0.9,
        pre_render_time=10,
    ) -> None:
        super().__init__(
            grid_size,
            render_mode,
            render_grid,
            render_gradient,
            vector,
            dispersion_inc,
            dispersion_start,
            timestep_limit,
            disaster_position,
            drone_amount,
            drone_speed,
            drone_probability_of_detection,
            pre_render_time,
        )
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
        self.seen_states = {pos for pos in self.agents_positions.values()}
        zero_states = np.where(self.probability_matrix.get_matrix() == 0)
        print(zero_states)
        self.not_seen_states = self.all_states - self.seen_states
        infos = self.compute_infos(False)
        self.cumm_pos = 0
        self.repeated_coverage = 0
        return obs, infos

    def create_observations(self):
        observations = {}
        # self.probability_matrix.step(self.drone.speed)

        probability_matrix = self.probability_matrix.get_matrix()
        for agent in self.agents:
            observation = (
                (self.agents_positions[agent][0], self.agents_positions[agent][1]),
                probability_matrix,
            )
            observations[agent] = observation

        return observations

    def pre_search_simulate(self):
        for _ in range(self.pre_render_steps):
            self.probability_matrix.step(self.drone.speed)

    def step(self, actions: dict[str, int]) -> tuple:
        if not self._was_reset:
            raise ValueError("Please reset the env before interacting with it")

        terminations = {a: False for a in self.agents}
        rewards = {a: self.reward_scheme.default for a in self.agents}
        truncations = {a: False for a in self.agents}
        self.timestep += 1

        prob_matrix = self.probability_matrix.get_matrix()
        for agent in self.agents:
            if agent not in actions:
                raise ValueError("Missing action for " + agent)

            drone_action = actions[agent]
            if drone_action not in self.action_space(agent):
                raise ValueError("Invalid action for " + agent)

            if self.timestep >= self.timestep_limit:
                rewards[agent] = self.reward_scheme.exceed_timestep
                truncations[agent] = True
                continue

            drone_x, drone_y = self.agents_positions[agent]
            new_position = self.move_drone((drone_x, drone_y), drone_action)
            if not self.is_valid_position(new_position):
                rewards[agent] = self.reward_scheme.leave_grid
                continue

            self.agents_positions[agent] = new_position
            new_x, new_y = new_position
            if new_position in self.not_seen_states:
                reward_poc = 1 / (self.timestep) * prob_matrix[new_y, new_x] * 1_000
                rewards[agent] = self.reward_scheme.search_cell + reward_poc
                self.seen_states.add(new_position)
                self.not_seen_states.remove(new_position)
                # Probability of sucess (POS) = POC * POD
                self.cumm_pos += prob_matrix[new_y, new_x] * self.pod
            else:
                self.repeated_coverage += 1

        # Get dummy infos
        is_completed = len(self.not_seen_states) == 0
        self.render()
        if is_completed:
            # TODO: Proper define reward for completing the search (R_done)
            rewards = {
                drone: self.reward_scheme.search_and_find for drone in self.agents
            }
            terminations = {drone: True for drone in self.agents}
        infos = self.compute_infos(is_completed)

        self.compute_drone_collision(terminations, rewards, truncations)
        # Get observations
        observations = self.create_observations()
        # If terminted, reset the agents (pettingzoo parallel env requirement)
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def compute_infos(self, is_completed: bool) -> dict[str, dict]:
        # TODO: Is this the best way to inform the coverage rate, Cum_pos and repetitions?
        coverage_rate = len(self.seen_states) / len(self.all_states)
        infos = {
            "is_completed": is_completed,
            "coverage_rate": coverage_rate,
            "repeated_coverage": self.repeated_coverage / len(self.all_states),
            "acumulated_pos": self.cumm_pos,
        }
        return {drone: infos for drone in self.agents}

    def action_space(self, agent):
        return Discrete(8)
