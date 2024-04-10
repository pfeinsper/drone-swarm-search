from gymnasium.spaces import Discrete
from .env_base import DroneSwarmSearchBase
from .constants import Actions


# TODO: Match env_base to conv_env -> If using particle sim, redo __init__ and reset.
class CoverageDroneSwarmSearch(DroneSwarmSearchBase):
    metadata = {
        "name": "DroneSwarmSearchCPP",
    }

    def __init__(
        self,
        grid_size=7,
        render_mode="ansi",
        render_grid=False,
        render_gradient=True,
        vector=(-0.5, -0.5),
        disperse_constant=10,
        timestep_limit=100,
        disaster_position=(0, 0),
        drone_amount=1,
        drone_speed=10,
        drone_probability_of_detection=0.9,
        pre_render_time=0,
    ) -> None:
        super().__init__(
            grid_size,
            render_mode,
            render_grid,
            render_gradient,
            vector,
            disperse_constant,
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

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        self.seen_states = {pos for pos in self.agents_positions.values()}
        self.not_seen_states = self.all_states - self.seen_states
        return obs, infos

    def create_observations(self):
        observations = {}
        self.probability_matrix.step(self.drone.speed)

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

        # TODO: Define the reward_scheme
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        truncations = {a: False for a in self.agents}
        prob_matrix = self.probability_matrix.get_matrix()

        for agent in self.agents:
            if agent not in actions:
                raise ValueError("Missing action for " + agent)

            drone_action = actions[agent]
            if drone_action not in self.action_space(agent):
                raise ValueError("Invalid action for " + agent)

            drone_x, drone_y = self.agents_positions[agent]
            if drone_action != Actions.SEARCH.value:
                new_position = self.move_drone((drone_x, drone_y), drone_action)
                if not self.is_valid_position(new_position):
                    rewards[agent] = self.reward_scheme["out_of_bounds"]
                else:
                    self.agents_positions[agent] = new_position
                    rewards[agent] = (
                        prob_matrix[drone_y][drone_x] * 10000
                        if prob_matrix[drone_y][drone_x] * 100 > 1
                        else -100
                    )
                    self.seen_states.add(self.agents_positions[agent])
                    self.not_seen_states.remove(self.agents_positions[agent])
            
            # Check truncation conditions (overwrites termination conditions)
            if self.timestep >= self.timestep_limit:
                rewards[agent] = self.reward_scheme["exceed_timestep"]
                truncations[agent] = True

        self.timestep += 1
        # Get dummy infos
        is_completed = len(self.not_seen_states) == 0
        infos = {drone: {"completed": is_completed} for drone in self.agents}

        self.compute_drone_collision(terminations, rewards, truncations)
        # Get observations
        observations = self.create_observations()
        # If terminted, reset the agents (pettingzoo parallel env requirement)
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def action_space(self, agent):
        return Discrete(8)
