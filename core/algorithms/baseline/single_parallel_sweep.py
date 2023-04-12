from enum import Enum


class possible_actions(Enum):
    left = 0
    right = 1
    up = 2
    down = 3
    search = 4


class SingleParallelSweep:
    def __init__(self, grid_size):
        """
        Parallel Sweep algorithm.

        The agent is the drone and starts at the bottom left corner of the grid. The goal is to
        search the person in the grid going all the way to the right, going down, and go all the
        way to the left, going down, and so on, until all the grid is searched.

        :param grid_size: The size of the grid
        """
        self.grid_size = grid_size
        self.drone_x = 0
        self.drone_y = 0

    def get_end_position(self):
        """
        Get the end position of the drone.

        :return: The end position of the drone
        """
        return self.grid_size - 1, 0 if self.grid_size % 2 == 0 else self.grid_size - 1

    def check_if_done(self):
        """
        Check if the drone is at the end position.

        :return: True if the drone is at the end position, False otherwise
        """
        end_position_x, end_position_y = self.get_end_position()
        return self.drone_x == end_position_x and self.drone_y == end_position_y

    def generate_next_movement(self):
        """
        Generate the next movement of the drone.

        :yield: The next action of the drone
        """
        if self.check_if_done():
            return

        is_going_right = True
        done = False

        while not done:
            if is_going_right:
                yield possible_actions.search
                yield possible_actions.right
                self.drone_y += 1

                if self.drone_y == self.grid_size - 1:
                    is_going_right = False
                    done = self.check_if_done()
                    if not done:
                        yield possible_actions.down
                        self.drone_x += 1
            else:
                yield possible_actions.search
                yield possible_actions.left
                self.drone_y -= 1

                if self.drone_y == 0:
                    is_going_right = True
                    done = self.check_if_done()
                    if not done:
                        yield possible_actions.down
                        self.drone_x += 1

        yield possible_actions.search

    def genarate_next_action(self):
        """
        Generate the next action of the drone.

        :yield: The next action of the drone
        """
        for action in self.generate_next_movement():
            yield {"drone": action.value}


if __name__ == "__main__":
    matrix_size = 5
    parallel_sweep = SingleParallelSweep(matrix_size)
    for action in parallel_sweep.genarate_next_action():
        print(action)
