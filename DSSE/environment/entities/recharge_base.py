import random

class RechargeBase:
    def __init__(self, grid_size, position=None):
        self.grid_size = grid_size

        if position is None:
            self.position = random.choice([
                (0, 0),                                     # top-left corner
                (0, self.grid_size - 1),                    # top-right corner
                (self.grid_size - 1, 0),                    # bottom-left corner
                (self.grid_size - 1, self.grid_size - 1)    # bottom-right corner
        ])
        else:
            self.position = position

    # Get the current position of the recharge base
    def get_position(self):
        return self.position
