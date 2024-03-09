from dataclasses import dataclass

@dataclass
class Circle:
    radius: int
    x0: int
    y0: int
    _increase_area: int = 0

    def increase_area(self):
        self.radius = 1 + self._increase_area
        self._increase_area += 0.5
    
    def update_center(self, new_center):
        self.x0 = new_center[1]
        self.y0 = new_center[0]