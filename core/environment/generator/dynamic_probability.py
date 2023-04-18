from numpy import array, zeros
from random import randint, shuffle
import copy
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def animate_map(animation_matrix):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25, left=0.25)

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(
        ax_slider,
        label="Frame",
        valmin=0,
        valmax=len(animation_matrix) - 1,
        valinit=0,
        valstep=1,
    )
    matshow = ax.matshow(animation_matrix[0])

    def update(_):
        # print(animation_matrix[slider.val].max())
        matshow.set_data(animation_matrix[slider.val])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    matshow.set_data(animation_matrix[0])
    plt.show()


def diffuse_probability(map: array, vector: tuple):
    x = vector[0] * 10
    y = vector[1] * 10
    map_copy = zeros((len(map), len(map[0])), dtype=float)
    for row in range(len(map)):
        for column in range(0, len(map[row])):
            if map[row][column] > 0:
                counter_x = int(x)
                counter_y = int(y)
                while abs(counter_x) >= 0:
                    while abs(counter_y) >= 0:
                        new_row = row + counter_y
                        new_col = column + counter_x
                        if (
                            (new_row < len(map))
                            and (new_col < len(map[row]))
                            and (new_row >= 0)
                            and (new_col >= 0)
                        ):
                            if counter_x != 0 and counter_y != 0:
                                divisor = 10 / (
                                    1 / ((counter_x**2 + counter_y**2) ** 0.5)
                                )
                            else:
                                divisor = (x**2 + y**2) ** 0.5
                            probability = map[row][column] / divisor
                            map_copy[row + counter_y][column + counter_x] += probability
                            if map_copy[row + counter_y][column + counter_x] >= 50:
                                map_copy[row + counter_y][
                                    column + counter_x
                                ] -= probability
                            # map[row][column] -= probability
                        if counter_y == 0:
                            break
                        if counter_y < 0:
                            counter_y += 1
                        else:
                            counter_y -= 1
                    if counter_x == 0:
                        break
                    if counter_x < 0:
                        counter_x += 1
                    else:
                        counter_x -= 1
                    counter_y = int(y)
    return map_copy


def dynamic_probability(map: array, vector: tuple, x: float, y: float):
    if abs(x) >= 1:
        x = 0
    if abs(y) >= 1:
        y = 0
    x += vector[0]
    y += vector[1]
    # map_copy = copy.deepcopy(map)
    map_copy = zeros((len(map), len(map[0])), dtype=float)
    for row in range(len(map)):
        for column in range(0, len(map[row])):
            if map[row][column] > 0:
                if row + int(y) < len(map) and column + int(x) < len(map[row]):
                    map_copy[row + int(y)][column + int(x)] += map[row][column]

    new_map = diffuse_probability(map_copy, vector)

    return new_map, x, y


# map = [
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         8.0,
#         12.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         10.0,
#         25.0,
#         30.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         7.0,
#         15.0,
#         13.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         5.0,
#         6.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
# ]


# print(len(map), len(map[0]))
# x = 0
# y = 0
# vector = (-0.1, 0.3)
# list_map = []
# map = array(map)
# list_map.append(map)


# for i in range(50):
#     new_map, new_x, new_y = dynamic_probability(map, vector, x, y)
#     map = copy.deepcopy(new_map)
#     x = copy.deepcopy(new_x)
#     y = copy.deepcopy(new_y)
#     list_map.append(map)


# animate_map(list_map)
