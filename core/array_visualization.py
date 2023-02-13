import matplotlib.pyplot as plt
from numpy import array
from matplotlib.widgets import Slider


def animate_with_slider(animation_matrix: list[array]):
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
    matshow = ax.matshow(animation_matrix)

    def update(_):
        matshow.set_data(animation_matrix[slider.val])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    matshow.set_data(animation_matrix[0])
    plt.show()


# Example usage:
# drone_example_matrix_1 = array([[1, 0, 0], [0, 0, 0], [0, 0, 2]])
# drone_example_matrix_2 = array([[0, 1, 0], [0, 0, 0], [0, 0, 2]])
# animation: list[array] = [drone_example_matrix_1, drone_example_matrix_2]
# animate_with_slider(animation)
