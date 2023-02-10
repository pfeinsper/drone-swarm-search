import matplotlib.pyplot as plt
from numpy import matrix
from matplotlib.widgets import Slider


def animate_with_slider(animation_matrix: list[matrix]):
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
        matshow.set_data(animation[slider.val])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    matshow.set_data(animation[0])
    plt.show()


drone_example_matrix_1 = matrix([[1, 0, 0], [0, 0, 0], [0, 0, 2]])
drone_example_matrix_2 = matrix([[0, 1, 0], [0, 0, 0], [0, 0, 2]])
animation: list[matrix] = [drone_example_matrix_1, drone_example_matrix_2]
animate_with_slider(animation)
