import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy import array


def animate_map(animation_matrix: list[array[array]]):
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
