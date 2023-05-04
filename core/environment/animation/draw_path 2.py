import turtle


def create_grid(cell_size, grid_size, sub_divisions, goal, turt):
    # setting the image as cursor
    turt.shape("circle")
    turt.fillcolor("black")
    turt.penup()
    turt.goto(-grid_size / 2, grid_size / 2)
    turt.pendown()
    turt.speed(100000)
    angle = 90
    for _ in range(4):
        turt.forward(grid_size)
        turt.right(angle)
    for _ in range(2):
        for _ in range(1, sub_divisions):
            turt.forward(cell_size)
            turt.right(angle)
            turt.forward(grid_size)
            turt.left(angle)
            angle = -angle
        turt.forward(cell_size)
        turt.right(angle)

    turt.penup()
    turt.goto(
        ((-grid_size / 2 + (cell_size / 2))) + (cell_size * goal[0]),
        ((grid_size / 2) - (cell_size / 2)) - (cell_size * goal[1]),
    )
    turt.color("black")
    turt.pensize(20)
    turt.pendown()
    turt.circle(cell_size / 10)


def cell_search_anim(turt, cell_size):
    turt.color("yellow")
    turt.pensize(cell_size / 50)
    turt.pendown()
    turt.forward(cell_size / 2)
    turt.backward(cell_size)
    turt.forward(cell_size / 2)
    turt.right(90)
    turt.forward(cell_size / 2)
    turt.backward(cell_size)
    turt.forward(cell_size / 2)
    turt.right(-90)
    turt.color("red")


def draw_path(cell_size, grid_size, sub_divisions, matrix, turt):
    movements = len(matrix)

    turt.penup()
    turt.goto(-grid_size / 2 + (cell_size / 2), (grid_size / 2) - (cell_size / 2))
    turt.color("red")
    turt.fillcolor("black")
    turt.pensize(cell_size / 50)
    turt.pendown()
    turt.speed(50)
    turt.setheading(0)

    currX = 0
    currY = 0
    dX = 0
    dY = 0
    for m in range(0, movements):
        for y, i in enumerate(matrix[m]):
            for x, e in enumerate(i):
                if e == "X" or e == "PX" or e == "F":
                    dX = x - currX
                    dY = y - currY

                    if dX < 0 and dY == 0:
                        turt.backward(cell_size)
                    elif dX > 0 and dY == 0:
                        turt.forward(cell_size)
                    elif dY < 0 and dX == 0:
                        turt.right(-90)
                        turt.forward(cell_size)
                        turt.right(90)
                    elif dY > 0 and dX == 0:
                        turt.right(90)
                        turt.forward(cell_size)
                        turt.right(-90)
                    elif dY > 0 and dX > 0:
                        turt.right(45)
                        turt.forward(cell_size * (2 ** (1 / 2)))
                        turt.right(-45)
                    elif dY > 0 and dX < 0:
                        turt.right(-45)
                        turt.backward(cell_size * (2 ** (1 / 2)))
                        turt.right(45)
                    elif dY < 0 and dX < 0:
                        turt.right(45)
                        turt.backward(cell_size * (2 ** (1 / 2)))
                        turt.right(-45)
                    elif dY < 0 and dX > 0:
                        turt.right(-45)
                        turt.forward(cell_size * (2 ** (1 / 2)))
                        turt.right(45)
                    else:
                        cell_search_anim(turt, cell_size)
                    currX = x
                    currY = y


def create_search_animation(matrix):
    GRID_SIZE = 600
    sub_divisions = len(matrix[0])
    cell_size = GRID_SIZE / float(sub_divisions)

    screen = turtle.Screen()
    screen.bgcolor("#3260a8")
    turt = turtle.Turtle()

    goal = [0, 0]
    for y, e in enumerate(matrix[0]):
        for x, j in enumerate(e):
            if j == "P":
                goal = [x, y]

    create_grid(cell_size, GRID_SIZE, sub_divisions, goal, turt)

    draw_path(cell_size, GRID_SIZE, sub_divisions, matrix, turt)

    screen.exitonclick()


# matrix2 = [
#     [
#         ["X", "0", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "P"],
#     ],
#     [
#         ["0", "X", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "P"],
#     ],
#     [
#         ["0", "0", "0", "0"],
#         ["X", "0", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "P"],
#     ],
#     [
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "X", "0", "0"],
#         ["0", "0", "0", "P"],
#     ],
#     [
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "X", "0"],
#         ["0", "0", "0", "P"],
#     ],
#     [
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "X"],
#         ["0", "0", "0", "P"],
#     ],
#     [
#         ["0", "0", "0", "0"],
#         ["0", "0", "X", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "P"],
#     ],
#     [
#         ["0", "0", "0", "0"],
#         ["0", "0", "X", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "P"],
#     ],
#     [
#         ["0", "0", "0", "X"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "P"],
#     ],
#     [
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "X"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "P"],
#     ],
#     [
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "X"],
#         ["0", "0", "0", "P"],
#     ],
#     [
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "0"],
#         ["0", "0", "0", "X"],
#     ],
# ]
# create_search_animation(matrix2)
