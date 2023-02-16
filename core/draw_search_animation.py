import turtle


def create_grid(cell_size, grid_size, sub_divisions, goal, turt):
    # setting the image as cursor
    turt.shape("circle")
    turt.fillcolor("black")
    turt.penup()
    turt.goto(-grid_size / 2, grid_size / 2)
    turt.pendown()
    turt.speed(1000)
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
    turt.circle(10)


def draw_path(cell_size, grid_size, sub_divisions, matrix, turt):
    movements = len(matrix)

    turt.penup()
    turt.goto(-grid_size / 2 + (cell_size / 2), (grid_size / 2) - (cell_size / 2))
    turt.color("red")
    turt.fillcolor("black")
    turt.pensize(20)
    turt.pendown()
    turt.speed(1)
    turt.setheading(0)

    currX = 0
    currY = 0
    dX = 0
    dY = 0
    for m in range(0, movements):
        for y, i in enumerate(matrix[m]):
            for x, e in enumerate(i):
                if e == 1:
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
                        turt.color("blue")
                        turt.pensize(20)
                        turt.pendown()
                        turt.circle(10)
                        turt.color("red")
                        pass
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
            if j == 2:
                goal = [x, y]

    create_grid(cell_size, GRID_SIZE, sub_divisions, goal, turt)

    draw_path(cell_size, GRID_SIZE, sub_divisions, matrix, turt)

    screen.exitonclick()


##### Example and Tests
# matrix1 = [
#     [[1, 0, 0], [0, 0, 0], [0, 0, 2]],
#     [[0, 1, 0], [0, 0, 0], [0, 0, 2]],
#     [[0, 0, 0], [0, 1, 0], [0, 0, 2]],
#     [[0, 0, 0], [0, 0, 0], [0, 1, 2]],
#     [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
# ]

# matrix2 = [
#     [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]],
#     [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]],
#     [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]],
#     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 2]],
#     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]],
#     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]],
#     [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 2]],
#     [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 2]],
#     [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]],
#     [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 2]],
#     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]],
#     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
# ]
