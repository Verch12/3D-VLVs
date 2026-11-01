import random

DIRECTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

def generate_maze_3d(GRID_SIZE, seed=67):
    rng = random.Random(seed)

    start_pos = (0, 0, 0)
    end_pos = (GRID_SIZE - 1, GRID_SIZE - 1, GRID_SIZE - 1)

    obstacle_cells = set()

    # Количество палок
    rods_count = round(GRID_SIZE**1.8)

    for _ in range(rods_count):
        axis = rng.choice(["x", "y"])
        length = GRID_SIZE
        x = rng.randrange(GRID_SIZE)
        y = rng.randrange(GRID_SIZE)
        z = rng.randrange(GRID_SIZE)

        if axis == "x":
            length = min(length, GRID_SIZE - x)
            for i in range(length):
                obstacle_cells.add((x + i, y, z))
        else:
            length = min(length, GRID_SIZE - y)
            for i in range(length):
                obstacle_cells.add((x, y + i, z))

    obstacle_cells.discard(start_pos)
    obstacle_cells.discard(end_pos)

    return start_pos, end_pos, obstacle_cells