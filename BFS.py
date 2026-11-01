from collections import deque

DIRECTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

GRID_SIZE = 20
def bfs(start, end, obstacles, grid_size=GRID_SIZE):
    queue = deque([start])
    parent = {start: None}

    while queue:
        current = queue.popleft()

        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path

        cx, cy, cz = current
        for dx, dy, dz in DIRECTIONS:
            neighbor = (cx + dx, cy + dy, cz + dz)
            nx, ny, nz = neighbor

            if (0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size
                    and neighbor not in obstacles
                    and neighbor not in parent):
                parent[neighbor] = current
                queue.append(neighbor)

    return None  # путь не найден