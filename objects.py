import numpy as np
from OpenGL.GL import *
from matrices import *

# ---------------------------------------------------------------------------
# Камера
# ---------------------------------------------------------------------------

class OrbitCamera:
    MIN_DIST, MAX_DIST = 2.5, 100.0
    MAX_PITCH = np.radians(89.0)

    def __init__(self):
        self.target = np.zeros(3, dtype=np.float32)
        self.sensitivity = 0.005
        self.reset()

    def reset(self):
        self.yaw = 0.6
        self.pitch = 0.4
        self.distance = 12.0
        self.setX = 0
        self.setY = 0

    def rotate(self, dx, dy):
        self.yaw -= dx * self.sensitivity
        self.pitch += dy * self.sensitivity
        self.pitch = float(np.clip(self.pitch, -self.MAX_PITCH, self.MAX_PITCH))

    def translate(self, dx, dy):
        self.setX -= dx * self.sensitivity
        self.setY += dy * self.sensitivity

    def zoom(self, amount):
        self.distance *= 0.9 ** amount
        self.distance = float(np.clip(self.distance, self.MIN_DIST, self.MAX_DIST))

    def view_matrix(self):
        offset = self.distance * np.array([
            np.cos(self.pitch) * np.sin(self.yaw),
            np.sin(self.pitch),
            np.cos(self.pitch) * np.cos(self.yaw),
        ], dtype=np.float32)

        forward = -offset / np.linalg.norm(offset)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 0.001:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        pan_factor = self.distance * 0.1 
        offset_vector = (right * self.setX * pan_factor) + (up * self.setY * pan_factor)
        new_target = self.target + offset_vector
        camera_pos = new_target + offset

        return look_at(camera_pos, new_target, world_up)

# ---------------------------------------------------------------------------
# Меш: геометрия, загруженная в видеопамять (создаётся один раз на форму)
# ---------------------------------------------------------------------------

class Mesh:
    def __init__(self, vertices, indices, draw_mode):
        """vertices: плоский список [x, y, z, r, g, b, a, ...]; indices: номера вершин."""
        self.vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        self.index_count = len(indices)
        self.draw_mode = draw_mode

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        stride = 7 * self.vertices.itemsize
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride,
                              ctypes.c_void_p(3 * self.vertices.itemsize))
        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(self.draw_mode, self.index_count, GL_UNSIGNED_INT, None)

    def delete(self):
        glDeleteBuffers(2, [self.vbo, self.ebo])
        glDeleteVertexArrays(1, [self.vao])

# ---------------------------------------------------------------------------
# Пакетные линии: геометрия, дублькаты, загруженная в видеопамять (создаётся один раз на форму)
# ---------------------------------------------------------------------------

class BatchedLines:
    def __init__(self, base_mesh, scaled):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        positions = [
            (0, y, z)
            for y in range(scaled)
            for z in range(scaled)
            ]

        base = np.asarray(base_mesh.vertices, dtype=np.float32)
        base = base.reshape(-1, 7)  # x, y, z, r, g, b, a
        
        all_vertices = []

        for rot in range(3):
            for pos in positions:
                offset = np.array(pos, dtype=np.float32)
                moved = base.copy()
                moved[:, 0:3] *= np.full(3, scaled-1, dtype=np.float32)
                moved[:, 0:3] += offset
                moved[:, 0:3] = np.roll(moved[:, 0:3], shift=rot, axis=1)
                moved[:, 0:3] -= np.full(3, 0.5, dtype=np.float32)
                all_vertices.append(moved)

        vertices = np.concatenate(all_vertices, axis=0)
        self.vertex_count = len(vertices)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            vertices.nbytes,
            vertices,
            GL_STATIC_DRAW
        )

        stride = 7 * vertices.itemsize

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, stride,
            ctypes.c_void_p(0)
        )

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1, 4, GL_FLOAT, GL_FALSE, stride,
            ctypes.c_void_p(3 * vertices.itemsize)
        )

        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINES, 0, self.vertex_count)

    def delete(self):
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])

# ---------------------------------------------------------------------------
# Объект сцены: меш + положение, размер и вращение
# ---------------------------------------------------------------------------

class SceneObject:
    def __init__(self, mesh, position=(0, 0, 0), size=(1, 1, 1), spin=(0.0, 0.0), new_color=()):
        self.mesh = mesh
        self.position = position
        self.new_color = new_color
        self.size = size
        self.spin_x, self.spin_y = spin     # скорости вращения, рад/с
        self.angle_x = 0.0
        self.angle_y = 0.0

    def update(self, dt):
        self.angle_x += self.spin_x * dt
        self.angle_y += self.spin_y * dt


    def model_matrix(self):
        # Порядок справа налево: сначала масштаб, потом поворот, потом перенос
        return (translate(*self.position)
                @ rotate_y(self.angle_y) @ rotate_x(self.angle_x)
                @ scale(*self.size))