import ctypes
import random

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders
from pygame.locals import (DOUBLEBUF, OPENGL, RESIZABLE, QUIT, KEYDOWN, K_ESCAPE,
                           K_r, K_l, K_g, VIDEORESIZE, MOUSEMOTION,
                           MOUSEWHEEL)
from figures import *
from objects import *
from BFS import *
from labyrinth import start_pos, end_pos, obstacle_cells

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
uniform mat4 mvp;
uniform mat4 model;     // НОВОЕ: нужна, чтобы получить позицию вершины в мире
uniform vec4 ocr;
out vec4 v_color;
out vec3 v_world_pos;  // НОВОЕ
void main() {
    v_world_pos = (model * vec4(position, 1.0)).xyz;
    gl_Position = mvp * vec4(position, 1.0);
    v_color = color * ocr;
}
"""

FRAGMENT_SHADER = """
#version 330 core
uniform bool has_new_color;
uniform vec4 new_color;

// НОВОЕ: параметры освещения
uniform bool use_lighting;   // false для линий, осей и лампочки
uniform float light_intensity;
uniform vec3 light_pos;      // неподвижная точка света (в мировых координатах)
uniform vec3 light_color;
uniform vec3 view_pos;       // позиция камеры (для бликов)

in vec4 v_color;
in vec3 v_world_pos;
out vec4 frag_color;

void main() {
    vec4 base = has_new_color ? new_color : v_color;

    if (!use_lighting) {
        frag_color = base;
        return;
    }

    // Нормаль грани считаем прямо в шейдере: производные мировой позиции
    // по экрану дают два вектора, лежащие в плоскости грани.
    vec3 n = normalize(cross(dFdx(v_world_pos), dFdy(v_world_pos)));

    vec3 to_light = light_pos - v_world_pos;
    float dist = length(to_light);
    vec3 l = to_light / dist;                    // направление на свет
    vec3 v = normalize(view_pos - v_world_pos);  // направление на камеру
    vec3 h = normalize(l + v);

    float ambient = 0.2;                                                     // фоновый свет
    float diff = max(dot(n, l), 0.0);                                        // рассеянный свет
    float spec = (diff > 0.0) ? pow(max(dot(n, h), 0.0), 32.0) * 0.4 : 0.0;  // блик
    float att = 1.0 / (1.0 + 0.045 * dist + 0.0075 * dist * dist);           // затухание

    vec3 rgb = (ambient + diff * att) * base.rgb * light_color*light_intensity + spec * att * light_color*light_intensity;
    frag_color = vec4(rgb, base.a);
}
"""

def main():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

    width, height = 900, 650
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)

    program = shaders.compileProgram(
        shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
    )
    mvp_location = glGetUniformLocation(program, "mvp")
    model_location = glGetUniformLocation(program, "model")
    ocr_location = glGetUniformLocation(program, "ocr")

    has_new_c_location = glGetUniformLocation(program, "has_new_color")
    new_c_location = glGetUniformLocation(program, "new_color")

    use_light_location = glGetUniformLocation(program, "use_lighting")
    light_intensity_location = glGetUniformLocation(program, "light_intensity")
    light_pos_location = glGetUniformLocation(program, "light_pos")
    light_color_location = glGetUniformLocation(program, "light_color")
    view_pos_location = glGetUniformLocation(program, "view_pos")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.08, 0.08, 0.12, 1.0)

    cube = make_cube()
    pyramid = make_pyramid()
    floor = make_floor()
    line = make_line()
    axis = make_axis()
    meshes = [cube, pyramid, floor, line, axis]

    GRID_SIZE = 20
    LIGHT_INTENSITY = 2
    LIGHT_POS = (10.0, 25.0, 10.0)
    LIGHT_COLOR = (1.0, 1.0, 1.0)
    LIGHT_VISIBLE = False
    GRID_VISIBLE = False

    axis_obj = SceneObject(axis)
    axis_obj.lit = False

    light_obj = SceneObject(cube, position=LIGHT_POS, new_color=(*LIGHT_COLOR, 1), size=(0.5, 0.5, 0.5))
    light_obj.lit = False
    light_obj.name = "light"

    objects = [
        axis_obj,
        light_obj,
        #SceneObject(cube, position=(0, 0, 0),size=(0.5, 0.5, 0.5), new_color=(0.08, 0.86, 0.61, 0.2)),
    ]


    net_batch = BatchedLines(line, GRID_SIZE+1)
    #objects[1].angle_y = np.radians(90)
    #objects[1].size = (2, 1, 1)
    #objects[1].position = (0, 2, 1)

    camera = OrbitCamera()
    clock = pygame.time.Clock()
    running = True

    """start_pos = (0, 0, 0)
    end_pos = (5, 0, 5)
    obstacle_cells = [
        (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1),
        (2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4),
        (3, 2, 2), (3, 3, 2), (1, 2, 2), (4, 2, 2),
    ]"""
    def add_cube(pos, color):
        objects.append(SceneObject(cube, position=pos,size=(0.5, 0.5, 0.5), new_color=color))

    objects.append(SceneObject(cube, position=start_pos,size=(0.5, 0.5, 0.5), new_color=(0, 0.7, 0, 1)))
    objects.append(SceneObject(cube, position=end_pos,size=(0.5, 0.5, 0.5), new_color=(0.7, 0, 0, 1)))
    for i in obstacle_cells: add_cube(i, (0.5, 0.5, 0.5, 1))
    for i in bfs(start_pos, end_pos, obstacle_cells, GRID_SIZE): add_cube(i, (0.08, 0.86, 0.61, 0.2))

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_r:
                    camera.reset()
                elif event.key == K_l:
                    LIGHT_VISIBLE = not LIGHT_VISIBLE
                elif event.key == K_g:
                    GRID_VISIBLE = not GRID_VISIBLE
            elif event.type == VIDEORESIZE:
                width, height = max(event.w, 1), max(event.h, 1)
                glViewport(0, 0, width, height)
            elif event.type == MOUSEMOTION and event.buttons[0]:
                camera.rotate(*event.rel)
            elif event.type == MOUSEMOTION and event.buttons[2]:
                camera.translate(*event.rel)
            elif event.type == MOUSEWHEEL:
                camera.zoom(event.y)

        proj = perspective(45.0, width / height, 0.1, 100.0)
        view = camera.view_matrix()
        proj_view = proj @ view
        eye = np.linalg.inv(view)[:3, 3]

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)

        # Параметры света одинаковы для всего кадра
        glUniform1f(light_intensity_location, LIGHT_INTENSITY)
        glUniform3f(light_pos_location, *LIGHT_POS)
        glUniform3f(light_color_location, *LIGHT_COLOR)
        glUniform3f(view_pos_location, *eye)
        glUniform4f(ocr_location, 1.0, 1.0, 1.0, 1.0)

        for obj in objects:
            if getattr(obj, "name", None) == "light" and not LIGHT_VISIBLE:
                continue

            model = obj.model_matrix()
            mvp = proj_view @ model
            if obj.new_color:
                new_c = np.array(obj.new_color, dtype=np.float32)
                glUniform1i(has_new_c_location, GL_TRUE)
                glUniform4fv(new_c_location, 1, new_c)
            else:
                glUniform1i(has_new_c_location, GL_FALSE)
            glUniform1i(use_light_location, getattr(obj, "lit", True))
            glUniformMatrix4fv(model_location, 1, GL_TRUE, model)
            glUniformMatrix4fv(mvp_location, 1, GL_TRUE, mvp)
            obj.mesh.draw()

        if GRID_VISIBLE:
            glUniform1i(use_light_location, GL_FALSE)
            glUniform1i(has_new_c_location, GL_FALSE)
            glUniformMatrix4fv(model_location, 1, GL_TRUE, np.identity(4, dtype=np.float32))
            glUniformMatrix4fv(mvp_location, 1, GL_TRUE, proj_view)
            glUniform4fv(ocr_location, 1, np.array([0.5, 0.5, 0.5, 0.2], dtype=np.float32))
            net_batch.draw()

        pygame.display.flip()

        fps = clock.get_fps()
        pygame.display.set_caption(f"FPS: {fps:.1f}")

    for resource in [*meshes, net_batch]:
        resource.delete()
    glDeleteProgram(program)
    pygame.quit()


if __name__ == "__main__":
    main()