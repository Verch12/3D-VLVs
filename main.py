import ctypes
import random

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders
from pygame.locals import (DOUBLEBUF, OPENGL, RESIZABLE, QUIT, KEYDOWN, K_ESCAPE,
                           K_r, K_SPACE, K_BACKSPACE, VIDEORESIZE, MOUSEMOTION,
                           MOUSEWHEEL)
from figures import *
from objects import *


VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
uniform mat4 mvp;
uniform vec4 ocr;
out vec4 v_color;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    v_color = color * ocr;
}
"""

FRAGMENT_SHADER = """
#version 330 core
uniform bool has_new_color;
uniform vec4 new_color;

in vec4 v_color;
out vec4 frag_color;
void main() {
    if (has_new_color)
        frag_color = new_color;
    else
        frag_color = v_color;
    }
"""



# ---------------------------------------------------------------------------
# Основная программа
# ---------------------------------------------------------------------------

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
    ocr_location = glGetUniformLocation(program, "ocr")

    has_new_c_location = glGetUniformLocation(program, "has_new_color")
    new_c_location = glGetUniformLocation(program, "new_color")

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

    objects = [
        SceneObject(axis),
        SceneObject(cube, position=(0, 0, 0),size=(0.5, 0.5, 0.5), new_color=(0.08, 0.56, 0.61, 1)),
    ]


    net_batch = BatchedLines(line, 20)
    #objects[1].angle_y = np.radians(90)
    #objects[1].size = (2, 1, 1)
    #objects[1].position = (0, 2, 1)

    camera = OrbitCamera()
    clock = pygame.time.Clock()
    running = True

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
        proj_view = proj @ camera.view_matrix()   # общая часть считается один раз за кадр

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)
        for obj in objects:
            mvp = proj_view @ obj.model_matrix()
            if obj.new_color:
                new_c = np.array(obj.new_color, dtype=np.float32)
                glUniform1i(has_new_c_location, GL_TRUE)
                glUniform4fv(new_c_location, 1, new_c)
            else:
                glUniform1i(has_new_c_location, GL_FALSE)
            glUniformMatrix4fv(mvp_location, 1, GL_TRUE, mvp)
            #glUniform4fv(ocr_location, 1, ocr)
            obj.mesh.draw()

        glUniform1i(has_new_c_location, GL_FALSE)
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
