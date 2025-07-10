import pygame

NORMAL = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
PURPLE = (160, 32, 240)



class Screen:
    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((800, 800))
        self._set_color(NORMAL)


    def update_state(self, state):
        if state == "idle":
            self._set_color(NORMAL)
        elif state == "recording":
            self._set_color(GREEN)
        elif state == "stop":
            self._set_color(PURPLE)
        elif state == "delete":
            self._set_color(RED)
        else:
            raise Exception("Invalid state sent to Demo Collection Screen")
    def _set_color(self, color):
        self._screen.fill(color)
        pygame.display.flip()