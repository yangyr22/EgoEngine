import pygame

NORMAL = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
PURPLE = (160, 32, 240)

KEY_START = pygame.K_s
KEY_CONTINUE = pygame.K_c
KEY_SABOTAGE = pygame.K_x
KEY_QUIT_RECORDING = pygame.K_q


class KBReset:
    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((800, 800))
        self._set_color(NORMAL)
        self._save = False
        self._sabotage = False

    def update(self) -> str:
        pressed_last = self._get_pressed()
        if KEY_QUIT_RECORDING in pressed_last:
            self._set_color(RED)
            self._save = False
            return "normal"

        if self._save:
            if not self._sabotage:
                if KEY_SABOTAGE in pressed_last:
                    self._sabotage = True
                    self._set_color(PURPLE)
                    return "sabotage"
                else:
                    return "save"
            else:
                if KEY_SABOTAGE in pressed_last:
                    self._sabotage = False
                    self._set_color(GREEN)
                    return "save"
                else:
                    return "sabotage"

        if KEY_START in pressed_last:
            self._set_color(GREEN)
            self._save = True
            return "start"

        self._set_color(NORMAL)
        return "normal"

    def _get_pressed(self):
        pressed = []
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pressed.append(event.key)
        return pressed

    def _set_color(self, color):
        self._screen.fill(color)
        pygame.display.flip()


def main():
    kb = KBReset()
    while True:
        state = kb.update()
        if state == "start":
            print("start")


if __name__ == "__main__":
    main()
