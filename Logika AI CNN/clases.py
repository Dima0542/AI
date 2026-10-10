import pygame
import torch.nn as nn

class Vex(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.GELU(),

            nn.Conv2d(8, 12, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.GELU(),

            nn.Conv2d(12, 16, 3, 1),
            nn.GELU(),
        )

        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.linear_layer(y_conv)
        return y

class Tile:
    def __init__(self, x, y, w ,h, sound, color=0):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color
        self.sound = sound

    def draw(self, window):
        brightness = 255 - self.color
        pygame.draw.rect(window, (brightness, brightness, brightness), self.rect)

    def update(self, update_color):
        old_color = self.color

        self.color += update_color
        self.color = max(0, min(255, self.color))

        return abs(self.color - old_color)

class Button:
    def __init__(self, x, y, w, h, text, font, color, colide_color=10):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color
        self.start_color = color
        self.colide_color = colide_color
        self.text = text
        self.font = font

    def draw(self, window):
        pygame.draw.rect(window, self.color, self.rect)

        text_surface = self.font.render(self.text, True, (255, 255, 255))

        text_rect = text_surface.get_rect(center=self.rect.center)

        window.blit(text_surface, text_rect)

    def colidemause(self, pos_mouse):
        if self.rect.collidepoint(pos_mouse):
            color_list = list(self.start_color)

            for index in range(len(color_list)):
                color_list[index] -= self.colide_color

                if color_list[index] < 0:
                    color_list[index] = 0

            self.color = tuple(color_list)

            return True

        else:
            self.color = self.start_color
            return False

    def click(self, pos_mouse):
        if self.rect.collidepoint(pos_mouse): return True
        return False
