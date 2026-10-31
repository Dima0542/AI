import pygame
import torch
from clases import Tile, Button, Vex

pygame.init()

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
GRAY = (79, 79, 79)
GREEN = (4, 148, 7)

TILE_SIZE = 15
START_X, START_Y = 49, 99

BRUSH_RADIUS, BRUSH_POWER = 1, 150
ERASER_RADIUS, ERASER_POWER = 2, -255

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CNN MNIST")
clock = pygame.time.Clock()

click_sound = pygame.mixer.Sound("sound\\click.mp3")
clear_sound = pygame.mixer.Sound("sound\\clear_botton_sound.wav")

vex = Vex(10)
vex.load_state_dict(torch.load("weights\\vex_weight.pth", weights_only=True))
vex.eval()


def create_tiles():
    tiles = []

    for row in range(28):
        tile_row = []

        for col in range(28):
            x = START_X + col * TILE_SIZE
            y = START_Y + row * TILE_SIZE

            tile = Tile(x, y, TILE_SIZE, TILE_SIZE, click_sound, color=0)
            tile_row.append(tile)

        tiles.append(tile_row)

    return tiles

def brush(tiles, row, col, radius, power):
    max_change = 0

    for r in range(row - radius, row + radius + 1):
        for c in range(col - radius, col + radius + 1):
            if 0 <= r < 28 and 0 <= c < 28:
                distance = ((r - row) ** 2 + (c - col) ** 2) ** 0.5

                if distance <= radius:
                    strength = 1 - distance / (radius + 1)
                    change = tiles[r][c].update(power * strength)
                    max_change = max(max_change, change)

    return max_change

def predict(tiles):
    x = [[tile.color for tile in row] for row in tiles]

    x = torch.tensor(x, dtype=torch.float32)
    x = (x / 255.0).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        return torch.softmax(vex(x), dim=1)[0]


tiles = create_tiles()

border = pygame.Rect(START_X - 10, START_Y - 10, TILE_SIZE * 28 + 20, TILE_SIZE * 28 + 20)

font = pygame.font.SysFont("Arial", 20)
clear_button = Button(START_X - 10, 540, 440, 50, "Очистити", font, GRAY, 20)

class_bars = [pygame.Rect(550, 140 + i * 35, 0, 25) for i in range(10)]
class_texts = [font.render(str(i), True, (0, 0, 0)) for i in range(10)]

count = 0
running = True

while running:
    count += 1
    window.fill(WHITE)
    pygame.draw.rect(window, (36, 36, 36), border)

    mouse_pos = pygame.mouse.get_pos()
    mouse_x, mouse_y = mouse_pos

    col = (mouse_x - START_X) // TILE_SIZE
    row = (mouse_y - START_Y) // TILE_SIZE

    if 0 <= col < 28 and 0 <= row < 28:
        buttons = pygame.mouse.get_pressed()

        if buttons[0]:
            change = brush(tiles, row, col, BRUSH_RADIUS, BRUSH_POWER)

            if count % 10 == 0 and change > 0:
                click_sound.set_volume(change / 255)
                click_sound.play()

        if buttons[2]:
            change = brush(tiles, row, col, ERASER_RADIUS, ERASER_POWER)

            if count % 10 == 0 and change > 0:
                click_sound.set_volume(change / 255)
                click_sound.play()

    for tile_row in tiles:
        for tile in tile_row:
            tile.draw(window)

    clear_button.colidemause(mouse_pos)
    clear_button.draw(window)

    probabilities = predict(tiles)

    for i in range(10):
        class_bars[i].width = int(probabilities[i] * 100)

        pygame.draw.rect(window, GREEN, class_bars[i])
        window.blit(class_texts[i], (520, class_bars[i].y))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and clear_button.click(event.pos):
                changed = any(tile.color > 0 for row in tiles for tile in row)

                if changed:
                    clear_sound.play()

                for tile_row in tiles:
                    for tile in tile_row:
                        tile.color = 0

    pygame.display.update()
    clock.tick(100)

pygame.quit()
