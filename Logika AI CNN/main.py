import pygame
import torch
from clases import Tile, Button
from clases import Vex

vex = Vex(10)
vex.load_state_dict(torch.load("weights\\vex_weight.pth", weights_only=True))
vex.eval()

pygame.init()

wight, height = 800, 600

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (79, 79, 79)
GREEN = (4, 148, 7)

tile_size, start_x, start_y = 15, 49, 99
brush_radius, brush_power = 1, 150
eraser_radius, eraser_power = 2, -255

click_sound = pygame.mixer.Sound("sound\\click.mp3")

get_click = False

window = pygame.display.set_mode((wight, height))
pygame.display.set_caption("Convolutional Neural Network MNIST")

clock = pygame.time.Clock()

def zone_draw(tile_size, start_x, start_y, border_width):
    tiles = []

    zone_size = tile_size * 28

    border = pygame.Rect(start_x - border_width, start_y - border_width, zone_size + border_width * 2, zone_size + border_width * 2)

    for row in range(28):
        row_tiles = []

        for col in range(28):
            x = start_x + col * tile_size
            y = start_y + row * tile_size

            tile = Tile(x, y, tile_size, tile_size, click_sound, color=0)

            row_tiles.append(tile)

        tiles.append(row_tiles)

    return tiles, border

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

font = pygame.font.SysFont("Arial", 20)

clear_Button = Button(start_x-10, 540, 440, 50, "Очистити", font, GRAY, 20)

tiles, border = zone_draw(tile_size, start_x, start_y, 10)

classes_numbers = []
classes_numbers_num = []
classes_numbers_text = []
classes_font = pygame.font.SysFont("Arial", 20)

for index in range(10):
    classes_numbers_num.append(index)
    classes_numbers.append(pygame.rect.Rect(550, 140, 0, 25))
    text = classes_font.render(str(index), True, (0, 0, 0))
    classes_numbers_text.append(text)

for index in range(10):
    classes_numbers[index].y += 35 * index

count = 0
running = True
while running:
    count += 1
    window.fill(WHITE)
    pygame.draw.rect(window, (36, 36, 36), border)

    mouse_pos = pygame.mouse.get_pos()
    mouse_x = mouse_pos[0]
    mouse_y = mouse_pos[1]

    if pygame.mouse.get_pressed()[0]:
        col = (mouse_x - start_x) // tile_size
        row = (mouse_y - start_y) // tile_size

        if 0 <= col < 28 and 0 <= row < 28:
            change = brush(tiles, row, col, radius=brush_radius, power=brush_power)

            if count % 10 == 0 and change > 0:
                volume = change / 255
                click_sound.set_volume(volume)
                click_sound.play()

    if pygame.mouse.get_pressed()[2]:
        col = (mouse_x - start_x) // tile_size
        row = (mouse_y - start_y) // tile_size

        if 0 <= col < 28 and 0 <= row < 28:
            change = brush(tiles, row, col, radius=eraser_radius, power=eraser_power)

            if count % 10 == 0 and change > 0:
                volume = change / 255
                click_sound.set_volume(volume)
                click_sound.play()

    for tile_row in tiles:
        for tile_col in tile_row:
            tile_col.draw(window)

    clear_Button.colidemause(mouse_pos)
    clear_Button.draw(window)

    x = []

    for row in tiles:
        row_list_number = []
        for col in row:
            row_list_number.append(col.color)
        x.append(row_list_number)

    x = torch.tensor(x, dtype=torch.float32)
    x = x / 255.0
    x = x.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        y = torch.softmax(vex(x), dim=1)[0]

    for index in range(10):
        classes_numbers[index].width= int(y[index] * 100)
        pygame.draw.rect(window, GREEN, classes_numbers[index])
        window.blit(classes_numbers_text[index], (520, classes_numbers[index].y))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                get_click = clear_Button.colidemause(mouse_pos)


    if get_click:
        for tile_row in tiles:
            for tile_col in tile_row:
                tile_col.color = 0


    pygame.display.update()
    clock.tick(100)
pygame.quit()
