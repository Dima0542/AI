import torch
from PIL import Image, ImageDraw
from clases import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model().to(device)
model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
model.eval()

x_rect = torch.tensor([1, 0], dtype=torch.float32).to(device)
x_circle = torch.tensor([0, 1], dtype=torch.float32).to(device)

with torch.no_grad():
    y_rect = (model(x_rect).reshape(10, 10).detach().cpu() * 255).tolist()
    y_circle = (model(x_circle).reshape(10, 10).detach().cpu() * 255).tolist()

img_rect = Image.new("L", (100, 100), color=0)
img_circle = Image.new("L", (100, 100), color=0)

draw_rect = ImageDraw.Draw(img_rect)
draw_circle = ImageDraw.Draw(img_circle)

for index_y, num_y in enumerate(y_rect):
    for index_x, num_x in enumerate(num_y):
        x1 = index_x * 10
        y1 = index_y * 10
        x2 = x1 + 10
        y2 = y1 + 10

        draw_rect.rectangle((x1, y1, x2, y2), fill=int(num_x))

for index_y, num_y in enumerate(y_circle):
    for index_x, num_x in enumerate(num_y):
        x1 = index_x * 10
        y1 = index_y * 10
        x2 = x1 + 10
        y2 = y1 + 10

        draw_circle.rectangle((x1, y1, x2, y2), fill=int(num_x))

img_rect.save("img_rect.png")
img_circle.save("img_circle.png")