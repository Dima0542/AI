import torch
from customtkinter import CTk, CTkButton
from PIL import Image, ImageDraw
from pathlib import Path
from clases import Model

model = Model()
model.load_state_dict(torch.load("vex.pth", weights_only=True))
model.eval()

def generate_rect():
    global model
    y = (model(torch.tensor([1, 0], dtype=torch.float32)).reshape(10, 10).detach().cpu() * 255).tolist()
    img = Image.new("L", (100, 100), color=0)
    draw = ImageDraw.Draw(img)

    for index_y, num_y in enumerate(y):
        for index_x, num_x in enumerate(num_y):
            x1 = index_x * 10
            y1 = index_y * 10
            x2 = x1 + 10
            y2 = y1 + 10

            draw.rectangle((x1, y1, x2, y2), fill=int(num_x))

    desktop_path = Path.home() / "Desktop"
    file_path = desktop_path / "Rect.png"
    img.save(file_path)

def generate_circle():
    global model
    y = (model(torch.tensor([0, 1], dtype=torch.float32)).reshape(10, 10).detach().cpu() * 255).tolist()
    img = Image.new("L", (100, 100), color=0)
    draw = ImageDraw.Draw(img)

    for index_y, num_y in enumerate(y):
        for index_x, num_x in enumerate(num_y):
            x1 = index_x * 10
            y1 = index_y * 10
            x2 = x1 + 10
            y2 = y1 + 10

            draw.rectangle((x1, y1, x2, y2), fill=int(num_x))

    desktop_path = Path.home() / "Desktop"
    file_path = desktop_path / "Circle.png"
    img.save(file_path)

window = CTk()
window.geometry("700x400")
window.title("Program")
window.configure(bg="white")
window.resizable(False, False)

button_generate_rect = CTkButton(window, width=228, height=78, text="Generate Rect", font=("Courier", 28, "bold"), command=generate_rect)
button_generate_rect.place(x=120, y=150)

button_generate_circle = CTkButton(window, width=228, height=78, text="Generate circle", font=("Courier", 27, "bold"), command=generate_circle)
button_generate_circle.place(x=398, y=150)

window.mainloop()