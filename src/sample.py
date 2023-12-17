import os
import torch
from torchvision import transforms

from model import CatDogClassifier
from utils import DATA_DIR, read_image

if __name__ == "__main__":
    label_map = {0: "cat", 1: "dog"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(DATA_DIR, "models", "cat_dog_classifier-20231217155700-45.pth")
    model = CatDogClassifier().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    img1 = read_image(os.path.join(DATA_DIR, "tests", "c.jpg"))
    img2 = read_image(os.path.join(DATA_DIR, "tests", "d.jpg"))
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img1)
        print(output)
        output = torch.softmax(output, dim=1)
        _, predicted = torch.max(output.data, 1)
        print(label_map[predicted.item()])
        output = model(img2)
        output = torch.softmax(output, dim=1)
        print(output)
        _, predicted = torch.max(output.data, 1)
        print(label_map[predicted.item()])
