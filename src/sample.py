import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from model import CatDogClassifier
from utils import DATA_DIR, read_image

if __name__ == "__main__":
    label_map = {0: "cat", 1: "dog"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(DATA_DIR, "models", "cdc-e020950a-ad2a-4859-94a2-08a39fbe4892-100.pth")
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
        output = torch.softmax(output, dim=1)
        print(output)
        _, predicted = torch.max(output.data, 1)
        print(label_map[predicted.item()])

        output = model(img2)
        output = torch.softmax(output, dim=1)
        print(output)
        _, predicted = torch.max(output.data, 1)
        print(label_map[predicted.item()])

    import torchvision

    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=True).to(device)
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
        _, predicted = torch.max(output["out"].data, 1)
        print(predicted.shape)
        plt.imshow(predicted.cpu().numpy().squeeze())
        plt.show()

        output = model(img2)
        print(output)
        _, predicted = torch.max(output["out"].data, 1)
        print(predicted.shape)
        plt.imshow(predicted.cpu().numpy().squeeze())
        plt.show()
