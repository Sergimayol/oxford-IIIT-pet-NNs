import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AnimalSegmentationPretained2, CatDogClassifier, AnimalSegmentationPretained
from train import get_animalseg_dataset
from utils import DATA_DIR, read_image, MODELS_DIR


import numpy as np


def visualize_convolutions(model, input_image, layer_indices=None):
    model.eval()
    with torch.no_grad():
        # Configura el modelo para visualizar las convoluciones
        _, conv_output = model(input_image, see_convs=True)

        # Extrae las salidas de las capas de convolución especificadas o de todas las capas
        if layer_indices is None:
            layer_indices = range(conv_output.size(1))  # Todas las capas
        activations = [conv_output[:, idx, :, :].squeeze().cpu().numpy() for idx in layer_indices]

        # Configura el grid para mostrar las activaciones
        num_layers = len(layer_indices) // 4
        cols = 8  # Puedes ajustar el número de columnas en el grid
        rows = int(np.ceil(num_layers / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
        fig.subplots_adjust(hspace=0.5)  # Ajusta el espaciado vertical entre subgráficos

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j 
                if idx < num_layers:
                    axes[i, j].imshow(activations[idx], cmap='viridis')
                    axes[i, j].axis('off')
                    axes[i, j].set_title(f'Channel {layer_indices[idx]}')

        plt.show()

    return model.last_conv_output  # Devuelve la salida de la última capa convolucional

def trimap2f(t, trimap):
    return (t(trimap) * 255.0 - 1) / 2

if __name__ == "__main__":
    r"""
    import cv2
    from torchvision import transforms 

    # Ruta de la imagen del trimap
    trimap_path = r"C:\Users\Sergi\Documents\GitHub\oxford-IIIT-pet-NNs\data\annotations\trimaps\Abyssinian_1.png"

    # Cargar la imagen del trimap
    trimap_image = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
    img = read_image(os.path.join(DATA_DIR, "annotations", "trimaps", "Abyssinian_1.png"), mode="L")

    # Transformación a tensor
    to_tensor = transforms.ToTensor()
    trimap_tensor = to_tensor(img)
    img = trimap2f(to_tensor, img)

    # Mostrar el tensor (opcional)
    print("Tensor del trimap:\n", trimap_tensor)

    trimap_denormalized = trimap_tensor * 255.0
    print("Tensor del trimap:\n", trimap_denormalized, trimap_denormalized.shape, trimap_tensor.shape)
    print(trimap_denormalized.tolist())
    tmp = trimap_denormalized.squeeze().to(torch.int64)
    tmp -= 1

    # Mostrar la imagen desnormalizada (opcional)
    plt.imshow(trimap_denormalized[0].numpy().astype(np.uint8))  # asumiendo que es un tensor de un solo canal
    plt.title("Trimap Desnormalizado")
    plt.axis('off')
    plt.show()
    print(img.shape, img)
    t = transforms.ToPILImage()
    plt.imshow(t(img))
    plt.show()
    exit()
    # https://www.kaggle.com/code/dhruv4930/oxford-iiit-pets-segmentation-using-pytorch#Trimap-Legend
    tr, tt = get_animalseg_dataset()
    # Show a batch of images
    batch = next(iter(tr))
    _, labels = batch
    print(labels.shape, labels[0].squeeze().shape, labels[0].squeeze())
    # Plot the labels which are the trimap images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(labels[0].squeeze(), cmap="gray")
    plt.show()


    label_map = {0: "cat", 1: "dog"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(MODELS_DIR, "cat_dog_classifier_ed8715ca-baa5-48e8-8224-3333807f1622_final.pth")
    model = CatDogClassifier().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    img1 = read_image(os.path.join(DATA_DIR, "tests", "saint_bernard_114.jpg"))
    img2 = read_image(os.path.join(DATA_DIR, "tests", "japanese_chin_135.jpg"))
    img3 = read_image(os.path.join(DATA_DIR, "tests", "a.jpg"))
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)
    img3 = transform(img3).unsqueeze(0).to(device)

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

        output= model(img3)
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
    from ultralytics import YOLO

    model = YOLO("./runs/detect/train5/weights/best.pt")
    model.to("cuda")
    img1 = read_image(os.path.join(DATA_DIR, "tests", "c.jpg"))
    img2 = read_image(os.path.join(DATA_DIR, "tests", "d.jpg"))

    results = model(img2)
    result = results[0]
    import cv2

    image = cv2.imread(os.path.join(DATA_DIR, "tests", "d.jpg"))

    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        print(f"[INFO] Object type: {class_id}, Coordinates: {cords}, Probability: {conf}")
        cv2.rectangle(image, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{class_id} {conf}",
            (cords[0], cords[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    image = cv2.resize(image, (800, 600))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    #model = AnimalSegmentationPretained()
    model = AnimalSegmentationPretained2()
    #model = torch.hub.load("pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=True)
    model.to("cuda")
    path = os.path.join(MODELS_DIR, "animal_segmentation_983842c5-885a-4ca7-9642-20ae324fbb40.pth")
    path = os.path.join(MODELS_DIR, "animal_segmentation_e460a04b-fd87-479f-a0d5-a36392ccc882.pth")
    model.load_state_dict(torch.load(path))
    model.eval()
    img1 = read_image(os.path.join(DATA_DIR, "tests", "a.jpg"))
    img2 = read_image(os.path.join(DATA_DIR, "tests", "japanese_chin_135.jpg"))
    img3 = read_image(os.path.join(DATA_DIR, "tests", "saint_bernard_114.jpg"))
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    img1 = transform(img1).unsqueeze(0).to("cuda")
    img2 = transform(img2).unsqueeze(0).to("cuda")
    img3 = transform(img3).unsqueeze(0).to("cuda")

    with torch.no_grad():
        output = model(img1)[0]
        plt.imshow(output.cpu().numpy().squeeze())
        plt.show()
        output = model(img2)[0]
        plt.imshow(output.cpu().numpy().squeeze())
        plt.show()
        output = model(img3)[0]
        plt.imshow(output.cpu().numpy().squeeze())
        plt.show()
