import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

DATA_DIR = "./assets/images"


def load_vgg():
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11_bn', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg13', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg13_bn', pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16_bn', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19_bn', pretrained=True)
    model.eval()
    return model


def loading_data():
    img_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=img_transform)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=8, pin_memory=True)
    return data_loader


def create_images_encoding(vgg, data_loader):
    encodings = []
    images_it = iter(data_loader)
    with torch.no_grad():
        n_batches = len(images_it)
        for image_batch in range(n_batches):
            images, labels = next(images_it)
            if torch.cuda.is_available():
                images = images.cuda()
            output = vgg(images)
            for encoding in output:
                encodings.append(encoding)
    encodings = np.array(encodings)
    return encodings


def L2_NORM(img_enc_sel, img_enc_inf):
    return torch.norm(((img_enc_sel * img_enc_inf)), 2, -1)


def findTopKSimilar(data_loader, encodings, image_index, k=5):
    distances = []
    index = 0
    images_it = iter(data_loader)
    n_batches = len(images_it)
    for image_batch in range(n_batches):
        images, _ = next(images_it)
        batch_size = len(images)
        for image_i in range(batch_size):
            encodings_index = index + image_i
            sel_img_encoding = encodings[image_index]
            curr_img_encoding = encodings[encodings_index]
            distance = L2_NORM(sel_img_encoding, curr_img_encoding)
            if torch.cuda.is_available():
                distance = distance.detach().cpu().numpy()
            distances.append(distance)
            # print("Torch NORM L2 Distance is : ", distance)
        index += batch_size
    distances_topK_indicies = np.argpartition(distances, -k)[-k:]
    distances = np.array(distances)
    return distances[distances_topK_indicies], distances_topK_indicies


def findTopKSimilar_simple(encodings, image_index, k=5):
    distances = []
    for image_idx in range(encodings.shape[0]):
        sel_img_encoding = torch.tensor(encodings[image_index])
        curr_img_encoding = torch.tensor(encodings[image_idx])
        distance = L2_NORM(sel_img_encoding, curr_img_encoding)
        if torch.cuda.is_available():
            distance = distance.detach().cpu().numpy()
        distances.append(distance)
    distances_topK_indicies = np.argpartition(distances, -k)[-k:]
    distances = np.array(distances)
    return distances[distances_topK_indicies], distances_topK_indicies


def obtain_similiar_images(top_k_indicies):
    dir_files = os.listdir(DATA_DIR)
    print(len(dir_files))
    sorted_dir_files = sorted(dir_files)
    similar_images = np.array(sorted_dir_files)
    similar_images = similar_images[top_k_indicies]
    return similar_images


def show_k_images(similar_images):
    plt.figure()
    for filename in similar_images:
        plt.imshow(Image.open(os.path.join(DATA_DIR, filename)))


def prepare_encodings():
    feats_dir = "./assets/feats"
    encodings = None
    for enc_name in sorted(os.listdir(feats_dir)):
        _encodings = np.load(f'{feats_dir}/{enc_name}', 'r')
        _encodings = _encodings['x'].T
        _encodings = np.mean(_encodings, axis=0)
        # _encodings = _encodings[0,:]
        _encodings = np.reshape(_encodings, newshape=(1, 2048))
        if encodings is None:
            encodings = _encodings
        else:
            encodings = np.concatenate([encodings, _encodings], axis=0)

    return encodings

def prepare_encodings_good():
    feats_dir = "./assets/feats_a"
    encodings = None
    for enc_name in sorted(os.listdir(feats_dir)):
        _encodings = np.load(f'{feats_dir}/{enc_name}', 'r', allow_pickle=True)
        _encodings = _encodings['encodings'].detach().cpu().numpy()
        if encodings is None:
            encodings = _encodings
        else:
            encodings = np.concatenate([encodings, _encodings], axis=0)

    return encodings


def get_nearest_images_idx(chosen_idx):
    vgg = load_vgg()
    if torch.cuda.is_available():
        vgg.to('cuda')

    # data_loader = loading_data()
    # encodings = create_images_encoding(vgg, data_loader)
    encodings = prepare_encodings_good()
    print("chosenid", chosen_idx)
    print("encoding", encodings.shape)
    _, top_k_indicies = findTopKSimilar_simple(encodings, chosen_idx, k=5)
    print("topk", top_k_indicies)
    similar_images = obtain_similiar_images(top_k_indicies)
    return similar_images
