import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import torch
import random

DATA_DIR = "./assets/images"


def load_resnet():
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11_bn', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg13', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg13_bn', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16_bn', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19_bn', pretrained=True)
    model = models.resnet152(pretrained=True)
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
    return math.sqrt(((img_enc_sel-img_enc_inf)**2).sum(axis=0))


def find_topK_similar(data_loader, encodings, image_index, k=5):
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
    distances_topK_indicies = np.argpartition(distances, k)[:k]
    distances = np.array(distances)
    return distances[distances_topK_indicies], distances_topK_indicies


def find_topK_similar_simple(encodings, image_index, k=5, difficulty=1):
    distances = []
    sel_img_encoding = torch.tensor(encodings[image_index])
    for image_idx in range(encodings.shape[0]):
        curr_img_encoding = torch.tensor(encodings[image_idx])
        distance = L2_NORM(sel_img_encoding, curr_img_encoding)
        if torch.cuda.is_available():
            distance = distance.detach().cpu().numpy()
        distances.append(distance)

    return difficulty_img_selection(k,difficulty,np.array(distances))

def difficulty_img_selection(k, difficulty, distances):
    range_inf_idx = []
    range_sup_idx = []
    x1 = 20.0
    x2 = 22.0
    x3 = 22.75
    x4 = 24.0

    dist_size = distances.shape[0]
    indices = np.argsort(distances)
    distances_ordered = distances[indices]

    '''plt.hist(distances_ordered)
    plt.title('Image Distances from Image {}'.format(chosen_id))
    plt.xlabel('Distance between images')
    plt.ylabel('Amount of images with that distance')
    plt.savefig('distance_distribution/dist_chosen_id_{}.png'.format(chosen_id))'''

    if difficulty == 1: #easy
        lim_inf_inf = 0
        lim_inf_sup = 0
        lim_sup_inf = x2
        lim_sup_sup = x3
    if difficulty == 2: #medium
        lim_inf_inf = x1
        lim_inf_sup = x2
        lim_sup_inf = x3
        lim_sup_sup = x4
    if difficulty == 3: #hard
        lim_inf_inf = 0
        lim_inf_sup = x1
        lim_sup_inf = x4
        lim_sup_sup = distances_ordered[dist_size-1]

    dist_aux = []
    for idx in range(distances.shape[0]):
        dist_idx = distances[idx]
        if dist_idx > 0  and not (dist_idx in dist_aux):
            if lim_inf_inf <= dist_idx < lim_inf_sup:
                dist_aux.append(dist_idx)
                range_inf_idx.append(idx)
            elif lim_sup_inf <= dist_idx < lim_sup_sup:
                range_sup_idx.append(idx)
                dist_aux.append(dist_idx)
    if len(range_inf_idx) < k and len(range_inf_idx) < len(range_sup_idx):
        difficulty_range_idx = np.array(range_sup_idx)
    else:
        difficulty_range_idx = np.array(range_inf_idx)
    #difficulty_range_idx = np.append(np.array(range_inf_idx), np.array(range_sup_idx)).astype(int)  # por algum motivo estava a transformar num vetor de floats
    if len(difficulty_range_idx) == 0:
        _, difficulty_range_idx = difficulty_img_selection(k, difficulty + 1, distances) # nunca vai ser chamado no quando se esta no nivel dificil
    np.random.shuffle(difficulty_range_idx)
    difficulty_range_idx = difficulty_range_idx[:k]

    print("Range Inf: {}".format(range_inf_idx))
    print("Range Sup: {}".format(range_sup_idx))
    print("Range Total: {}".format(difficulty_range_idx))
    print("Presented distances: {}".format(distances[difficulty_range_idx]))

    return distances[difficulty_range_idx], difficulty_range_idx


def obtain_similiar_images(top_k_indicies, chosen_idx):
    dir_files = os.listdir(DATA_DIR)
    sorted_dir_files = sorted(dir_files)
    #sorted_dir_files.remove(".comments")
    np_dir_files = np.array(sorted_dir_files)
    np_sim_dir_files = np_dir_files[top_k_indicies]

    indexes = remove_file_extension(np_sim_dir_files)
    chosen = np_dir_files[chosen_idx][:-4]

    print(indexes)
    return indexes, chosen

def remove_file_extension(test_list):
    return [sub[ : -4] for sub in test_list]

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

def prepare_encodings_resnet():
    feats_dir = "./assets/feats_a"
    encodings = None
    for enc_name in sorted(os.listdir(feats_dir)):
        _encodings = np.load(f'{feats_dir}/{enc_name}', 'r', allow_pickle=True)
        _encodings = _encodings['encodings']
        if encodings is None:
            encodings = _encodings
        else:
            encodings = np.concatenate([encodings, _encodings], axis=0)

    return encodings


def get_nearest_images_idx(chosen_idx, difficulty=1):
    resnet = load_resnet()
    if torch.cuda.is_available():
        resnet.to('cuda')

    # data_loader = loading_data()
    # encodings = create_images_encoding(vgg, data_loader)
    encodings = prepare_encodings_resnet()
    print(f'Chosenid: {chosen_idx}')
    print(f'Encoding shape: {encodings.shape}')
    _, top_k_indicies = find_topK_similar_simple(encodings, chosen_idx, k=5, difficulty=difficulty)
    print(f'Topk_indicies: {top_k_indicies}')
    similar_images, chosen  = obtain_similiar_images(top_k_indicies, chosen_idx)
    return similar_images, chosen
