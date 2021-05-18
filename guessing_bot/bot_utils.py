import numpy as np

def pool_images(n_images, pool_size):
    """ ex: 1000 images with pool size of 4
        return: [238, 123, 12, 39]
    """
    pool = np.random.randint(n_images, size=pool_size)
    return pool

def stack_images_array(images, pool_size):
    n_images = len(images)
    pool_image_indeces = pool_images(n_images, pool_size)
    return images[pool_image_indeces]

def imgID_2_filename(dataSubType, imgId):
    imgFilename = f'COCO_{dataSubType}_{str(imgId).zfill(12)}.jpg'
    return imgFilename

def stack_images(dataSubType, n_images, pool_size):
    pool_image_indeces = pool_images(n_images, pool_size)
    selected_image_idx = pool_image_indeces[0]
    images_filenames = np.array([imgID_2_filename(dataSubType, imgId) for imgId in pool_image_indeces])
    return selected_image_idx, images_filenames

def random_pool_input_generator(dataSubType, n_images, vqa, pool_size):
    selected_image_idx, images_filenames = stack_images(dataSubType, n_images, pool_size)
    annIds = vqa.getQuesIds(imgIds=[selected_image_idx])
    anns = vqa.loadQA(annIds)
    input_data = {
        'pool_idx': images_filenames,
        'questions': annIds,
        'answers': anns
    }
    return input_data

def multiple_input(dataSubType, n_images, vqa, pool_size):
    inputs = []
    for i in range(n_images):
        single_input = random_pool_input_generator(dataSubType, n_images, vqa, pool_size)
        inputs.append(single_input)
    inputs = np.array(inputs)
    return inputs

