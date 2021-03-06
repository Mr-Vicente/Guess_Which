from VQA_API import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os

from Guess_Bot import Guesser_Bot
import bot_utils as ut

dataDir		=   '../../VQA_dataset'
versionType =   'v2_' # this should be '' when using VQA v2.0 dataset
taskType    =   'OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    =   'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType =   'train2014'
annFile     =   f'{dataDir}/Annotations/{versionType}{dataType}_{dataSubType}_annotations.json'
quesFile    =   f'{dataDir}/Questions/{versionType}{taskType}_{dataType}_{dataSubType}_questions.json'
imgDir 		=   f'{dataDir}/Images/{dataType}/{dataSubType}/'

def load_model():
    return Guesser_Bot()

def train_bot(bot, data):
    bot.train(data)
    return bot

def load_data():
    vqa = VQA(annFile, quesFile)
    annIds = vqa.getQuesIds(quesTypes='how many')
    anns = vqa.loadQA(annIds)
    randomAnn = random.choice(anns)
    vqa.showQA([randomAnn])
    imgId = randomAnn['image_id']
    imgFilename = f'COCO_{dataSubType}_{str(imgId).zfill(12)}.jpg'
    if os.path.isfile(imgDir + imgFilename):
        I = io.imread(imgDir + imgFilename)
        plt.imshow(I)
        plt.axis('off')
        plt.show()

def load_data_real():
    vqa = VQA(annFile, quesFile)
    a = ut.random_pool_input_generator(imgDir, dataSubType, vqa, 6)
    print(a)

def fetch_dataset():
    vqa = VQA(annFile, quesFile)
    valid_image_idx = ut.find_valid_images_idx(imgDir)[:10].tolist()
    inputs = []
    for image_idx in valid_image_idx:
        model_input = {
            'image_idx': image_idx,
            'QAs': []
        }
        annIds = vqa.getQuesIds(imgIds=[image_idx])
        for q_i in range(len(annIds)):
            ann = vqa.loadQA(annIds[q_i])[0]
            q,a = vqa.get_qa(ann)
            model_input['QAs'].append((q,a))
        inputs.append(model_input)
    return inputs, valid_image_idx

def main():
    #bot = load_model()
    #data = load_data_real()
    #bot = train_bot(bot, data)
    test()

if __name__ == "__main__":
    main()
