from VQA_API import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os

from Guess_Bot import Guesser_Bot

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

def main():
    bot = load_model()
    data = load_data()
    bot = train_bot(bot, data)

if __name__ == "__main__":
    main()
