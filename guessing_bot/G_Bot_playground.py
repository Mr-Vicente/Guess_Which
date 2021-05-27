from VQA_API import VQA
from G_Bot import G_Bot
import bot_utils as ut
import torch as T

dataDir		=   '../../VQA_dataset'
versionType =   'v2_' # this should be '' when using VQA v2.0 dataset
taskType    =   'OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    =   'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType =   'train2014'
annFile     =   f'{dataDir}/Annotations/{versionType}{dataType}_{dataSubType}_annotations.json'
quesFile    =   f'{dataDir}/Questions/{versionType}{taskType}_{dataType}_{dataSubType}_questions.json'
imgDir 		=   f'{dataDir}/Images/{dataType}/{dataSubType}/'

params = {
    'vocabSize': 20573,
    'embedSize': 300,
    'rnnHiddenSize': 512,
    'dialogInputSize': 512,
    'numLayers': 2,
    'imgFeatureSize': 2048,
    'numRounds': 1,
    'numEpochs': 10
}

def fetch_dataset(n_elementes):
    vqa = VQA(annFile, quesFile)
    valid_image_idx = ut.find_valid_images_idx(imgDir)[:n_elementes].tolist()
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

def load_dataset(token_to_idx):
    # 1. create Dataset and DataLoader object
    print("Creating Dataset and DataLoader\n")
    Xs, Ys = fetch_dataset(10)
    train_dataset = ut.VQDataset(Xs, Ys, token_to_idx)

    batch_s = 2
    train_dataloader = T.utils.data.DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
    return train_dataloader

def main():
    guessing_bot = G_Bot(params)
    train_dataloader = load_dataset(guessing_bot.token_to_ix)
    guessing_bot.train(train_dataloader, params)

if __name__ == "__main__":
    main()