from VQA_API import VQA
from G_Bot import G_Bot
import bot_utils as ut
import torch as T
import pickle
import numpy as np

dataDir		=   '../static/VQA_dataset'
versionType =   'v2_' # this should be '' when using VQA v2.0 dataset
taskType    =   'OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    =   'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType =   'val2014'
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
    'numRounds': 3,
    'numEpochs': 1
}

with open('../pickles/image_encodings.p', 'rb') as f:
    image_encoding = pickle.load(f)

def fetch_dataset(n_elementes):
    vqa = VQA(annFile, quesFile)
    valid_image_idx = list(image_encoding.keys())
    #valid_image_idx = ut.find_valid_images_idx(imgDir)[:n_elementes].tolist()
    print(valid_image_idx)
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

def load_dataset(token_to_idx, n_images=10):
    # 1. create Dataset and DataLoader object
    print("Creating Dataset and DataLoader\n")
    Xs, Ys = fetch_dataset(n_images)
    train_dataset = ut.VQDataset(Xs, Ys, token_to_idx)

    batch_s = 1
    train_dataloader = T.utils.data.DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
    return train_dataloader, Ys

def main():
    guessing_bot = G_Bot(params)
    train_dataloader, idxs = load_dataset(guessing_bot.token_to_ix)
    guessing_bot.reset()
    guessing_bot.train(train_dataloader, params)
    T.save(guessing_bot.state_dict(), '')
    """q_idx, a_idx = ut.convert_statements_to_idx("how many people?", "2", guessing_bot.token_to_ix)
    q, a = T.tensor(np.array([q_idx])), T.tensor(np.array([a_idx]))
    ql, al = T.tensor(np.array([[len(q_idx)]])), T.tensor(np.array([[len(a_idx)]]))
    #print("main: ", q, a, ql, al)
    #guessing_bot.encoder.reset()
    guessing_bot.observe(ques=q, anws=a, ql=ql, al=al)
    idx = guessing_bot(y=idxs)
    real_idx = idxs[idx]
    filename = ut.imgID_2_filename(dataSubType, real_idx)
    print(filename)"""

if __name__ == "__main__":
    main()