from PIL import Image
from vqa import VQA
import torch
import pandas as pd
import numpy as np

DATA_DIR = "/Users/fredericovicente/Desktop/Mestrado/4ºano/2ºSemestre/PW/Experiments/quem_e_quem/assets/images"

def main():
    vqa_object = VQA('mfb')

    image_idx = 1
    question = "what on the stove?"

    #
    image = np.array(Image.open(f'{DATA_DIR}/{image_idx}.jpg').convert('RGB'))
    feats = np.load(f'assets/feats/{image_idx}.npz')

    image_feat = torch.tensor(feats['x'].T)  # (num_objects, 2048)
    print(image_feat)
    # bboxes = torch.tensor(feats['bbox'])  # (num_objects, 4)

    # Get the dict from the net
    ret = vqa_object.inference(question, image_feat)
    print(ret)
    soft_proj = torch.softmax(ret['proj_feat'], dim=-1)
    values, indices = torch.topk(soft_proj, 5)

    values, indices = values.squeeze(0), indices.squeeze(0)


    df = {}
    df['answers'] = []
    df['confidence'] = []

    for idx in range(indices.shape[0]):
        df['answers'].append(vqa_object.ix_to_answer[str(indices[idx].item())])
        df['confidence'].append(100 * values[idx].item())
    df = pd.DataFrame(df)
    print(df)
    # ret['img']['iatt_maps'].squeeze().transpose(1,0)

    # question is the question string, and att is a nd.ndarray of shape (n_glimpses, num_words)
    # ret['text']['qatt'].squeeze().transpose(1,0).detach().numpy()


if __name__ == '__main__':
    main()

