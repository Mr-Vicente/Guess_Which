from PIL import Image
from vqa import VQA
import torch
import pandas as pd
import numpy as np

def model_work(image_id, question, encoding=None):

    #image = np.array(Image.open(f'{DATA_DIR}/{image_idx}.jpg').convert('RGB'))
    if image_id==None:
        vqa_object = VQA('mfb', 'eval')
        image_feat = torch.tensor([encoding])  # (num_objects, 2048)
        print(image_feat.size())
    else:
        image_idx = image_id[:-4]
        vqa_object = VQA('mfb')
        feats = np.load(f'assets/feats/{image_idx}.npz')
        image_feat = torch.tensor(feats['x'].T)  # (num_objects, 2048)

    ret = vqa_object.inference(question, image_feat)
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
    return df

