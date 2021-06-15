from nltk.tokenize import word_tokenize
from guessing_bot.VQA_API import VQA
import guessing_bot.bot_utils as ut
import vqa_main
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torch
from torchvision import transforms

##################################
#        VQA API SETUP
##################################

dataDir		=   '../../VQA_dataset'
versionType =   'v2_' # this should be '' when using VQA v2.0 dataset
taskType    =   'OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    =   'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType =   'train2014'
annFile     =   f'{dataDir}/Annotations/{versionType}{dataType}_{dataSubType}_annotations.json'
quesFile    =   f'{dataDir}/Questions/{versionType}{taskType}_{dataType}_{dataSubType}_questions.json'
imgDir 		=   f'{dataDir}/Images/{dataType}/{dataSubType}/'

##################################
#        Gather dataset
##################################

def load_dataset(n_elementes = 10):
    vqa = VQA(annFile, quesFile)
    valid_image_idx = ut.find_valid_images_idx(imgDir)[:n_elementes].tolist()
    images_answers = []
    for image_idx in valid_image_idx:
        annIds = vqa.getQuesIds(imgIds=[image_idx])
        model_input = {
            'image_idx': image_idx,
            'answers': [],
            'questions': []
        }
        for q_i in range(len(annIds)):
            ann = vqa.loadQA(annIds[q_i])[0]
            q, a = vqa.get_qa(ann)
            model_input['answers'].append(a)
            model_input['questions'].append(q)
        images_answers.append(model_input)
    return images_answers, valid_image_idx

def resnet152():
    model = models.resnet152(pretrained=True)
    model.eval()
    model.fc = nn.Identity()
    return model

def load_image(image_idx):
    filename = ut.imgID_2_filename(dataSubType,image_idx)
    img_path = imgDir + filename
    img = Image.open(img_path).convert('RGB')
    return img


def jaccard_similarity(answer_pred, answer_target):
    answer_pred, answer_target = set(answer_pred), set(answer_target)
    n_intersection = float(len(answer_pred.intersection(answer_target)))
    n_union = len(answer_pred.union(answer_target))
    return n_intersection / n_union

img_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3)     ,
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def eval():
    images_content, valid_image_idx = load_dataset(1000)
    resnet = resnet152()
    image_scoring = {}
    for content in images_content:
        idx = content['image_idx']
        image = load_image(idx)
        image = img_transform(image)
        image = torch.unsqueeze(image, 0)
        qs = content['questions']
        aws = content['answers']
        encoding = resnet(image)
        encoding = encoding[0].detach().numpy()
        scores = []
        for question, target_answer in zip(qs,aws):
            df = vqa_main.model_work(None, question, encoding)
            answer_pred = df['answers'][0]
            answer_tokens_pred = word_tokenize(answer_pred)
            answer_tokens_target = word_tokenize(target_answer)
            score = jaccard_similarity(answer_tokens_pred, answer_tokens_target)
            scores.append(score)
        image_score = sum(scores)/len(scores)
        image_scoring[str(idx)] = image_score
    print(image_scoring)


if __name__ == "__main__":
    eval()

