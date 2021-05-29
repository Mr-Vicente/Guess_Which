import torch as T
import numpy as np
import os
import preprocess as p
import math

def find_valid_images_idx(path):
    filenames = os.listdir(path)
    images_idx = np.array([filename_2_idx(filename) for filename in filenames])
    return images_idx

def filename_2_idx(filename):
    return int(filename.split('_')[-1][:-4])

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

def stack_images(images_path, dataSubType, pool_size):
    pool_image_indeces = find_valid_images_idx(images_path)
    print('pool_image_indeces: ', pool_image_indeces)
    n_images = len(pool_image_indeces)
    pool_indeces = pool_images(n_images, pool_size)
    print('pool_indeces: ', pool_indeces)
    pool_image_indeces = pool_image_indeces[pool_indeces]
    print('pool_image_indeces: ', pool_image_indeces)
    selected_image_idx = pool_image_indeces[0]
    images_filenames = np.array([imgID_2_filename(dataSubType, imgId) for imgId in pool_image_indeces])
    return selected_image_idx, images_filenames

def random_pool_input_generator(images_path, dataSubType, vqa, pool_size):
    selected_image_idx, images_filenames = stack_images(images_path, dataSubType, pool_size)
    annIds = vqa.getQuesIds(imgIds=[selected_image_idx])
    anns = vqa.loadQA(annIds)
    input_data = {
        'pool_idx': images_filenames,
        'questions': annIds,
        'answers': anns
    }
    output_onehot = np.zeros(pool_size)
    output_onehot[selected_image_idx] = 1
    return input_data, output_onehot

def multiple_input(dataSubType, n_images, vqa, pool_size):
    inputs = []
    for i in range(n_images):
        single_input, target = random_pool_input_generator(dataSubType, n_images, vqa, pool_size)
        inputs.append(single_input)
    inputs = np.array(inputs)
    return inputs

def calc_distance(img_enc_sel, img_enc_inf):
    return math.sqrt(((img_enc_sel-img_enc_inf)**2).sum(axis=0))


# ---------------------------------------------------

class VQDataset(T.utils.data.Dataset):

  def __init__(self, Xs, Ys, token_to_ix):
    self.device = 'cuda' if T.cuda.is_available() else 'cpu'
    self.questions, self.answers, self.questions_lens, self.answers_lens\
        = self.convert_dict_to_vec(Xs, token_to_ix)

    #self.x_data = [questions, answers]   #T.tensor([questions, answers]).to(self.device)
    self.y_data = Ys                    #T.tensor(Ys).to(self.device)

  def __len__(self):
    return len(self.questions)  # required

  def __getitem__(self, idx):
    if T.is_tensor(idx):
        idx = idx.tolist()
    questions = self.questions[idx]
    answers = self.answers[idx]
    questions_lens = self.questions_lens[idx]
    answers_lens = self.answers_lens[idx]
    target = self.y_data[idx]
    sample = {
        'questions' : questions,
        'answers': answers,
        'questions_lens' : questions_lens,
        'answers_lens': answers_lens,
        'target' : target
    }
    return sample

  def convert_dict_to_vec(self, Xs, token_to_ix):
    proc_statement = p.proc_ques
    questions = []
    questions_lens = []
    answers = []
    answers_lens = []
    for X in Xs:
      qs = []
      aws = []
      for qa in X['QAs']:
          question = qa[0]
          answer = qa[1]
          ques_ix = proc_statement(question, token_to_ix, max_token=14)
          ans_ix = proc_statement(answer, token_to_ix, max_token=14)
          qs.append(ques_ix)
          aws.append(ans_ix)
      qs = qs[:3]
      aws = aws[:3]
      questions.append(qs)
      answers.append(aws)
      questions_lens.append(len(qs))
      answers_lens.append(len(aws))
    questions, answers, questions_lens, answers_lens = np.array(questions), np.array(answers),\
                                                         np.array(questions_lens), np.array(answers_lens)

    return T.tensor(questions), T.tensor(answers), T.tensor(questions_lens), T.tensor(answers_lens)



# ---------------------------------------------------


def convert_statements_to_idx(question, answer, token_to_ix):
    proc_statement = p.proc_ques
    ques_ix = proc_statement(question, token_to_ix, max_token=14)
    ans_ix = proc_statement(answer, token_to_ix, max_token=14)
    return ques_ix, ans_ix

