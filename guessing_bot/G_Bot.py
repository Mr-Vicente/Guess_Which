import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from VQ_encoder import QA_Encoder
from torch.autograd import Variable
import bot_utils as ut
import pickle

class G_Bot(nn.Module):
    def __init__(self, params):
        super(G_Bot, self).__init__()
        self.vocabSize = params['vocabSize']
        self.embedSize = params['embedSize']
        self.rnnHiddenSize = params['rnnHiddenSize']
        self.dialogInputSize = params['dialogInputSize']
        self.numLayers = params['numLayers']
        self.imgFeatureSize = params['imgFeatureSize']
        self.numRounds = params['numRounds']

        self.encoder = QA_Encoder(self.vocabSize, self.embedSize, self.rnnHiddenSize, self.dialogInputSize, self.numLayers)
        self.featureNet = nn.Linear(self.rnnHiddenSize, self.imgFeatureSize)
        self.featureNetInputDropout = nn.Dropout(0.5)

        with open('../pickles/token_to_ix.pkl', 'rb') as f:
            self.token_to_ix = pickle.load(f)

        with open('../pickles/resnet_encoding_dic.p', 'rb') as f:
            self.image_encoding = pickle.load(f)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(self.token_to_ix)
        self.reset()

    def reset(self):
        '''Delete dialog history.'''
        self.questions = []
        self.answers = []
        self.encoder.reset()

    def observe(self, ques=None, anws=None, ql=None, al=None):
        if ques is not None:
            self.questions.append(ques)
            self.answers.append(ques)
        self.encoder.observe(-1, ques=ques,quesLens=ql)
        self.encoder.observe(-1,ans=anws, ansLens=al)

    def forward(self, x=None):
        '''
        Forward pass the last observed question
        '''
        encStates = self.encoder()
        imageEnconding = self.predictImage(encStates)
        if x == None:
            x = self.image_encoding.values()
        distances = []
        for enc in x:
            dist = ut.calc_distance(enc, imageEnconding)
            distances.append(dist)
        distances = np.array(distances)
        indices = np.argsort(distances)
        distances_ordered = distances[indices]
        print(distances_ordered)
        return indices[0]

    def predictImage(self,encStates):
        '''
        Predict/guess an fc7 vector given the current conversation history. This can
        be called at round 0 after the caption is observed, and at end of every round
        (after a response from A-Bot is observed).
        '''
        # h, c from lstm
        h, c = encStates
        return self.featureNet(self.featureNetInputDropout(h[-1]))



    def train(self, dataloader, params):
        lRate = 0.001
        #paramters
        opt_parameters = list(self.encoder.parameters()) + list(self.featureNet.parameters())
        optimizer = optim.Adam(opt_parameters, lr=lRate)
        mse_criterion = nn.MSELoss(reduce=False)

        # Iterating over dialog rounds
        def batch_iter(dataloader):
            for epochId in range(params['numEpochs']):
                for idx, batch in enumerate(dataloader):
                    yield epochId, idx, batch

        for epochId, idx, batch in batch_iter(dataloader):
            optimizer.zero_grad()
            loss = 0

            # Moving current batch to GPU, if available
            if self.device == 'cuda':
                batch = {key: v.cuda() if hasattr(v, 'cuda') \
                    else v for key, v in batch.items()}

            questions = Variable(batch['questions'], requires_grad=False)
            questions_lens = Variable(batch['questions_lens'], requires_grad=False)
            answers = Variable(batch['answers'], requires_grad=False)
            answers_lens = Variable(batch['answers_lens'], requires_grad=False)
            target = Variable(batch['target'], requires_grad=False)
            for round in range(self.numRounds):
                print('round: ', round)
                self.encoder.reset()
                self.encoder.observe(round, ques=questions, quesLens=questions_lens)
                self.encoder.observe(round, ans=answers, ansLens=answers_lens)

                encStates = self.encoder()
                image_predictions = self.predictImage(encStates)
                for i in range(len(image_predictions)):
                    image_prediction = image_predictions[i]
                    t_image = target[i].detach().cpu().numpy().tolist()
                    # t_image must exist in image_encoding or else: poof
                    target_image = torch.tensor(self.image_encoding[str(9)])
                    feat_dist = mse_criterion(image_prediction, target_image)
                    feat_dist = torch.mean(feat_dist)
                    loss += feat_dist

            loss = loss / self.numRounds
            loss.backward()
            optimizer.step()
