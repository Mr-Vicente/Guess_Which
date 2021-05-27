import torch
import torch.nn as nn
from torch.autograd import Variable

import utils

class QA_Encoder(nn.Module):
    def __init__(self, vocabSize, embedSize, rnnHiddenSize, dialogInputSize, numLayers):
           super(QA_Encoder, self).__init__()
           self.vocabSize = vocabSize
           self.embedSize = embedSize
           self.rnnHiddenSize = rnnHiddenSize
           self.numLayers = numLayers
           self.wordEmbed = nn.Embedding(vocabSize, embedSize, padding_idx=0)

           self.factRNN = nn.LSTM(
               self.embedSize,
               self.rnnHiddenSize,
               self.numLayers,
               batch_first=True,
               dropout=0)

           self.dialogRNN = nn.LSTMCell(dialogInputSize, self.rnnHiddenSize)

    def _initHidden(self):
        '''Initial dialog rnn state - initialize with zeros'''
        # Dynamic batch size inference
        assert self.batchSize != 0, 'Observe something to infer batch size.'
        someTensor = self.dialogRNN.weight_hh.data
        h = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        c = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        return (Variable(h), Variable(c))

    def reset(self):
        # batchSize is inferred from input
        self.batchSize = 0

        # Input data

        self.questionTokens = []
        self.questionEmbeds = []
        self.questionLens = []

        self.answerTokens = []
        self.answerEmbeds = []
        self.answerLengths = []

        # Hidden embeddings
        self.factEmbeds = []
        self.questionRNNStates = []
        self.dialogRNNInputs = []
        self.dialogHiddens = []

    def observe(self,
                round,
                ques=None,
                ans=None,
                quesLens=None,
                ansLens=None):
        '''
        Store dialog input to internal model storage
        Note that all input sequences are assumed to be left-aligned (i.e.
        right-padded). Internally this alignment is changed to right-align
        for ease in computing final time step hidden states of each RNN
        '''
        if ques is not None:
            assert round == len(self.questionEmbeds)
            assert quesLens is not None, "Questions lengths required!"
            ques, quesLens = self.processSequence(ques, quesLens)
            self.questionTokens.append(ques)
            self.questionLens.append(quesLens)
            self.batchSize = 2#len(self.questionTokens)
        if ans is not None:
            assert round == len(self.answerEmbeds)
            assert ansLens is not None, "Answer lengths required!"
            ans, ansLens = self.processSequence(ans, ansLens)
            self.answerTokens.append(ans)
            self.answerLengths.append(ansLens)

    def processSequence(self, seq, seqLen):
        ''' Strip <START> and <END> token from a left-aligned sequence'''
        return seq[:, 1:], seqLen - 1

    def embedInputDialog(self):
        '''
        Lazy embedding of input:
            Calling observe does not process (embed) any inputs. Since
            self.forward requires embedded inputs, this function lazily
            embeds them so that they are not re-computed upon multiple
            calls to forward in the same round of dialog.
        '''
        # Embed questions
        while len(self.questionEmbeds) < len(self.questionTokens):
            idx = len(self.questionEmbeds)
            self.questionEmbeds.append(
                self.wordEmbed(self.questionTokens[idx]))
        # Embed answers
        while len(self.answerEmbeds) < len(self.answerTokens):
            idx = len(self.answerEmbeds)
            self.answerEmbeds.append(self.wordEmbed(self.answerTokens[idx]))

    def embedFact(self, factIdx):
        '''Embed facts i.e. round 0 or question-answer pair otherwise'''
        # QA pairs
        print('FactIdx: ', factIdx)
        quesTokens, quesLens = \
            self.questionTokens[factIdx], self.questionLens[factIdx]
        ansTokens, ansLens = \
            self.answerTokens[factIdx], self.answerLengths[factIdx]

        qaTokens = utils.concatPaddedSequences(
            quesTokens, quesLens, ansTokens, ansLens, padding='right')
        qa = self.wordEmbed(qaTokens)
        qaLens = quesLens + ansLens
        qaEmbed, states = utils.dynamicRNN(
            self.factRNN, qa, qaLens, returnStates=True)
        factEmbed = qaEmbed
        factRNNstates = states
        self.factEmbeds.append((factEmbed, factRNNstates))

    def embedQuestion(self, qIdx):
        '''Embed questions'''
        quesIn = self.questionEmbeds[qIdx]
        quesLens = self.questionLens[qIdx]
        if self.useIm == 'early':
            image = self.imageEmbed.unsqueeze(1).repeat(1, quesIn.size(1), 1)
            quesIn = torch.cat([quesIn, image], 2)
        qEmbed, states = utils.dynamicRNN(
            self.quesRNN, quesIn, quesLens, returnStates=True)
        quesRNNstates = states
        self.questionRNNStates.append((qEmbed, quesRNNstates))

    def concatDialogRNNInput(self, histIdx):
        currIns = [self.factEmbeds[histIdx][0]]
        '''if self.isAnswerer:
            currIns.append(self.questionRNNStates[histIdx][0])
        if self.useIm == 'late':
            currIns.append(self.imageEmbed)'''
        hist_t = torch.cat(currIns, -1)
        self.dialogRNNInputs.append(hist_t)

    def embedDialog(self, dialogIdx):
        if dialogIdx == 0:
            hPrev = self._initHidden()
        else:
            hPrev = self.dialogHiddens[-1]
        inpt = self.dialogRNNInputs[dialogIdx]
        hNew = self.dialogRNN(inpt, hPrev)
        self.dialogHiddens.append(hNew)

    def forward(self):
        '''
        Returns:
            A tuple of tensors (H, C) each of shape (batchSize, rnnHiddenSize)
            to be used as the initial Hidden and Cell states of the Decoder.
            See notes at the end on how (H, C) are computed.
        '''

        # Lazily embed input Image, Captions, Questions and Answers
        self.embedInputDialog()

        round = len(self.questionEmbeds) - 1

        # Lazy computation of internal hidden embeddings (hence the while loops)

        # Infer any missing facts
        while len(self.factEmbeds) <= round:
            factIdx = len(self.factEmbeds)
            self.embedFact(factIdx)

        '''
        while len(self.questionRNNStates) <= round:
            qIdx = len(self.questionRNNStates)
            self.embedQuestion(qIdx)
        '''
        # Concat facts and/or questions (i.e. history) for input to dialogRNN
        while len(self.dialogRNNInputs) <= round:
            histIdx = len(self.dialogRNNInputs)
            self.concatDialogRNNInput(histIdx)

        # Forward dialogRNN one step
        while len(self.dialogHiddens) <= round:
            dialogIdx = len(self.dialogHiddens)
            self.embedDialog(dialogIdx)

        # Latest dialogRNN hidden state
        dialogHidden = self.dialogHiddens[-1][0]

        '''
        Return hidden (H_link) and cell (C_link) states as per the following rule:
        (Currently this is defined only for numLayers == 2)
        C_link == Fact encoding RNN cell state (factRNN)
        H_link ==
            Layer 0 : Fact encoding RNN hidden state (factRNN)
            Layer 1 : DialogRNN hidden state (dialogRNN)
        '''
        factRNNstates = self.factEmbeds[-1][1]  # Latest factRNN states
        C_link = factRNNstates[1]
        H_link = factRNNstates[0][:-1]
        H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)

        return H_link, C_link