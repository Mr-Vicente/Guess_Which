
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

def getSortedOrder(lens):
    sortedLen, fwdOrder = torch.sort(
        lens.contiguous().view(-1), dim=0, descending=True)
    _, bwdOrder = torch.sort(fwdOrder)
    if isinstance(sortedLen, Variable):
        sortedLen = sortedLen.data
    sortedLen = sortedLen.cpu().numpy().tolist()
    return sortedLen, fwdOrder, bwdOrder

def dynamicRNN(rnnModel,
               seqInput,
               seqLens,
               initialState=None,
               returnStates=False):
    '''
    Inputs:
        rnnModel     : Any torch.nn RNN model
        seqInput     : (batchSize, maxSequenceLength, embedSize)
                        Input sequence tensor (padded) for RNN model
        seqLens      : batchSize length torch.LongTensor or numpy array
        initialState : Initial (hidden, cell) states of RNN
    Output:
        A single tensor of shape (batchSize, rnnHiddenSize) corresponding
        to the outputs of the RNN model at the last time step of each input
        sequence. If returnStates is True, also return a tuple of hidden
        and cell states at every layer of size (num_layers, batchSize,
        rnnHiddenSize)
    '''
    sortedLen, fwdOrder, bwdOrder = getSortedOrder(seqLens)
    sortedSeqInput = seqInput.index_select(dim=0, index=fwdOrder)
    packedSeqInput = pack_padded_sequence(
        sortedSeqInput, lengths=sortedLen, batch_first=True)
    print(packedSeqInput)

    if initialState is not None:
        hx = initialState
        sortedHx = [x.index_select(dim=1, index=fwdOrder) for x in hx]
        assert hx[0].size(0) == rnnModel.num_layers  # Matching num_layers
    else:
        hx = None
    _, (h_n, c_n) = rnnModel(packedSeqInput, hx)

    rnn_output = h_n[-1].index_select(dim=0, index=bwdOrder)

    if returnStates:
        h_n = h_n.index_select(dim=1, index=bwdOrder)
        c_n = c_n.index_select(dim=1, index=bwdOrder)
        return rnn_output, (h_n, c_n)
    else:
        return rnn_output

def concatPaddedSequences(seq1, seqLens1, seq2, seqLens2, padding='right'):
    '''
    Concates two input sequences of shape (batchSize, seqLength). The
    corresponding lengths tensor is of shape (batchSize). Padding sense
    of input sequences needs to be specified as 'right' or 'left'
    Args:
        seq1, seqLens1 : First sequence tokens and length
        seq2, seqLens2 : Second sequence tokens and length
        padding        : Padding sense of input sequences - either
                         'right' or 'left'
    '''

    concat_list = []
    cat_seq = torch.cat([seq1, seq2], dim=1)
    maxLen1 = seq1.size(1)
    maxLen2 = seq2.size(1)
    maxCatLen = cat_seq.size(1)
    batchSize = seq1.size(0)
    for b_idx in range(batchSize):
        len_1 = seqLens1[b_idx]
        len_2 = seqLens2[b_idx]

        cat_len_ = len_1 + len_2
        if cat_len_ == 0:
            raise RuntimeError("Both input sequences are empty")

        elif padding == 'left':
            pad_len_1 = maxLen1 - len_1
            pad_len_2 = maxLen2 - len_2
            if len_1 == 0:
                print("[Warning] Empty input sequence 1 given to "
                      "concatPaddedSequences")
                cat_ = seq2[b_idx][pad_len_2:]

            elif len_2 == 0:
                print("[Warning] Empty input sequence 2 given to "
                      "concatPaddedSequences")
                cat_ = seq1[b_idx][pad_len_1:]

            else:
                cat_ = torch.cat([seq1[b_idx][pad_len_1:],
                                  seq2[b_idx][pad_len_2:]], 0)
            cat_padded = F.pad(
                input=cat_,  # Left pad
                pad=((maxCatLen - cat_len_), 0),
                mode="constant",
                value=0)
        elif padding == 'right':
            if len_1 == 0:
                print("[Warning] Empty input sequence 1 given to "
                      "concatPaddedSequences")
                cat_ = seq2[b_idx][:len_1]

            elif len_2 == 0:
                print("[Warning] Empty input sequence 2 given to "
                      "concatPaddedSequences")
                cat_ = seq1[b_idx][:len_1]

            else:
                cat_ = torch.cat([seq1[b_idx][:len_1],
                                  seq2[b_idx][:len_2]], 0)
                # cat_ = cat_seq[b_idx].masked_select(cat_seq[b_idx].ne(0))
            cat_padded = F.pad(
                input=cat_,  # Right pad
                pad=(0, (maxCatLen - cat_len_)),
                mode="constant",
                value=0)
        else:
            raise (ValueError, "Expected padding to be either 'left' or \
                                'right', got '%s' instead." % padding)
        #concat_list.append(cat_padded.unsqueeze(0))
        concat_list.append(cat_padded)
    concat_output = torch.cat(concat_list, 0)
    return concat_output