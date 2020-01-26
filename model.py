import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

#T =  temperature for softmax

def masked_softmax(vector, mask, dim  = -1, memory_efficient = False, mask_fill_value =  -1e32, T = 1):
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector/T, dim=dim)
    return result

class HierarchicalFBMA(nn.Module):
    """
    Hierarchical Attn model with word level multi-hop attn
    """
    def __init__(self, args):
        super(HierarchicalFBMA, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.emb_dim, padding_idx=0)
        self._sent_encoder_gru = GRUEncoder(args.emb_dim, args.lstm_hidden, args.bidirectional, self.embedding)
        self._doc_encoder_gru = GRUEncoder(args.aspects*args.lstm_hidden*(2 if args.bidirectional else 1), args.lstm_hidden, args.bidirectional)
        self.word_attn = FBMA(args.lstm_hidden, args.aspects, args.bidirectional)
        self.sent_attn = IntraAttention(args.lstm_hidden, args.bidirectional)
        self._sent_classifier = SentClassifier(args.lstm_hidden*(2 if args.bidirectional else 1), args.mlp_nhid, args.dropout, args.nclass) #300+500
        self._softmax = torch.nn.Softmax()

    def forward(self, inp, lens):
        in_ = inp.view(-1, inp.size()[-1]) #batch_size*num_sents x num_words
        gru_out_word = self._sent_encoder_gru(in_) #batch_size*num_sents x num_words x hid_dim 229x70x100
        word_mask = ~(inp.view(inp.size(0)*inp.size(1), -1)==0).unsqueeze(1).repeat(1, self.args.aspects, 1)  # mask out 0 padded words 229x num_hopsx num_words

        word_attn_weights = self.word_attn(gru_out_word, word_mask) # batch_size*num_sents x attn_hops x num_words 229x10x50

        sent_emb = torch.bmm(word_attn_weights, gru_out_word)

        sent_emb = sent_emb.view(in_.size(0), sent_emb.size(1)*sent_emb.size(2))

        # sent_emb = torch.sum(torch.mul(word_attn_weights.unsqueeze(2).repeat(1, 1, gru_out_word.size()[-1]).transpose(2,1), gru_out_word.transpose(2,1)), dim=2)  #224 x 500
        sent_emb = sent_emb.view(inp.size(0), inp.size(1), -1)

        gru_out_sent = self._doc_encoder_gru(sent_emb)

        sent_mask = ~(torch.sum(~(inp==0), dim=2)==0)
        sent_attn_weights = self.sent_attn(gru_out_sent, sent_mask)

        # weighted average of sent embs
        doc_emb = torch.sum(torch.mul(sent_attn_weights.unsqueeze(2).repeat(1, 1, gru_out_sent.size()[-1]).transpose(2,1), gru_out_sent.transpose(2,1)), dim=2)  #224 x 5
        logits = self._sent_classifier(doc_emb)

        return logits, word_attn_weights, sent_attn_weights #sigmoid is for binary

    def set_embedding(self, embedding):
        assert self.embedding.weight.size() == embedding.size()
        self.embedding.weight.data.copy_(embedding.data)
        self._sent_encoder_gru.set_embedding(embedding)


class GRUEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, bidirectional=False, embedding=None, dropout=0.4):
        super(GRUEncoder, self).__init__()
        self.embedding = embedding
        self.gru = nn.GRU(emb_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)

    def forward(self, inp, init_states=None):
        batch_size = inp.size(0)  #bsz x num_sents
        emb_sequence = (self.embedding(inp) if self.embedding is not None
                            else inp)
        device = inp.device
        init_states = self.init_lstm_states(batch_size, device)

        gru_out, final_state = self.gru(emb_sequence, init_states)
        return gru_out

    def init_lstm_states(self, batch_size, device):
        return torch.zeros((2 if self.gru.bidirectional else 1), batch_size, self.gru.hidden_size).to(device)

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self.embedding.weight.size() == embedding.size()
        self.embedding.weight.data.copy_(embedding.data)


class SentClassifier(nn.Module):
    """
    Two layer MLP
    """
    def __init__(self, emb_size, nhid, dropout, nclasses):
        super(SentClassifier, self).__init__()
        self.hidden = nn.Linear(emb_size, nhid)
        self.hidden2op = nn.Linear(nhid, nclasses, bias=True)
        self.nclasses = nclasses
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp): #inp = [bsz, sent_emb+doc_emb]
        hid = self.dropout(self.hidden(inp))
        return self.hidden2op(hid)


class FBMA(nn.Module):
    """
    Factorized Bilinear Multi-Aspect Attention Module.
    """
    def __init__(self, hidden_dim, num_aspect, bidirectional=True, u_w_dim=32, proj_word=None):
        super(FBMA, self).__init__()
        self.hidden_dim = hidden_dim
        self.u_it_W = nn.Parameter(torch.Tensor(hidden_dim * (2 if bidirectional else 1), hidden_dim * (2 if bidirectional else 1)))
        self.u_it_b = nn.Parameter(torch.Tensor(hidden_dim * (2 if bidirectional else 1), 1))
        self.P =  nn.Parameter(torch.Tensor(hidden_dim * (2 if bidirectional else 1), num_aspect)) # 10 op attention
        self.Q =  nn.Parameter(torch.Tensor(u_w_dim, num_aspect))
        self.dropout = nn.Dropout(p=0.4)
        self.u_w = nn.Parameter(torch.Tensor(u_w_dim, 1))
        self.softmax_word = nn.Softmax(dim=2)
        self.u_it_W.data.uniform_(-0.1, 0.1)
        # self.weight_proj_word.data.uniform_(-0.1, 0.1)
        self.u_it_b.data.uniform_(-0.1, 0.1)
        nn.init.xavier_normal_(self.u_w)
        self.P.data.uniform_(-0.1, 0.1)
        self.Q.data.uniform_(-0.1, 0.1)

    def forward(self, inp, mask):
        word_squish = torch.tanh(torch.bmm(self.u_it_W.repeat(inp.size(0), 1, 1), inp.transpose(2,1)) + self.u_it_b.repeat(inp.size(0), 1, inp.size(1)))
        # word_squish = bszx hid_dim x num_words
        PTx = torch.bmm(self.P.repeat(inp.size(0),1, 1).transpose(2,1), word_squish)
        QTy = torch.bmm(self.Q.repeat(inp.size(0),1, 1).transpose(2,1), self.u_w.repeat(inp.size(0), 1, inp.size(1)))
        word_attn_logits = torch.tanh(torch.mul(PTx, QTy)) # tanh removed
        word_attn_logits_norm = torch.norm(word_attn_logits, p=2, dim=1, keepdim=True) # normalization accross the hop dimension
        word_attn_logits = word_attn_logits.div(word_attn_logits_norm)
        word_attn = masked_softmax(word_attn_logits, mask.float(), dim=2)
        return word_attn #multi-hop word attn

class IntraAttention(nn.Module):
    """
    From HAN
    """
    def __init__(self, hidden_dim, bidirectional=True):
        super(IntraAttention, self).__init__()
        self.weight_W_word = nn.Parameter(torch.Tensor(hidden_dim * (2 if bidirectional else 1), hidden_dim * (2 if bidirectional else 1)))
        self.bias_word = nn.Parameter(torch.Tensor(hidden_dim * (2 if bidirectional else 1), 1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(hidden_dim * (2 if bidirectional else 1), 1))
        self.softmax_word = nn.Softmax(dim=1)
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.bias_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1,0.1)

    def forward(self, inp, mask):
        # dim of inp = 224 x 50 x hid_dim  repeat word weight for batchsize and perform parallel computation
        word_squish = torch.tanh(torch.bmm(self.weight_W_word.repeat(inp.size(0), 1, 1), inp.transpose(2,1)) + self.bias_word.repeat(inp.size(0), 1, inp.size(1)))
        # word_squish = bszx hid_dim x num_words
        word_attn_logits = torch.bmm(word_squish.transpose(2,1), self.weight_proj_word.repeat(inp.size(0), 1, 1)) #repeat the projection word for bsz times
        word_attn = masked_softmax(word_attn_logits.squeeze(2), mask.float(), dim=1)
        # mask attn weights
        return word_attn
