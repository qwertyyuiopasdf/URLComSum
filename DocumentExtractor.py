import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from torch.nn import Parameter
import torch
import gensim.downloader as api
from nltk import word_tokenize, sent_tokenize
from typing import Tuple
import numpy as np

PAD = 0
UNK = 1
START = 2
END = 3
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'

class DocumentExtractor(nn.Module):

    def __init__(self, tkner, embedding_dim, hidden_dim, num_attention_head=5, dropout_rate=0.5):
        super(DocumentExtractor, self).__init__()
        self.tkner = tkner
        if self.tkner == 'w2v':
            with open('vocab_npa.npy','rb') as f:
                self.id2word = dict(enumerate(np.load(f), 1))
                self.word2id = {v: k for k, v in self.id2word.items()}
            with open('embs_npa.npy','rb') as f:
                embs_npa = np.load(f)
            self.embmodel = nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float(), padding_idx=0)
            #self.embmodel = api.load('word2vec-google-news-300')
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.embmodel = BertModel.from_pretrained('bert-base-uncased',  output_hidden_states = True)
            self.embmodel.eval()
            self.embmodel.cuda()

        self.embedding_dim = torch.from_numpy(embs_npa).size(1)
        self.vocab_size = torch.from_numpy(embs_npa).size(0)

        self.hidden_dim = int(self.embedding_dim/2)
        self.wordlstm1 = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=2, bidirectional=True, batch_first = True)
        self.wordattn = nn.MultiheadAttention(embed_dim=self.hidden_dim*2, num_heads=num_attention_head, dropout=dropout_rate)
        self.word_layer_norm = nn.LayerNorm(self.hidden_dim*2)
        self.wordlstm2= nn.LSTM(self.hidden_dim*4, self.hidden_dim, num_layers=2, bidirectional=False, batch_first = True)

        self.lstm1 = nn.LSTM(self.hidden_dim*2, self.hidden_dim, num_layers=2, bidirectional=True, batch_first = True)
        self.label_attn1 = nn.MultiheadAttention(embed_dim=self.hidden_dim*2, num_heads=num_attention_head, dropout=dropout_rate)
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim*2)
        self.lstm2= nn.LSTM(self.hidden_dim*4, self.hidden_dim, num_layers=2, bidirectional=True, batch_first = True)
        self.label_attn2 = nn.MultiheadAttention(embed_dim=self.hidden_dim*2, num_heads=num_attention_head)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim*2)
        self.lstm3= nn.LSTM(self.hidden_dim*4, self.hidden_dim, num_layers=2, bidirectional=False, batch_first = True)

        self.decoder = Decoder(self.hidden_dim*2, self.hidden_dim)
        self.decoder_input0 = Parameter(torch.FloatTensor(self.hidden_dim*2), requires_grad=False)
        nn.init.uniform(self.decoder_input0, -1, 1)

    def forward(self, document, summary_size, max_sentence_size=30, max_article_size=30):
        
        original_documents = []
        embs = []
        pad_tokenized_texts = []
        tokenized_texts = []
        for doc in document:
            ss = []
            tokenized_sents = []

            sentences = sent_tokenize(doc)
            original_documents.append(sentences)

            for sentence in sentences:
                tokenized_sent = word_tokenize(sentence)
                ls = max(0, max_sentence_size - len(tokenized_sent))

                convert_sentence = [convert_word2id(w,self.word2id) for w in tokenized_sent] + [PAD] * ls
                tokenized_sent = tokenized_sent + [PAD_TOKEN] * ls

                tokenized_sents.append(tokenized_sent[:max_sentence_size])
                ss.append(convert_sentence[:max_sentence_size])

            ss = ss[::-1][:max_article_size][::-1]
            lm = max(0, max_article_size - len(ss))

            for _ in range(lm):
                ss.append([PAD] * max_sentence_size)
                tokenized_sents.append([PAD_TOKEN] * max_sentence_size)
            pad_tokenized_texts.append(torch.tensor(ss))
            tokenized_texts.append(tokenized_sents)

        pad_tokenized_texts = torch.stack(pad_tokenized_texts).cuda().long()

        word_lstm_hidden_list = []

        pad_tokenized_texts = pad_tokenized_texts.permute(1, 0, 2)

        for sent in pad_tokenized_texts:
            embs = self.embmodel(sent)
            word_lstm_out, (word_lstm_hidden, word_lstm_cell) = self.wordlstm1(embs)
            word_attn_output, word_attn_output_weights = self.wordattn(word_lstm_out.permute(1, 0, 2), embs.permute(1, 0, 2), embs.permute(1, 0, 2))
            word_attn_output = self.word_layer_norm(word_attn_output)
            word_lstm_out = torch.cat([word_lstm_out, word_attn_output.permute(1, 0, 2)], -1)
            word_lstm_out, (word_lstm_hidden, word_lstm_cell) = self.wordlstm2(word_lstm_out)

            word_lstm_hidden_list.append(word_lstm_hidden.view(-1, 2*self.hidden_dim))

        

        sent_s = torch.stack(word_lstm_hidden_list, 1)

        batch_size = sent_s.size(0)
        input_length = sent_s.size(1)

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)
        seq_size = embs.size()
        lstm_out, (lstm_hidden, lstm_cell) = self.lstm1(sent_s)
        attn_output, attn_output_weights = self.label_attn1(lstm_out.permute(1, 0, 2), sent_s.permute(1, 0, 2), sent_s.permute(1, 0, 2))
        attn_output = self.layer_norm1(attn_output)
        lstm_out = torch.cat([lstm_out, attn_output.permute(1, 0, 2)], -1)
        lstm_out, (lstm_hidden, lstm_cell) = self.lstm2(lstm_out)
        attn_output, attn_output_weights = self.label_attn2(lstm_out.permute(1, 0, 2), sent_s.permute(1, 0, 2), sent_s.permute(1, 0, 2))
        attn_output = self.layer_norm2(attn_output)
        lstm_out = torch.cat([lstm_out, attn_output.permute(1, 0, 2)], -1)
        
        lstm_out, (lstm_hidden, lstm_cell) = self.lstm3(lstm_out)

        idx_batch = torch.zeros(lstm_out.size(0), input_length, dtype=torch.long)
        dec_in = torch.zeros(lstm_out.size(0), 1, self.hidden_dim, dtype=torch.float).cuda()
        encoder_hidden = (lstm_hidden, lstm_cell)
        decoder_hidden0 = (encoder_hidden[0][-1], encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(sent_s,decoder_input0,decoder_hidden0, lstm_out, 10)

        summaries = []

        valid_pointer_batch = []
        for doc, idx in zip(original_documents, pointers):
            valid_idx = []
            for pos in idx:
                if int(pos) < len(doc):
                   valid_idx.append(int(pos))
                if len(valid_idx) >= summary_size or len(valid_idx) >= len(doc):
                    break

            
            valid_idx = sorted(valid_idx)
            valid_pointer_batch.append(valid_idx)
            summary = []
            for pos in valid_idx:
                summary.append(doc[int(pos)])
            summaries.append(" ".join(summary))

        outputs = torch.mean(outputs, dim=-1)

        return summaries, outputs, valid_pointer_batch


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim):

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)

        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context, summary_length):
        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            h, c = hidden
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = F.sigmoid(input)
            forget = F.sigmoid(forget)
            cell = F.tanh(cell)
            out = F.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * F.tanh(c_t)
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = F.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        for _ in range(input_length):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)
            masked_outs = outs * mask

            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            mask  = mask * (1 - one_hot_pointers)

            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden


class Attention(nn.Module):

    def __init__(self, input_dim,
                 hidden_dim):
        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        nn.init.uniform(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)

def convert_word2id(w, word2id):
    try:
        return word2id[w]
    except:
        return UNK