import numpy as np
import torch, os, time
from scipy.spatial.distance import cosine
from sklearn.cluster import SpectralClustering
from transformers.modeling_bert import BertModel
from transformers.tokenization_bert import BertTokenizer
import ot
import pickle as pkl
import nltk
import math
import statistics
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
from itertools import combinations

nltk.download('punkt')

BERT_NUM_TOKEN = 30522
def cosine_similarity(a, b):
    return (a @ b.T) / (np.linalg.norm(a)*np.linalg.norm(b))

def construct_BOW(tokens):
    bag_vector = np.zeros(BERT_NUM_TOKEN)        
    for token in tokens:            
        bag_vector[token] += 1                            
    return bag_vector/len(tokens)



class CoverageScorer:
    # Depending on how many words are used a large fraction of the last X summaries
    def __init__(self, tkner="bert", device="cuda", costmatrix_filename="COST_MATRIX_bert.pickle"):
    #def __init__(self, device="cpu", costmatrix_filename="COST_MATRIX.pickle"):
        
        self.tkner = tkner

        if self.tkner == 'w2v':
            self.model = api.load('word2vec-google-news-300')
            
        else:
            self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
            self.model.eval()
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            max_bytes = 2**31 - 1

            bytes_in = bytearray(0)
            input_size = os.path.getsize(costmatrix_filename)
            with open(costmatrix_filename, 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)

                self.COST_MATRIX = pkl.loads(bytes_in)

    def score(self, summaries, bodies, idx_batch=None, bodies_tokenized=None, lengths=None, extra=None):
        scores = []
        with torch.no_grad():
            for i in range(len(summaries)):
                summary = summaries[i]
                doc = bodies[i]
                if len(summary)==0:
                    score = 1
                else:
                    if self.tkner == 'w2v':
                        s = self.model.wmdistance(doc, summary)
                        if math.isinf(s) or math.isnan(s):
                            score = 1.0
                        else:
                            score = s

                    else:
                        summary_token = self.tokenizer.encode(summary) 
                        body_token = self.tokenizer.encode(doc)

                        summary_bow = construct_BOW(summary_token)
                        body_bow = construct_BOW(body_token)

                        score = sparse_ot(summary_bow, body_bow, self.COST_MATRIX) 

                scores.append(1-score)
            
        return scores, None

def BERT_embedding(model, token):
    with torch.no_grad():
        #outputs = model(torch.tensor([tokenizer.encode(text)]))
        outputs = model(torch.tensor([[token]]))
        hidden_states = outputs[2]

        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        token_vecs_sum = []

        for i in token_embeddings:
            sum_vec = torch.sum(i[-4:], dim=0)
        
            token_vecs_sum.append(np.array(sum_vec))
#    print("token_vecs_sum", token_vecs_sum)
    return token_vecs_sum
    #return token_vecs_sum


def sparse_ot(weights1, weights2, M):
    """ Compute Wasserstein distances"""
    
    weights1 = weights1/weights1.sum()
    weights2 = weights2/weights2.sum()
    
    active1 = np.where(weights1)[0]
    active2 = np.where(weights2)[0]
    
    weights_1_active = weights1[active1]
    weights_2_active = weights2[active2]
    try1 = M[active1][:,active2]
    M_reduced = np.ascontiguousarray(M[active1][:,active2])
    
    return ot.emd2(weights_1_active,weights_2_active,M_reduced)



import nltk

nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize
import re
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_string

import pandas as pd
import numpy as np

import gensim.downloader as api

vocab_size = 50000

if __name__ == "__main__":

    wv = api.load('word2vec-google-news-300')
    word2id={}
    id2word={}
    for i, (w, _) in enumerate(wv.most_common(vocab_size), 4):
        word2id[w] = i
        id2word[i] = w

    TOKEN_EMBED={}
    for i in range(len(wv.index_to_key)):
        TOKEN_EMBED[i]=wv[id2word[i]]

    COST_MATRIX = np.zeros((vocab_size, vocab_size))
    #with open('TOKEN_EMBED.pickle', 'rb') as handle:
    #    TOKEN_EMBED = pkl.load(handle)

    for i in range(vocab_size):
        for j in range(vocab_size):
            if i == j:
                COST_MATRIX[i,j] = 0
            elif i < j:
                COST_MATRIX[i,j] = cosine(TOKEN_EMBED[i], TOKEN_EMBED[j])
            elif j < i:
                COST_MATRIX[i,j] = COST_MATRIX[j,i]
        print("i", i)

    with open('COST_MATRIX_w2v.pickle', 'wb') as handle:
        pkl.dump(COST_MATRIX, handle , protocol=4)

    input_size = os.path.getsize("COST_MATRIX.pickle")
    with open("COST_MATRIX.pickle", 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    COST_MATRIX = pkl.loads(bytes_in)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contents = []
    content1 = ["By . Daily Mail Reporter . PUBLISHED: . 15:49 EST, 10 February 2014 . | . UPDATED: . 15:53 EST, 10 February 2014 . This is the moment rescuers raced against the clock to save a bull from mud on a flooded riverbank to stop it from being swept away by a rising tide. It took more than four hours to drag the frightened bull to safety after it became trapped on the banks of the River Lune in Overton, Lancaster, Lancashire. The terrified animal was spotted by neighbours and had sunk deep into the thick mud, unable to move anything but its head. Rescue: Morecambe RNLI crew members battle against the incoming tide to rescue the bull . Colin Midwinter, Morecambe RNLI hovercraft crew member, said: 'This operation was an excellent example in demonstrating how the combined resources and expertise of the various rescue organisations can achieve successful outcomes under challenging circumstances.' Fire crews, the RNLI, a coastguard team and other rescuers battled against the incoming tide as they finally dragged the bull free. Two sets of firefighters attended and a request was sent for Morecambe RNLIâ€™s hovercraft to attend to give safety cover along with Bay Search and Rescue. Fire crews, the RNLI, a coastguard team and other rescuers battled against the incoming tide as they finally dragged the bull free . It took more than four hours to drag the frightened bull to safety after it became trapped on the banks of the River Lune in Overton, Lancaster . Bay Search and Rescue brought in its specialist tracked Hagglund vehicle to take firefighters and equipment to where the bull was stuck. The hovercraft crew began working to free the animal using their own mud rescue equipment and lifting equipment on the Bay Search and Rescue vehicle was"]
    contents.extend(content1)
    contents.extend(content1)
    contents.extend(content1)
    contents.extend(content1)

    summaries = []
    summary1 = ['hello']
    summaries.extend(summary1)
    summary2 = ['daily mail']
    summaries.extend(summary2)
    summary3 = content1
    summaries.extend(summary3)
    summary4 = ["By . Daily Mail Reporter . PUBLISHED: . 15:49 EST, 10 February 2014 . | . UPDATED: . 15:53 EST, 10 February 2014 . This is the moment rescuers raced against the clock to save a bull from mud on a flooded riverbank to stop it from being swept away by a rising tide. It took more than four hours to drag the" ]
    summaries.extend(summary4)
    

    #model_file = os.path.join(models_folder, "bert_coverage_cnndm_bs64_0.bin")
    kw_cov = CoverageScorer(device = device)

    scores, no_summ_acc = kw_cov.score(summaries, contents)

    for body, summaries, score in zip(contents, summaries, scores):
        print("----------------")
        print("----------------")
        print("----------------")
        print(body)
        print("---")
        print(summaries)
        print("---")
        print(score)
