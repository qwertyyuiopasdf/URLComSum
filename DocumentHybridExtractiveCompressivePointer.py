import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch
from nltk import word_tokenize
from typing import Tuple
import numpy as np
from DocumentExtractor import DocumentExtractor 
from DocumentCompressivePointer import DocumentCompressivePointer


class DocumentHybridExtractiveCompressivePointer(nn.Module):

    def __init__(self, tkner, embedding_dim, hidden_dim, num_attention_head=5, dropout_rate=0.5):
        super(DocumentHybridExtractiveCompressivePointer, self).__init__()
        self.extractor = DocumentExtractor(tkner, embedding_dim, hidden_dim, num_attention_head, dropout_rate)
        self.compressor = DocumentCompressivePointer( tkner, embedding_dim, hidden_dim, num_attention_head, dropout_rate)

    def forward(self, document, extractor_summary_size, compressor_summary_size):
        
        extractor_outputs, extractor_pointers, extractor_hidden = self.extractor(document, extractor_summary_size)
        compressor_outputs, compressor_pointers, compressor_hidden = self.compressor(extractor_outputs, compressor_summary_size)
        
        return extractor_outputs, extractor_pointers, extractor_hidden, compressor_outputs, compressor_pointers, compressor_hidden
