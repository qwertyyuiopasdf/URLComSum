from transformers.tokenization_gpt2 import GPT2Tokenizer as GPT2Tok
from transformers.tokenization_bert import BertTokenizer as BertTok
import nltk

class Capita:
    def forward(self, text):
        words = text.split(" ")
        final_words = []
        for word in words:
            if not word.isalpha():
                final_words.append(word.lower())
            else:
                if word.islower():
                    pass
                elif word.isupper():
                    final_words.append("⇧")
                elif word[0].isupper() and word[1:].islower():
                    final_words.append("↑")
                else:
                    final_words.append("↑")
                final_words.append(word.lower())
        return " ".join(final_words)

    def backward(self, text):
        words = text.split(" ")
        final_words = []
        all_caps = False; capitalized = False
        for w in words:
            if w == "⇧": all_caps = True
            elif w == "↑": capitalized = True
            else:
                final_word = w
                if all_caps: final_word = final_word.upper()
                elif capitalized:
                    if len(final_word) <= 1: final_word = final_word.upper()
                    else: final_word = final_word[0].upper()+final_word[1:]
                final_words.append(final_word)
                all_caps = False; capitalized = False
        return " ".join(final_words)

class BERTCacheTokenizer:
    def __init__(self):
        self.cache = {}
        self.cache_keys = []
        self.tokenizer = BertTok.from_pretrained("bert-base-uncased")
        # self.tokenizer.max_len = 10000 # This was removed in later transformer tokenizers

    def encode(self, text):
        if text in self.cache:
            return self.cache[text]

        output = self.tokenizer.encode(text)

        if len(self.cache) > 1000:
            del self.cache[self.cache_keys.pop(0)]
        self.cache[text] = output
        self.cache_keys.append(text)
        return output

class GPT2Tokenizer:
    def __init__(self):
        self.tokenizer = GPT2Tok.from_pretrained("gpt2")
        # self.tokenizer.max_len = 10000

        self.pad_tok, self.start_tok, self.end_tok = "<PAD>", " ST", " END"

        self.pad_id = 0
        self.start_id = self.tokenizer.encode(self.start_tok)[0]
        self.end_id =   self.tokenizer.encode(self.end_tok)[0]
        self.vocab_size =  self.tokenizer.vocab_size

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
