from transformers.modeling_gpt2 import GPT2LMHeadModel, GPT2Config

import torch.utils.data.dataset
import utils_tokenizer
import torch, tqdm, math

def pad(data, padval=0):
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=padval)

class GeneTransformer:
    def __init__(self, max_output_length=25, max_input_length=300, device='cpu', tokenizer_type='gpt2', bpe_model="", starter_model=None, word_count=None):
        if tokenizer_type == "gpt2":
            self.tokenizer = utils_tokenizer.GPT2Tokenizer()
            config = GPT2Config.from_pretrained("gpt2")

        elif tokenizer_type == "bpecap":
            self.tokenizer = utils_tokenizer.BPETokenizer(bpe_model)
            config = GPT2Config.from_dict({"finetuning_task": None, "initializer_range": 0.02,
                            "layer_norm_epsilon": 1e-05, "n_ctx": 1024, "n_embd": 768, "n_head": 12, "n_layer": 12, "n_positions": 1024, "num_labels": 1,
                            "resid_pdrop": 0.1, "use_bfloat16": False, "vocab_size": self.tokenizer.vocab_size})
        else:
            print("Tokenizer unrecognized. Should be gpt2 or bpecap.")
            exit()

        self.model = GPT2LMHeadModel(config)

        self.model.to(device)
        self.device = device
        if starter_model is not None:
            self.reload(starter_model)

        self.max_output_length = max_output_length
        self.max_input_length = max_input_length

        self.model.train()
        self.mode = "train"
        if word_count is not None:
            self.word_count = word_count

    def train_batch(self, bodies, summaries, special_append=None, no_preinput=False):
        inputs, summ_inp, summ_out = self.preprocess_batch(bodies, summaries, special_append)
        past = None
        if not no_preinput:
            _, past = self.model(input_ids=inputs, past=None)
        #logits, _ = self.model(input_ids=summ_inp, past_key_values=past)
        logits, _ = self.model(input_ids=summ_inp, past=past)
        crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss = crit(logits.view(-1, self.tokenizer.vocab_size), summ_out.contiguous().view(-1))
        return loss

    def train(self):
        self.model.train()
        self.mode = 'train'

    def eval(self):
        self.model.eval()
        self.mode = 'eval'
        
    def reload(self, from_file):
        print(self.model.load_state_dict(torch.load(from_file)))

    def save(self, to_file):
        torch.save(self.model.state_dict(), to_file)

    def preprocess_input(self, bodies, special_append=None):
        if special_append is None:
            special_append = [[] for i in range(len(bodies))]
        inputs = [torch.LongTensor(spe+self.tokenizer.encode(body)) for body, spe in zip(bodies, special_append)]
        inputs = pad(inputs, padval=0)
        inputs = inputs[:, :self.max_input_length].to(self.device)
        return inputs

    def preprocess_batch(self, bodies, summaries, special_append=None):
        inputs = self.preprocess_input(bodies, special_append)

        # Big hack
        if special_append is None:
            special_append = [[] for i in range(len(bodies))]

        summaries = [spe+self.tokenizer.encode(summ) for summ, spe in zip(summaries, special_append)]

        summaries = [summ[:(self.max_output_length-1)] for summ in summaries] # We cut short, but we want the end token at the end

        summ_inp = pad([torch.LongTensor([self.tokenizer.start_id]+summ) for summ in summaries], padval=0).to(self.device)
        summ_out = pad([torch.LongTensor(summ+[self.tokenizer.end_id]) for summ in summaries], padval=-1).to(self.device)
        return inputs, summ_inp, summ_out

    def score(self, summaries, bodies, idx_batch=None, bodies_tokenized=None, lengths=None, extra=None):
        # Unconditional rating of the summaries
        self.model.eval()

        inputs, summ_inp, summ_out = self.preprocess_batch(bodies, summaries)
        summ_out = summ_out.contiguous()

        with torch.no_grad():
            logits, _ = self.model(input_ids=summ_inp[:1024], past=None)

            crit = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = crit(logits.view(-1, self.tokenizer.vocab_size), summ_out.view(-1)).view(summ_out.shape)
            mask = (summ_inp != torch.LongTensor([0]).to(self.device)).float()
            non_pad_count = torch.sum(mask, dim=1)

            p_us = []
            total_word = sum(self.word_count.values())

            for idx, summary in enumerate(summaries):
                tokens = self.tokenizer.encode(' '.join(summary))
                
                p_u = 1
                
                for token in tokens:
                    try:
                        p_u *= self.word_count[token]/total_word
                    except:
                        p_u *= 1 /total_word#in case the word is not found in the training dataset

                p_us.append(math.log(p_u+0.001))

        
        p_us = torch.tensor(p_us).to(self.device)
        loss_per = (torch.sum(loss, dim=1) - p_us)/ non_pad_count 

        score = (10.0 - loss_per) / 10.0
        return score.tolist(), None

    def score_pairs(self, bodies, summaries):
        if self.mode != 'eval':
            print("BEWARE. Model is not in eval mode.")

        inputs, summ_inp, summ_out = self.preprocess_batch(bodies, summaries)

        with torch.no_grad():
            _, past = self.model(input_ids=inputs, past=None)
            logits, _ = self.model(input_ids=summ_inp, past=past)

            crit = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = crit(logits.view(-1, self.tokenizer.vocab_size), summ_out.view(-1)).view(summ_out.shape)
            mask = (summ_inp != torch.LongTensor([0]).to(self.device)).float()
            non_pad_count = torch.sum(mask, dim=1)
            loss_per = torch.sum(loss, dim=1) / non_pad_count

        return loss_per.tolist()



