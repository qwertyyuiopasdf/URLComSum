from torch.utils.data import DataLoader, RandomSampler
import torch, os, sys, time, argparse, pickle5 as pkl, numpy as np
from torch.optim import AdamW
from model_generator import GeneTransformer
from model_coverage_ot import CoverageScorer
from datasets import load_dataset
from datetime import datetime, timedelta
from utils_logplot import LogPlot
import torch.nn as nn
import threading, queue
import gc
import torch.nn.functional as F
from DocumentHybridExtractiveCompressivePointer import DocumentHybridExtractiveCompressivePointer

torch.manual_seed(12345)

gc.collect()
user = os.getlogin()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")

parser.add_argument("--root_folder", type=str, default="/home/"+user+"/")
parser.add_argument("--train_batch_size", type=int, default=5, help="Training batch size.")
parser.add_argument("--embedding_dim", type=int, default=300, help="Embedding size")
parser.add_argument("--hidden_dim", type=int, default=int(300/2), help="Hidden size")
parser.add_argument("--lstm_layers", type=int, default=3, help="LSTM layers")
parser.add_argument("--dropout", type=int, default=0.1, help="Dropout Rate")
parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs to run over the data.")
parser.add_argument("--optim_every", type=int, default=4, help="Optimize every x backprops. A multiplier to the true batch size.")
parser.add_argument("--max_ext_output_length", type=int, default=46, help="Maximum output length. Saves time if the sequences are short.")
parser.add_argument("--max_comp_output_length", type=int, default=46, help="Maximum output length. Saves time if the sequences are short.")
parser.add_argument("--fluency_ratio", type=float, default=1.0, help="RL score ratio for fluency")
parser.add_argument("--coverage_ratio", type=float, default=1.0, help="RL score ratio for coverage")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--save_every", type=int, default=60, help="Number of seconds between any two saves.")
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
parser.add_argument("--ckpt_every", type=int, default=200, help="If 0, checkpointing is not used. Otherwise, checkpointing is done very x seconds.")
parser.add_argument("--ckpt_lookback", type=int, default=100, help="When checkpointing, will consider the avg total score of the last x samples.")
parser.add_argument("--dataset_str", type=str, default="cnn_dailymail") ## cnn_dailymail, newsroom, gigaword, xsum
parser.add_argument("--dataset_doc_field", type=str, default="article") ##cnn_dailymail = article, newsroom=text, gigaword=document, xsum = document
parser.add_argument("--resume_training", type=bool, default=False) 
parser.add_argument("--tkner", type=str, default="w2v") 

args = parser.parse_args()

models_folder = "/home/models/"
log_folder = "/home/logs/"

summarizer = DocumentHybridExtractiveCompressivePointer(args.tkner, args.embedding_dim, args.hidden_dim)

print("summarizer", summarizer)
summarizer.cuda()

ckpt_every = args.ckpt_every
ckpt_lookback = int((args.ckpt_lookback+args.train_batch_size-1)/args.train_batch_size)
total_score_history = []
best_ckpt_score = None

ckpt_file = os.path.join(models_folder, "summarizer_"+args.experiment+"_ckpt.bin")
ckpt_optimizer_file = os.path.join(models_folder, "summarizer_optimizer_"+args.experiment+"_ckpt.bin")


n_epochs = args.n_epochs

print("---------------")

print("Summarizer loaded")

def collate_func(inps):
    return [inp[0].decode() for inp in inps]


no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

logplot_file = os.path.join(log_folder, "URLComSum_%s.log" % (args.experiment))
logplot = LogPlot(logplot_file)

optimizer = AdamW(summarizer.parameters(), lr=args.lr)
time_save = time.time()
time_ckpt = time.time()


print("Loading scorers")

scorers = [{"name": "coverage", "importance": args.coverage_ratio, "sign": 1.0, "model": CoverageScorer(tkner=args.tkner, device=args.device)},

           ]

if args.dataset_str == "cnn_dailymail":
    fluency_news_model_file = os.path.join(models_folder, "gpt2_fluency_cnn.bin")
    with open('word_count_cnn_dailymail.pickle', 'rb') as handle:
        WORD_COUNT = pkl.load(handle)
elif args.dataset_str == "newsroom":
    fluency_news_model_file = os.path.join(models_folder, "gpt2_fluency_newsroom.bin")
    with open('word_count_newsroom.pickle', 'rb') as handle:
        WORD_COUNT = pkl.load(handle)
elif args.dataset_str == "wikihow":
    fluency_news_model_file = os.path.join(models_folder, "gpt2_fluency_wikihow.bin")
    with open('word_count_wikihow.pickle', 'rb') as handle:
        WORD_COUNT = pkl.load(handle)
else:
    fluency_news_model_file = os.path.join(models_folder, "gpt2_fluency_xsum.bin")
    with open('word_count_xsum.pickle', 'rb') as handle:
        WORD_COUNT = pkl.load(handle)

scorers.append({"name": "fluency", "importance": args.fluency_ratio, "sign": 1.0, "model": GeneTransformer(max_output_length=args.max_comp_output_length, device=args.device, starter_model=fluency_news_model_file, word_count=WORD_COUNT)})


if args.resume_training:
    summarizer.load_state_dict(torch.load(ckpt_file))
    optimizer.load_state_dict(torch.load(ckpt_optimizer_file))
                    

my_queue = queue.Queue()
print("Started training")


def collate_func(inps):
    return [a for a in inps]


if args.dataset_str == "cnn_dailymail":
    dataset = load_dataset(args.dataset_str, '3.0.0', split='train')
elif args.dataset_str == "newsroom":
    dataset = load_dataset(args.dataset_str, data_dir="/home/newsroom_complete", split='train')
else:
    dataset = load_dataset(args.dataset_str, split='train')

print("Dataset size:", len(dataset))
dataloader = DataLoader(dataset, batch_size=args.train_batch_size, sampler=RandomSampler(dataset), drop_last=True, collate_fn=collate_func)

for epi in range(n_epochs):
    print("=================== EPOCH",epi, "===================")
    for ib, documents in enumerate(dataloader):
        gc.collect()
        torch.cuda.empty_cache()
        Timer = {}
        
        T1 = time.time()
        log_obj = {}


        bodies = [doc[args.dataset_doc_field] for doc in documents]

 
        T2 = time.time()
        Timer["preprocessing_starting"] = T2-T1

        # T1b = time.time()
        ext_sampled_summaries, ext_sampled_outputs, ext_sampled_idx_batch, comp_sampled_summaries, comp_sampled_outputs, comp_sampled_idx_batch = summarizer.forward(bodies, args.max_ext_output_length, args.max_comp_output_length)

        T3 = time.time()
        Timer["generator_sampled"] = T3-T2
        with torch.no_grad():
            ext_argmax_summaries, ext_argmax_outputs, ext_argmax_idx_batch, comp_argmax_summaries, comp_argmax_outputs, comp_argmax_idx_batch = summarizer.forward(bodies, args.max_ext_output_length, args.max_comp_output_length)

        T4 = time.time()
        Timer["generator_argmax"] = T4-T3

        ext_selected_logprobs = torch.sum(ext_sampled_outputs, dim=1)
        comp_selected_logprobs = torch.sum(comp_sampled_outputs, dim=1)

        batch_size, _ = comp_sampled_outputs.shape

        scores_track = {}
        total_sampled_scores = torch.FloatTensor([0.0] * batch_size).to(args.device)
        total_argmax_scores = torch.FloatTensor([0.0] * batch_size).to(args.device)
        for scorer in scorers:
            T = time.time()
            sampled_scores, extra = scorer['model'].score(comp_sampled_summaries, bodies, extra=None)
            
            sampled_scores = torch.FloatTensor(sampled_scores).to(args.device)
            argmax_scores, _ = scorer['model'].score(comp_argmax_summaries, bodies, extra=extra)
            argmax_scores  = torch.FloatTensor(argmax_scores).to(args.device)
            Timer["scores_"+scorer['name']] = time.time()-T
            total_sampled_scores += (scorer['sign'])*(scorer['importance'])*sampled_scores
            total_argmax_scores  += (scorer['sign'])*(scorer['importance'])*argmax_scores
            log_obj[scorer['name']+"_score"] = sampled_scores.mean().item()
            scores_track[scorer['name']+"_scores"] = sampled_scores

        T5 = time.time()
        Timer['all_scores'] = T5-T4

        ext_Loss = torch.mean((total_argmax_scores - total_sampled_scores) * ext_selected_logprobs)
        comp_Loss = torch.mean((total_argmax_scores - total_sampled_scores) * comp_selected_logprobs)


        ext_Loss.backward()
        comp_Loss.backward()

        T6 = time.time()
        Timer['backward'] = T6-T5

        if ib%args.optim_every == 0:
            optimizer.step()
            optimizer.zero_grad()

        T7 = time.time()
        Timer['optim'] = T7-T6

        avg_total = total_sampled_scores.mean().item()

        total_score_history.append(avg_total)
        #log_obj['summary_nwords'] = int(np.mean(sampled_end_idxs))
        log_obj['ext_Loss'] = ext_Loss.item()
        log_obj['comp_Loss'] = comp_Loss.item()
        log_obj['total_score'] = avg_total
        log_obj['count'] = batch_size
        logplot.cache(log_obj, prefix="T_")

        Tfinal = time.time()
        Timer['total'] = Tfinal - T1

        if (time.time()-time_save > args.save_every):
            print("==========================================")
            print(bodies[0])
            print("-----------")
            print(ext_sampled_summaries[0])
            print("-----------")
            print(comp_sampled_summaries[0])
            print("-----------")
            print("Total score:", total_sampled_scores[0].item())
            for scorer in scorers:
                print(scorer['name']+" score:", scores_track[scorer['name']+"_scores"][0].item())
            print("-----------")

            logplot.save(printing=True)
            # print(Timer)

            time_save = time.time()
            print("==========================================")

        if ckpt_every > 0 and len(total_score_history) > ckpt_lookback:
            current_score = np.mean(total_score_history[-ckpt_lookback:])
            
            if time.time()-time_ckpt > ckpt_every:
                revert_ckpt = best_ckpt_score is not None and current_score < min(1.2*best_ckpt_score, 0.8*best_ckpt_score) # Could be negative or positive
                print("================================== CKPT TIME, "+str(datetime.now())+" =================================")
                print("Previous best:", best_ckpt_score)
                print("Current Score:", current_score)
                print("[CKPT] Am I reverting?", ("yes" if revert_ckpt else "no! BEST CKPT"))
                if revert_ckpt:
                    summarizer.load_state_dict(torch.load(ckpt_file))
                    optimizer.load_state_dict(torch.load(ckpt_optimizer_file))
                time_ckpt = time.time()
                print("==============================================================================")

            if best_ckpt_score is None or current_score > best_ckpt_score:
                print("[CKPT] Saved new best at: %.3f %s" % (current_score, "["+str(datetime.now())+"]"))
                best_ckpt_score = current_score
                torch.save(summarizer.state_dict(), ckpt_file)
                torch.save(optimizer.state_dict(), ckpt_optimizer_file)
