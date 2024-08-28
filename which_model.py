import os, argparse, torch, random, sys, torchaudio
from Trainer import Trainer
from Load_model import Load_model, Load_data
from util import check_folder
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm
from models.SE_module import SE_module
import util
sys.path.append('../CMGAN')
from src.tools.compute_metrics import compute_metrics

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--data_folder', type=str, default='../noisy-vctk-16k')
    parser.add_argument('--feature', type=str, default='cross') # raw / ssl / cross
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--model', type=str, default='BLSTM')    
    parser.add_argument('--ssl_model', type=str, default='wavlm') # wav2vec2 / hubert / wavlm
    parser.add_argument('--size', type=str, default='large')  # base / large
    parser.add_argument('--finetune_SSL', type=str, default='PF') # PF / EF
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--target', type=str, default='IRM') #'MAP' or 'IRM'
    parser.add_argument('--task', type=str, default='SSL_SE') 
    parser.add_argument('--weighted_sum' , action='store_false')    
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    return args

test_set = 'NoisyAudiosTest1_0'
noisy_audio_paths = sorted(glob('/home/roger/project/EMO/dataset/MSP-PODCAST-Publish-1.11/{}/*'.format(test_set)))
noisy_audio_paths = noisy_audio_paths[2:3]
# print(noisy_audio_paths)

args = get_args()
model = SE_module(args)
model.eval()
model.cuda()

checkpoint = torch.load('./save_model/BLSTM_wavlm_IRM_epochs40_adamw_batch16_lr5e-05_cross_large_WSTrue_FTPF_finetune_MSP_noisy_norm.pth.tar')
model.load_state_dict(checkpoint['model'])

noisy_mean    = -0.00016752422864340985
noisy_std     = 0.09842836134288799

metrics_total = np.zeros(6)
num = len(noisy_audio_paths)
for path in tqdm(noisy_audio_paths):
    filename = os.path.basename(path)
    wav, sample_rate = torchaudio.load(path)
    wav = wav.cuda()
    target_wav, sample_rate = torchaudio.load(path.replace(test_set, 'BSSEAudiosTest1_0'))

    wav    = (wav - noisy_mean)/noisy_std
    enhanced_wav = model(wav, output_wav=True)
    wav = (wav*noisy_std) + noisy_mean
    enhanced_wav = (enhanced_wav*noisy_std) + noisy_mean

    difference = torch.sum(enhanced_wav.cpu()-target_wav)
    print(difference)