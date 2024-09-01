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

test_set_list = ['NoisyAudiosTest1_0', 'NoisyAudiosTest1_5', 'NoisyAudiosTest1_10', 'BSSEAudiosTest1_0', 'BSSEAudiosTest1_5', 'BSSEAudiosTest1_10', 'NoisyAudiosTest1_0_free', 'NoisyAudiosTest1_5_free', 'NoisyAudiosTest1_10_free', 'BSSEAudiosTest1_0_free', 'BSSEAudiosTest1_5_free', 'BSSEAudiosTest1_10_free']
# test_set = 'NoisyAudiosTest1_0_free'

def enhance(path):
        wav, sample_rate = torchaudio.load(path)
        clean_wav, sample_rate = torchaudio.load(path.replace(test_set, 'Audios'))

        metrics = compute_metrics(torch.squeeze(clean_wav, 0).numpy(), torch.squeeze(wav, 0).detach().cpu().numpy(), sample_rate, 0)
        metrics = np.array(metrics)

        return metrics

for test_set in test_set_list:
    print(test_set)
    noisy_audio_paths = sorted(glob('/home/roger/project/EMO/dataset/MSP-PODCAST-Publish-1.11/{}/*'.format(test_set)))

    metrics_total = Parallel(n_jobs=16, verbose=0)(delayed(enhance)(path) for path in noisy_audio_paths)
    metrics_total = np.array(metrics_total)
    metrics_avg = np.mean(metrics_total, axis=0)

    print("="*20)
    print(test_set)
    print(
        "pesq: ",
        metrics_avg[0],
        "csig: ",
        metrics_avg[1],
        "cbak: ",
        metrics_avg[2],
        "covl: ",
        metrics_avg[3],
        "ssnr: ",
        metrics_avg[4],
        "stoi: ",
        metrics_avg[5],
    )
