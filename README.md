# BSSE-SE
This is the official implementation of our paper *"[Boosting Self-Supervised Embeddings for Speech Enhancement](https://arxiv.org/abs/2204.03339)"*

## Requirements
- pytorch 1.10.2
- torchaudio 0.10.2
- pesq 0.0.3
- pystoi 0.3.3
- numpy 1.20.3
- tensorboardx 2.2
- tqdm 4.60.0
- scikit-learn 0.24.1
- pandas 1.2.4
- fairseq 0.11.0+f97cdf7

You can use pip to install Python depedencies.

```
pip install -r requirements.txt
```

## Data preparation

### Voice Bank--Demand Dataset
The Voice Bank--Demand Dataset is not provided by this repository. Please download the dataset and build your own PyTorch dataloader from [here](https://datashare.is.ed.ac.uk/handle/10283/1942?show=full).
For each `.wav` file, you need to first convert it into 16kHz format by any audio converter (e.g., [sox](http://sox.sourceforge.net/)).
```
sox <48K.wav> -r 16000 -c 1 -b 16 <16k.wav>
```

### Pretrained enhancement model weight
Please download the model weights from [here](https://drive.google.com/file/d/1s2EzhwCEvfJ-4COIz4LcdVXI8WexsJZE/view?usp=sharing), and make a folder named `save_model` then put the weight file under the folder. 

### Result on Voice Bank--Demand
Experiment Date | PESQ | CSIG | CBAK | COVL
-|-|-|-|-
2022-04-30 | 3.20 | 4.52 | 3.58 | 3.88

## Usage

## Pre-Trained Models
Please download the pretrained model first if you want to used ssl feature and put the weight under the `save_model` folder (e.g, `save_model/WavLM-Base+.pt`). The pretrain model can be downloaded by below link.
<br> WavLM
Model | Pre-training Dataset | Fine-tuning Dataset | Model
|---|---|---|---
WavLM Base |  [960 hrs LibriSpeech](http://www.openslr.org/12)| -  | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/WavLM-Base.pt?sv=2020-04-08&st=2021-11-05T00%3A35%3A31Z&se=2022-11-06T00%3A35%3A00Z&sr=b&sp=r&sig=JljnRVzyHY6AjHzhVmHV5KyQQCvvGfgp9D2M02oGJBU%3D) <br> [Google Drive](https://drive.google.com/file/d/19-C7SMQvEFAYLG5uc47NX_MY03JCbI4x/view?usp=sharing)
WavLM Base+ | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main)| -  |  [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-04-08&st=2021-11-05T00%3A34%3A47Z&se=2022-10-06T00%3A34%3A00Z&sr=b&sp=r&sig=Gkf1IByHaIn1t%2FVEd9D6WHjZ3zu%2Fk5eSdoj21UytKro%3D) <br> [Google Drive](https://drive.google.com/file/d/1PlbT_9_B4F9BsD_ija84sUTVw7almNX8/view?usp=sharing) 
WavLM Large | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main)| -  | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/WavLM-Large.pt?sv=2020-08-04&st=2021-11-22T10%3A03%3A53Z&se=2022-11-23T10%3A03%3A00Z&sr=b&sp=r&sig=3kB8dwTCyIS8YQ7gW5oXmDrXV%2FAaLmoxBS37oPpFsz4%3D) <br> [Google Drive](https://drive.google.com/file/d/1rMu6PQ9vz3qPz4oIm72JDuIr5AHIbCOb/view?usp=sharing) 




Please download the pretrained [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) first and put the weight under the `save_model` folder (e.g, `save_model/WavLM-Base+.pt`). Wav2vec and Hubert model can be downloaded from [here]() and [here](). Run the following command to train the speech enhancement model:
```
python main.py \
    --data_folder <root/dir/of/dataset> 
    --model BLSTM 
    --ssl_model <wavlm/hubert/wav2vec2>
    --feature <raw/ssl/cross> 
    --size <base/large> 
    --target IRM 
    --finetune_SSL <PF/EF/None> 
    --weighted_sum
```

add `--mode test` in the command line and the rest remain the same to evaluate the speech enhancement model:
```
python main.py --mode test ... 
```


## Citation
Please cite the following paper if you find the codes useful in your research.

```
@article{hung2022boosting,
  title={Boosting Self-Supervised Embeddings for Speech Enhancement},
  author={Hung, Kuo-Hsuan and Fu, Szu-wei and Tseng, Huan-Hsin and Chiang, Hsin-Tien and Tsao, Yu and Lin, Chii-Wann},
  journal={arXiv preprint arXiv:2204.03339},
  year={2022}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
