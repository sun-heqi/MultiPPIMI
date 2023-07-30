# MultiPPIMI: a multimodal deep learning framework for predicting interactions between protein-protein interaction targets and modulators

<div align="left">

## Introduction
This repository contains the PyTorch implementation of the MultiPPIMI framework, a multimodal deep learning approach with a bilinear attention network that explicitly models interactions between PPI targets and modulators. The model takes modulator SMILES strings (1D) and molecular graphs (2D), as well as PPI partner protein sequences, to make predictions.

![Model Architecture of MultiPPIMI](https://github.com/sun-heqi/MultiPPIMI/blob/main/figure/framework_figure.png)



## Acknowledgements





## Environments
Install packages under conda env
```python
conda create -n MultiPPIMI python=3.7
conda activate MultiPPIMI

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pytorch-geometric -c rusty1s -c conda-forge
pip install ogb==1.3.5
pip install rdkit
```


## Data



## Training

```python
python ./train.py --fold 1 --eval_setting S1 
```

## Testing

```python
python ./test.py --fold 1 --eval_setting S1 --input_model_file ./setting_S1_fold1.model
```

## License



## Cite:
