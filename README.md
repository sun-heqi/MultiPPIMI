# MultiPPIMI: a multimodal deep learning framework for predicting interactions between protein-protein interaction targets and modulators


## Introduction
This repository contains the PyTorch implementation of the MultiPPIMI framework, a multimodal deep learning approach with a bilinear attention network that explicitly models interactions between PPI targets and modulators. The model takes modulator SMILES strings (1D) and molecular graphs (2D), as well as PPI partner protein sequences, to make predictions.

## Framework
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
The `data` folder contains all experimental data used in MultiPPIMI. In `data/folds`, we have full data split under four settings (S1: random split, S2: modulator cold-start, S3: PPI cold-start, S4: cold pair). 


## Reproduce results
For the warm-start experiments with vanilla DrugBAN, you can directly run the following command. 
```bash
for fold in 1..5; do
  python main.py --fold $fold --eval_setting S1
done
```



## License



## Cite:
