# A Multimodal Deep Learning Framework for Predicting PPI-Modulator Interactions


## Introduction
This repository contains the PyTorch implementation of the MultiPPIMI framework, a multimodal deep learning approach with a bilinear attention network that explicitly models interactions between PPI targets and modulators. 

## Framework
![Model Architecture of MultiPPIMI](https://github.com/sun-heqi/MultiPPIMI/blob/main/figure/framework_figure.png)



## Acknowledgements
This implementation is inspired and partially based on earlier works [1], [2].




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
For the warm-start experiments with MultiPPIMI, for one fold as an example, you can directly run the following command. 
```bash
python main.py --fold 1 --eval_setting S1
```

For the cold-start experiments on one fold as an example.
```bash
python main.py --fold 1 --eval_setting S2
python main.py --fold 1 --eval_setting S3
python main.py --fold 1 --eval_setting S4
```

## Cite:

* Heqi Sun, Jianmin Wang, Hongyan Wu, Shenggeng Lin, Junwei Chen, Jinghua Wei, Shuai Lv, Yi Xiong, Dong-Qing Wei. A Multimodal Deep Learning Framework for Predicting PPI-Modulator Interactions. bioRxiv 2023.08.03.551827; doi: https://doi.org/10.1101/2023.08.03.551827 


## References

* [1] Bai, P., Miljković, F., John, B. et al. Interpretable bilinear attention network with domain adaptation improves drug–target prediction. Nat Mach Intell 5, 126–136 (2023). https://doi.org/10.1038/s42256-022-00605-1  
* [2] Liu, S., Wang, H., Liu, W., Lasenby, J., Guo, H. and Tang, J., 2021. Pre-training molecular graph representation with 3d geometry. arXiv preprint arXiv:2110.07728.   

