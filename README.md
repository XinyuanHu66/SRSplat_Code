
<p align="center">
  <h1 align="center">SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images</h1>
  <h3 align="center"><a href="https://arxiv.org/abs/2511.12040">Paper</a> | <a href="https://srsplat.github.io">Project Page</a> </h3>


## News
<ul>
<li><b>11/03/26 Update:</b> Check out Feng's <a href="https://xiangfeng66.github.io/SR3R/">SR3R [CVPR '26]</a>, which predicts HR 3DGS representations from sparse LR views via a learned mapping network. </li>
</ul>


## Installation

Clone this project and prepare the environment:

```bash
conda create -n srsplat python=3.10
conda activate srsplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_w_version.txt
```

## Datasets

We use the same training datasets as pixelSplat and MVSplat. Below we quote pixelSplat's [detailed instructions](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) on getting datasets.

> pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the Real Estate 10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

> If you would like to convert downloaded versions of the Real Estate 10k and ACID datasets to our format, you can use the [scripts here](https://github.com/dcharatan/real_estate_10k_tools). Reach out to us (pixelSplat) if you want the full versions of our processed datasets, which are about 500 GB and 160 GB for Real Estate 10k and ACID respectively.


## TODO list
- [x] Release basic code of the TADC.
- [ ] Release the code for reference gallery generation.
- [ ] Release all checkpoints and more useful scripts.

## BibTeX
```bash
@article{hu2025srsplat,
  title={SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images},
  author={Hu, Xinyuan and Shi, Changyue and Yang, Chuxiao and Chen, Minghao and Ding, Jiajun and Wei, Tao and Wei, Chen and Yu, Zhou and Tan, Min},
  journal={arXiv preprint arXiv:2511.12040},
  year={2025}
}
```
