# M2TR: Multi-modal Multi-scale Transformers for DeepfakeDetection

## Introduction

This is the official pytorch implementation of [Multi-modal Multi-scale for Deepfake detection](https://arxiv.org/abs/2104.09770), which is accepted by ICMR 2022.


<p align="center">
 <img width="75%" src="./imgs/network.png" />
</p>


## Model Zoo

The baseline models on three versions of [FF-DF](https://github.com/ondyari/FaceForensics) dataset and [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) are provided.

| Dataset | Accuracy | model |
| --- | --- | --- | 
| FF++ (Raw) | 99.50 | [FF-RAW](https://drive.google.com/file/d/1_HaPE6r7Zzof2mmLmmc4fbIbqyWs17S0/view?usp=sharing) |
| FF++ (C23) | 97.93 | [FF-C23](https://drive.google.com/file/d/1XRIllA6p5YnITztl1burwcr5l7LAcpqv/view?usp=sharing) 
| FF++ (C40) | 92.89 | [FF-C40](https://drive.google.com/file/d/1xhclIjoh8GkVvoVefjDY-itdaV0VaMxY/view?usp=sharing) |
| CelebDF |99.76 |[CelebDF](https://drive.google.com/file/d/1_HaPE6r7Zzof2mmLmmc4fbIbqyWs17S0/view?usp=sharing) |

## Training and Evaluation

```
python run.py --cfg ./configs/m2tr.yaml
```

## License

This project is released under the MIT license.


## Citations

```bibtex
@article{wang2021m2tr,
  inproceedings={M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection},
  author={Wang, Junke and Wu, Zuxuan and Chen, Jingjing and Jiang, Yu-Gang},
  booktitle={ICMR},
  year={2022}
}
```
