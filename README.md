# A2-CDic
Robust Deep Convolutional Dictionary Model with Alignment Assistance for Multi-Contrast MRI Super-resolution (TMI 2025)

### Authors: Pengcheng Lei, Miaomiao Zhang, Faming Fang, and Guixu Zhang

### Environment and dataset preparation
Please refer to [MC-VarNet](https://github.com/lpcccc-cv/MC-VarNet) and [MC-DuDoN](https://github.com/lpcccc-cv/MC-DuDoN)  to prepare the dataset.
To generate spatially misaligned dataset, please run
```bash
python data/image_transformation.py
python data/image_transformation-B-spline.py
```
