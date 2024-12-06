
# EyeFound

This code is based on the publicly available code from the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):

```bash
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```

This repository is based on `timm==0.3.2`, for which a fix is needed to work with PyTorch 1.8.1+.

---

## Visualization and Controlled Experiments

You can explore visualizations and perform controlled experiments using the provided Jupyter notebook:  
**`eyefound_visualization.ipynb`**

---

## Datasets for Downstream Tasks

Datasets for downstream tasks can be downloaded via the links provided in the main text. After downloading, save the CSV files to the `public_data` directory. Ensure the CSV files are properly formatted with the following two columns:
- `impath`: The absolute path to the image.
- `class`: The classification label for the image (as an integer).

---

## Fine-tuning with Pre-trained Checkpoints

### Ophthalmic Disease Classification
To fine-tune the model for ophthalmic disease classification, run:
```bash
bash scripts/opthalmic_loop.sh
```

### Systemic Disease Classification
To fine-tune the model for systemic disease classification, run:
```bash
bash scripts/chro_loop.sh
```

### Cross-validation for Diabetic Retinopathy
To perform cross-validation for diabetic retinopathy, run:
```bash
bash scripts/cross_data.sh
```

---

## Pre-training

To pre-train ViT-Large (recommended default) with multi-node distributed training, run:
```bash
python main_pretrain_dl.py
```
