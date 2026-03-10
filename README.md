# AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation

<p align="center">
  <b>Official implementation of AMLRIS for Referring Image Segmentation</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.11-red">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## Introduction

**AMLRIS** is a referring image segmentation framework based on **alignment-aware masked learning**, designed to improve cross-modal alignment between visual and linguistic representations.

This repository provides:

- Training and evaluation code for AMLRIS
- Support for **RefCOCO**, **RefCOCO+**, **RefCOCOg**, and **RefCOCOm**
- Pretrained backbone usage with **Swin-B** and **BERT-B**
- Single-GPU evaluation and multi-GPU training support

---

## Requirements

- Python 3.8
- PyTorch 1.11
- Other dependencies listed in `requirements.txt`

Requirements can also refer to the CARIS repository: https://github.com/lsa1997/CARIS

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

### 1. Download RefCOCO Annotations

Please download the annotation files for the following datasets:

- RefCOCO
- RefCOCO+
- RefCOCOg
- RefCOCOm

After downloading, organize them as follows:

```text
{YOUR_REFER_PATH}
├── refcoco
├── refcoco+
├── refcocog
└── refcocom
```

### 2. Download COCO Images

Download **COCO 2014 Train images** (`train2014`, about 83K images / 13GB).

After extracting `train2014.zip`, organize the images as follows:

```text
{YOUR_COCO_PATH}
└── train2014
```

---

## Pretrained Models

### Backbone Models

#### Swin-B

Download the pretrained Swin-B checkpoint from the official Swin Transformer repository:

```bash
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
```

#### BERT-B

Download the pretrained BERT-Base-Uncased model from HuggingFace.

Option 1: using `git`

```bash
git lfs install
git clone https://huggingface.co/bert-base-uncased
```

Option 2: using `huggingface_hub`

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='bert-base-uncased', local_dir='./bert-base-uncased')"
```

### Model Zoo

| Model  | Backbone        | Download |
|--------|------------------|----------|
| Swin-B | -                | `swin_base_patch4_window7_224_22k.pth` |
| BERT-B | -                | `bert-base-uncased` |
| AMLRIS | Swin-B + BERT-B  | `models` |

---

## Project Structure

```text
.
├── scripts/
│   ├── train_refcoco.sh      # Training script
│   └── test_refcoco.sh       # Evaluation script
├── models/                   # Model definitions
├── datasets/                 # Dataset loaders
├── utils/                    # Utility functions
├── requirements.txt          # Dependencies
└── README.md
```

---

## Training

By default, **fp16 training** is enabled for better memory efficiency.

### Train on RefCOCO

Before training, modify the following variables in `scripts/train_refcoco.sh`:

```bash
YOUR_COCO_PATH    # Path to COCO dataset
YOUR_REFER_PATH   # Path to RefCOCO annotations
YOUR_MODEL_PATH   # Path to pretrained models
YOUR_CODE_PATH    # Path to this codebase
```

Then run:

```bash
sh scripts/train_refcoco.sh
```

### Train on Other Datasets

You can change the `DATASET` variable in the script:

```bash
DATASET=refcoco    # Default
DATASET=refcoco+   # RefCOCO+
DATASET=refcocog   # RefCOCOg
DATASET=refcocom   # RefCOCOm
```

### RefCOCOg Split Setting

RefCOCOg provides two different splits: `umd` and `google`.

Please specify the split using `--splitBy`:

```bash
# Train on RefCOCOg with umd split
bash scripts/train_refcoco.sh --splitBy umd

# Train on RefCOCOg with google split
bash scripts/train_refcoco.sh --splitBy google
```

---

## Evaluation

Single-GPU evaluation is supported.

### Evaluate on RefCOCO

First modify the settings in `scripts/test_refcoco.sh`, then run:

```bash
bash scripts/test_refcoco.sh
```

### Evaluate on Other Datasets / Splits

You can modify `DATASET` and `SPLIT` in `scripts/test_refcoco.sh`:

```bash
# RefCOCO+
DATASET=refcoco+
SPLIT=val    # or testA / testB

# RefCOCOg
DATASET=refcocog
# Remember to add --splitBy umd or --splitBy google

# RefCOCOm trained model
# Can be directly evaluated on refcoco / refcoco+ / refcocog (umd)
DATASET=refcoco
```

### Notes for Evaluation

- For **RefCOCOg**, remember to specify `--splitBy umd` or `--splitBy google`
- Models trained on **RefCOCOm** can be directly evaluated on:
  - RefCOCO
  - RefCOCO+
  - RefCOCOg (umd)

No additional fine-tuning is required.


## Quick Start

```bash
# 1. Clone the repository
cd AMLRIS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download COCO images
# Follow the Dataset Preparation section above

# 4. Download RefCOCO annotations
# Follow the Dataset Preparation section above

# 5. Download pretrained Swin-B
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth

# 6. Download pretrained BERT-B
git lfs install
git clone https://huggingface.co/bert-base-uncased

# 7. Download AMLRIS pretrained models
# Please refer to the Model Zoo section

# 8. Train on RefCOCO
bash scripts/train_refcoco.sh

# 9. Evaluate on RefCOCO
bash scripts/test_refcoco.sh
```

---

## Notes

- fp16 training is enabled by default for memory efficiency
- Single-GPU evaluation is supported
- RefCOCOg requires specifying the split with `--splitBy umd` or `--splitBy google`
- Models trained on RefCOCOm can be transferred directly to RefCOCO / RefCOCO+ / RefCOCOg (umd) evaluation without extra fine-tuning

---

## References

This repo is mainly built based on **CARIS** and **DETRIS**. Thanks for their great work!


## Citation

If you find our code useful, please consider citing:

```bibtex
@inproceedings{chen2026amlris,
  title={AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation},
  author={Chen, T. and Yang, S. and Yang, Y. and others},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
