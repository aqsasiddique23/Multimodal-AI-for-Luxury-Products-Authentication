# Multimodal-AI-for-Luxury-Products-Authentication
Final year project comparing image-only, text-only, and multimodal models for detecting counterfeit luxury goods (bags, watches, shoes).

**Authors:** Aqsa Siddique | Aiemah Afzal

## Overview

Does combining a product's image and description authenticate luxury goods better than either alone? Three models were fine-tuned on one shared dataset and split:

| Model | Input | Role |
|---|---|---|
| CLIP ViT-B/32 | Image + text | Primary multimodal model |
| ResNet-50 | Image only | Baseline |
| RoBERTa-base | Text only | Baseline |

## Data and Pipeline

- 2,100 listings (1,501 authentic, 599 inauthentic) across bags, watches, and shoes, split 70/10/20 into train, validation, and test
- Text cleaning to remove giveaway words, plus validation of all images
- Letterbox resize to 224×224 (keeps aspect ratio, no cropping)
- Class imbalance handled with a weighted loss and augmentation on inauthentic training images only

The dataset is not included in this repository.

## Models

- **ResNet-50:** ImageNet-pretrained, fine-tuned with differential learning rates
- **RoBERTa-base:** `[CLS]` representation, then a linear classifier
- **CLIP:** image and text embeddings (512-d each) concatenated into a 1024-d vector, then a classifier head, with only the last 2 CLIP blocks unfrozen

All models use early stopping and are evaluated on the held-out test set with the best checkpoint.

## Results (test set, n = 420)

| Model | Accuracy | Counterfeits caught | Counterfeits missed |
|---|---|---|---|
| ResNet-50 | 91.4% | 70.8% | 35 |
| RoBERTa | 99.5% | 98.3% | 2 |
| CLIP | 94.0% | 82.5% | 21 |

CLIP beat the image-only baseline. Text-only RoBERTa scored highest, which suggests the listing text carries strong cues on its own. Results come from a small dataset (120 counterfeits in the test set).

## Running

Written for Google Colab (T4 GPU). Run the notebook cells in order and set the dataset paths in the Drive cell.

```bash
pip install open-clip-torch openpyxl scikit-learn tqdm transformers
```
