# VisionBalance-Balanced-Image-Classification-GIST-Augmentation



# VisionBalance — Balanced Image Classification (GIST + Augmentation)

**One-line:** A compact experimental pipeline that balances a small, imbalanced image dataset using augmentation, extracts raw and GIST features, and compares MLP and RNN classifiers.

---

## Table of contents

1. Overview
2. Key results (summary)
3. Dataset
4. Data preprocessing & augmentation
5. Feature extraction
6. Models and training
7. Reproducing experiments
8. Directory layout
9. Recommendations & next steps
10. Notes
11. License & contact

---

## 1. Overview

This study documents an end-to-end experimental pipeline for a small, imbalanced image classification task. The goal is to show how dataset balancing and hand-crafted features (GIST) affect model generalization compared with raw image inputs. Two model families were evaluated: a simple MLP and a simple RNN, each trained on both raw and GIST features.

## 2. Key results (summary)

* **Raw MLP:** Overall test accuracy ≈ **42%**
* **GIST MLP:** Overall test accuracy ≈ **67%**
* **Raw RNN:** Overall test accuracy ≈ **78%**
* **GIST RNN:** Overall test accuracy ≈ **88%**

> Full, per-class breakdowns and CSV outputs produced by `Evaluate_TestSet.ipynb` are included in `outputs/`.

## 3. Dataset

* **Original images:** 410
* **After augmentation (final dataset):** 2296 images
* The original dataset was highly imbalanced. The pipeline performs targeted augmentation of under-represented classes to reach a roughly balanced dataset prior to final training.

**Important:** keep raw un-augmented copies of original images in `data/original/` for reproducibility and ablation studies.

## 4. Data preprocessing & augmentation

* **Initial class balancing:** minority classes were augmented to reduce imbalance (see `data/stats_before_after.csv`).
* **Augmentation techniques used:** flips, erosion, dilation, and other simple morphological / geometric transforms.
* **Notes:** Augmentation increases sample diversity but cannot fully substitute for genuinely new images. Track per-class augmentation factors and seed augmentation operations for reproducibility.

## 5. Feature extraction

Two feature sets were used:

* **Raw features:** flattened or lightly preprocessed pixel vectors.
* **GIST features:** classical scene descriptors that summarize spatial structure and global texture information.

GIST features were observed to reduce overfitting and improve generalization in this study.

## 6. Models and training

* **Models:** Simple MLP and simple RNN variants. Implementation details (layers, activation, optimizer) are in `models/`.
* **Training split:** stratified 70% train / 30% test.
* **Loss & metrics:** cross-entropy loss; primary reported metric was accuracy (per-class and overall). We recommend also logging precision, recall, and F1.
* **Output:** `Evaluate_TestSet.ipynb` writes per-sample CSV predictions for each model to `outputs/` for downstream analysis.

## 7. Reproducing experiments

1. Set up environment: `pip install -r requirements.txt`.
2. Inspect original dataset in `data/original/` and per-class stats in `data/stats_before.csv`.
3. Run augmentation & balancing: `python scripts/augment_and_balance.py --input data/original --output data/augmented`.
4. Extract features: `python scripts/extract_gist.py --input data/augmented --out features/gist.npy` and `python scripts/extract_raw.py --input data/augmented --out features/raw.npy`.
5. Train models: `python train_mlp.py --features features/gist.npy --out models/gist_mlp.pth` (similar for RNN and raw features).
6. Evaluate: open `notebooks/Evaluate_TestSet.ipynb` and run the evaluation cells — this will generate CSV outputs in `outputs/`.

> Replace script names if your repository uses different filenames. The notebook includes helper cells that map filenames to the models used in the published results.

## 8. Directory layout (suggested)

```
/ (root)
├─ data/
│  ├─ original/         # raw images
│  ├─ augmented/        # after augmentation
│  └─ stats_before.csv
├─ features/
│  ├─ raw.npy
│  └─ gist.npy
├─ models/
├─ notebooks/
│  └─ Evaluate_TestSet.ipynb
├─ outputs/
├─ scripts/
│  ├─ augment_and_balance.py
│  ├─ extract_gist.py
│  └─ extract_raw.py
├─ requirements.txt
└─ README.md
```

## 9. Recommendations & next steps

* Try **transfer learning** with a pre-trained CNN (ResNet, EfficientNet) and fine-tune — this typically outperforms both raw-feature MLPs and classical features on small datasets.
* Use **stratified k-fold cross-validation** to reduce variance in the reported metrics.
* Log and report **macro F1 / per-class F1**, not only accuracy.
* Run an **ablation study**: measure performance impact of (a) balancing strategy, (b) each augmentation type, and (c) GIST vs raw vs fused features.
* Consider **class-weighted loss** or **focal loss** instead of brute-force oversampling when real samples are scarce.
* Add **explainability**: Grad-CAM (for CNNs), t-SNE / UMAP visualizations for feature clusters.

## 10. Notes

* Keep random seeds recorded for all steps (augmentation, split, training) to ensure reproducibility.
* When reporting results, show both training and test metrics and include confusion matrices to help identify consistent failure modes.

