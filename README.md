# Project Title: Deep Learning Solutions with TensorFlow/Keras for Real-World Challenges

## Topic: Automobile Industry & Self-Driving Cars

**Autonomous Driving & Traffic Flow Prediction**

---

## 1. Dataset and Problem Statement

**Dataset:** [Udacity Self-Driving Car Dataset (Kaggle)](https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset)

- 97,942 labels across 11 classes
- 15,000 images
- High-quality images (1920x1200) and a smaller 512x512 version for efficiency

**Problem Statement:**  
Develop an accurate and efficient object detection model for autonomous vehicles using the Udacity dataset. The model should identify and locate objects (pedestrians, cars, traffic lights, etc.) in various driving conditions.

**Key Challenges:**

- Adverse lighting (sunlight, night, overcast)
- Partial object occlusions
- Class imbalance (e.g., more cars than pedestrians)
- Real-time inference
- Diverse environments (urban, suburban, highways, intersections)
- Robust detection of small/distant objects
- Trajectory prediction (temporal modeling)

---

## 2. Dataset Selection and Justification

- **Large and diverse:** 97,942 labels, 15,000 images
- **Well-labeled:** High-quality, verified annotations
- **Varied conditions:** Lighting, weather, environments
- **Open-source:** Free to use
- **Purpose-built:** Designed for self-driving car research
- **High-resolution:** 1920x1200 and 512x512 images

---

## 3. Pipeline Overview

```mermaid
flowchart TD
    A[Download & Clean Dataset] --> B[Exploratory Data Analysis (EDA)]
    B --> C[Remove Unannotated Images]
    C --> D[Train/Validation Split]
    D --> E[Advanced Preprocessing & Augmentation]
    E --> F[Model: CNN + FPN + Transformer]
    F --> G[Training (Focal Loss, Class Weights)]
    G --> H[Evaluation (mAP, Per-Class AP, Small Object AP)]
    H --> I[Real-Time Inference Speed]
    F --> J[Temporal Modeling (LSTM/GRU Stub)]
    style J fill:#f9f,stroke:#333,stroke-width:2px
```

---

## 4. Exploratory Data Analysis (EDA)

- **Data Structure:**
  - Inspects annotation CSV for columns: filename, class, bbox coordinates, width, height
- **Class Distribution:**
  - Counts and visualizes object class frequencies
- **Image Analysis:**
  - Checks image sizes, aspect ratios, and average color values
- **Bounding Box Analysis:**
  - Examines bbox sizes, aspect ratios, and spatial distribution
- **Data Quality:**
  - Detects missing/corrupt data, images without annotations (removes them)
- **Visualization:**
  - Plots sample images with bounding boxes, histograms, and heatmaps

---

## 5. Preprocessing

- **Cleaning:**
  - Removes annotation rows for missing images
  - Filters invalid bounding boxes
  - Deletes unannotated images from the dataset directory
- **Augmentation:**
  - Horizontal flip, brightness/contrast, random crop, hue/saturation, RGB shift, motion blur, noise, CLAHE, gamma, coarse dropout, affine, sharpen, normalization (ImageNet mean/std)
- **No data leakage:**
  - All ML-specific preprocessing is fit/applied only on the training set

---

## 6. Model Architecture

- **Backbone:**
  - MobileNetV2 (pretrained on ImageNet)
- **Feature Pyramid Network (FPN):**
  - Multi-scale feature extraction for robust small object detection
- **Transformer Encoder:**
  - Self-attention for global context
- **Detection Head:**
  - TimeDistributed dense layers for class logits and bounding box regression
- **Temporal Modeling (Stub):**
  - LSTM/GRU module for future trajectory prediction (if sequential frames available)

---

## 7. Training Strategy

- **Data Preparation:**
  - Resize, normalize, augment images
- **Loss Functions:**
  - Focal loss (for class imbalance) or categorical cross-entropy
  - Smooth L1 loss for bounding box regression
- **Class Weights:**
  - Computed from class distribution to further address imbalance
- **Hyperparameter Tuning:**
  - Learning rate, batch size, optimizer, model dimensions
- **Evaluation Metrics:**
  - Mean Average Precision (mAP), per-class AP, small object AP, precision/recall, IoU

---

## 8. Evaluation & Reporting

- **Per-class mAP:**
  - Reports AP for each class
- **Small Object Detection:**
  - Reports mAP for small objects (area < 32x32)
- **Real-Time Inference:**
  - Measures and reports inference speed (FPS)
- **Visualization:**
  - Plots training/validation loss, accuracy, and sample predictions

---

## 9. Pipeline Features Addressing Project Challenges

- **Adverse Lighting:**
  - Data augmentation simulates various lighting conditions (brightness, gamma, CLAHE, color jitter)
- **Occlusion:**
  - Random cropping, dropout, and augmentation help model learn from partial objects
- **Class Imbalance:**
  - Focal loss and class weights mitigate imbalance
- **Small Objects:**
  - FPN for multi-scale features, per-class and small object mAP reporting
- **Real-Time:**
  - Inference speed (FPS) is measured
- **Diverse Environments:**
  - Augmentation and dataset variety
- **Temporal Modeling:**
  - LSTM/GRU stub for future extension to trajectory prediction

---

## 10. How to Run

1. **Install dependencies:**

   - Python 3.8+
   - TensorFlow, Keras, pandas, numpy, matplotlib, seaborn, albumentations, tqdm, PIL, kagglehub, scikit-learn

2. **Download the dataset:**

   - The script will automatically download the Udacity dataset using kagglehub.

3. **Run the pipeline:**

   - Execute `tensorflow.py` to perform EDA, preprocessing, training, and evaluation.

4. **Results:**
   - Training/validation metrics, per-class AP, small object AP, inference speed, and visualizations will be displayed.

---

## 11. Future Work

- Implement full temporal modeling for trajectory prediction using LSTM/GRU on sequential frames
- Experiment with other backbones (ResNet, EfficientNet)
- Integrate with real-time video streams for live inference
- Deploy as a REST API or edge device application

---

## 12. References

- [Udacity Self-Driving Car Dataset on Kaggle](https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset)
- [Albumentations Documentation](https://albumentations.ai/docs/)
- [TensorFlow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/)
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
- [Transformers for Vision](https://arxiv.org/abs/2010.11929)

---

## 13. Contact

For questions or collaboration, please contact the project maintainer.
