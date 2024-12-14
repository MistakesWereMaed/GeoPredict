# Predicting Geolocation Through Tweet Text

This repository contains the implementation of a machine learning model for predicting geographic locations from tweet text. The project leverages BERT embeddings and multitask learning to predict geospatial coordinates with high accuracy, particularly in areas with high PFAS contamination.
**Currently, this repository only contains the code files themselves**

---

## Project Overview

This project aims to:

- Develop a machine learning pipeline that predicts geolocations from tweets.
- Enhance prediction accuracy using a multitask learning approach.
- Analyze model performance in regions with high PFAS contamination, highlighting its potential for public health applications.

Key Features:

- Multitask learning framework.
- BERT for text embeddings.
- Geospatial prediction via a weighted MAE loss.
- Robust pipeline for preprocessing text and metadata.

---

## Data

### Datasets Used:

1. **GeoText**: A dataset containing over 350,000 geotagged tweets.
2. **GeoNames US Gazetteer**: Metadata for geographic locations, used to enhance prediction accuracy.

### Data Processing Pipeline:

- **Filtering**: Non-US data removed for focus.
- **Preprocessing**:
  - Named Entity Recognition (NER) with SpaCy.
  - Metadata generation using GeoNames.
- **Tokenization**: BERT tokenizer for text embeddings.
- **Imputation**: Missing metadata imputed using FAISS.

---

## Model Architecture

### Components:

1. **Text Embeddings**:
   - Pretrained BERT model.
2. **Metadata Processing**:
   - Linear regression module.
3. **Output Layer**:
   - Weighted softmax to generate multi-point geospatial predictions.

### Loss Function:

- Weighted Mean Absolute Error (MAE) for improved prediction accuracy.

---

## Results

### Best Case:
- **Accuracy (@161 km)**: 57.43%
- **Mean Spatial Error (SAE)**: 282.37 km

### Without Metadata:
- **Accuracy (@161 km)**: 55.13%
- **Mean SAE**: 322.12 km

### High PFAS Regions:
- **Accuracy (@161 km)**: 63.02%
- **Mean SAE**: 234.10 km

---

## Future Work

1. Expand dataset to include more representative regional data.
2. Integrate probabilistic models like Gaussian Mixture Models for outputs.
3. Enhance robustness of text preprocessing to handle noisy data.

---

## Acknowledgments

This work is inspired by:
- [Yves Scherrer and Nikola Ljubešić (2020)](https://arxiv.org/abs/2303.07865)
- [Rahimi et al. (2017)](https://arxiv.org/abs/1704.04008)
- [Jacob Eisenstein et al. (2010)](https://aclanthology.org/D10-1124)

---

## Contact

For questions or collaboration opportunities, please contact [Kiryn](mistakesweremaed@gmail.com).
