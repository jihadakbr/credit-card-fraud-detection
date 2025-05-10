# Credit Card Fraud Detection

This repository contains a project that implements a machine learning model to detect fraudulent credit card transactions using a dataset of anonymized transactions from European cardholders.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Project Methodology](#project-methodology)
3. [Files in this Repository](#files-in-this-repository)
   - [Credit Card Fraud Detection.ipynb](#credit-card-fraud-detectionipynb)
   - [utils.py](#utilspy)
4. [Installation](#installation)
5. [How to Use](#how-to-use)
6. [Model Performance](#model-performance)
7. [Conclusion](#conclusion)
8. [License](#license)
9. [Contact](#contact)

## Dataset Overview

The dataset used in this project is the **Credit Card Fraud Detection dataset** from Kaggle. It contains 198,608 credit card transactions, with 31 features, and the target variable is the `Class` column, where `1` represents a fraudulent transaction and `0` represents a non-fraudulent transaction.

### Dataset Columns
- `Time`: Time elapsed since the first transaction in seconds.
- `V1` to `V28`: Anonymized features representing the transaction details.
- `Amount`: The monetary amount of the transaction.
- `Class`: The target label indicating whether the transaction was fraudulent (1) or not (0).

## Project Methodology

- **Fraud Detection Model**: Developed a fraud detection model using a two-day anonymized credit card transaction dataset from September 2013.
- **Handling Class Imbalance**: Tackled the class imbalance problem by optimizing the model based on the precision-recall AUC (PR AUC) metric to improve fraud detection accuracy.
- **Feature Engineering**: Conducted hypothesis testing, correlation analysis, and feature selection, reducing the number of features from 31 to 15, which significantly enhanced the model's efficiency.
- **Model Fine-tuning**: Applied Grid Search Cross Validation for hyperparameter tuning, achieving a PR AUC of 0.812 and an F2 score of 0.794 using the Extra Trees Classifier.
- **Threshold Adjustment**: Adjusted the decision threshold to 0.570, balancing fraud detection accuracy while minimizing the impact of false negatives.

## Files in this Repository

- **`Credit Card Fraud Detection.ipynb`**: Jupyter Notebook containing the full data processing pipeline, machine learning model building, and evaluation. It includes all necessary steps to preprocess the dataset, train models, and make predictions.

- **`utils.py`**: This file contains various utility functions and libraries that support the project. It includes functions for data preprocessing, feature selection, model evaluation, and any other reusable code needed throughout the project. The goal is to keep the notebook clean and modular by importing necessary functions from this file.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

## How to Use

1. After cloning the repository and installing dependencies, open the `Credit Card Fraud Detection.ipynb` notebook in Jupyter.
   
2. Run the cells in the notebook sequentially:
    - The first few cells load and preprocess the data.
    - Feature selection and model training are carried out in subsequent cells.
    - The model is evaluated using various metrics, including PR AUC and F2 score.
    
3. The final cell in the notebook will display the performance of the model and print out the confusion matrix and classification report.
   
4. Optionally, you can adjust the decision threshold (default is 0.570) to tune the fraud detection performance based on your requirements.

5. If you'd like to use your own dataset, replace the provided dataset with your file and ensure that the data format matches the expected structure.

## Model Performance and Result

- **PR AUC**: 0.812
- **F2 Score**: 0.794
- **Decision Threshold**: 0.570

![auc score](https://raw.githubusercontent.com/jihadakbr/credit-card-fraud-detection/refs/heads/main/img/auc_score.png)

![feature importance](https://raw.githubusercontent.com/jihadakbr/credit-card-fraud-detection/refs/heads/main/img/feature_importance.png)

![pie chart](https://raw.githubusercontent.com/jihadakbr/credit-card-fraud-detection/refs/heads/main/img/pie_chart.png)

## Conclusion

The model performs well in detecting fraudulent transactions while maintaining a balance between precision and recall. Further improvements can be made by exploring additional models, feature engineering techniques, or different evaluation metrics based on the business requirements.

## License

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](LICENSE).

![CC BY-SA 4.0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-sa.svg)

# Contact
For questions or collaborations, feel free to reach out:

- Email: [jihadakbr@gmail.com](mailto:jihadakbr@gmail.com)
- LinkedIn: [linkedin.com/in/jihadakbr](https://www.linkedin.com/in/jihadakbr)
- Portfolio: [jihadakbr.github.io](https://jihadakbr.github.io/)
