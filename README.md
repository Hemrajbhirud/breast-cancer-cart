# Breast Cancer Classification Using CART Decision Tree

**Machine Learning Project – Unit 5**

A machine learning project that classifies breast tumour samples as **Benign** or **Malignant** using the **CART (Classification and Regression Trees)** algorithm.

The project uses an interpretable Decision Tree model and a Python-based Tkinter desktop application to demonstrate predictions, model training, and visualizations.

## Project Overview

The objective is to develop a classification model that identifies whether a breast tumour is benign or malignant using numerical measurements of cell nuclei.

CART uses a series of decision rules to classify samples, making its predictions easier to understand and visualize.

## Dataset

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

The dataset is also available through `sklearn.datasets.load_breast_cancer()`.

| Property | Details |
|---|---|
| Total Samples | 569 |
| Input Features | 30 |
| Classes | Benign and Malignant |
| Benign Samples | 357 |
| Malignant Samples | 212 |
| Training Samples | 455 |
| Testing Samples | 114 |
| Train-Test Split | 80:20 |

The dataset contains measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

## Algorithm Used

### CART – Classification and Regression Trees

CART is a supervised machine learning algorithm that builds a binary decision tree.

It divides data into smaller groups using feature-based conditions until it reaches a classification decision.

**Model Configuration:**

- Algorithm: CART Decision Tree
- Splitting Criterion: Gini Impurity
- Maximum Tree Depth: 4
- Task: Binary Classification
- Library: Scikit-learn

### Gini Impurity

The model uses Gini Impurity to select the most suitable split at each decision node.

**Formula:**

`Gini = 1 - Σ(pᵢ²)`

Where `pᵢ` represents the proportion of samples belonging to class `i`.

A lower Gini value indicates better separation between classes.

## Project Workflow

1. Load the Breast Cancer Wisconsin dataset.
2. Explore and preprocess the data.
3. Divide the dataset into training and testing sets.
4. Train a CART Decision Tree using Gini Impurity.
5. Limit the tree depth to reduce overfitting.
6. Evaluate the model using classification metrics.
7. Visualize the decision tree and confusion matrix.
8. Use the desktop application to demonstrate predictions.

## Model Performance

The reproduced depth-four CART experiment reported the following results:

| Evaluation Metric | Result |
|---|---|
| Accuracy | 93.86% |
| Precision (Malignant) | 90.70% |
| Recall (Malignant) | 92.86% |
| F1-Score (Malignant) | 91.76% |
| Correct Predictions | 107 / 114 |

These results are from the reproduced benchmark experiment. Results may differ when the model is retrained with different settings or data splits.

## Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Tkinter
- Visual Studio Code

## Application Features

The desktop application provides:

- **Dashboard:** Displays dataset information and class distribution.
- **Model Training:** Trains the CART classifier and displays evaluation metrics.
- **Visualizations:** Shows the decision tree, confusion matrix, and dataset charts.
- **Prediction:** Accepts user inputs and predicts whether a sample is benign or malignant.

The application simplifies input by accepting eight measurements and estimating the remaining features. Consequently, predictions through the application may differ from evaluation results obtained using all 30 measured features.

## Installation and Execution

**1. Clone the repository**

`git clone <your-repository-url>`

**2. Open the project directory**

`cd breast-cancer-cart`

**3. Install dependencies**

`pip install -r requirements.txt`

**4. Run the application**

`python main.py`

The application opens a desktop interface for training, visualizing, and testing the classification model.

## Limitations

- The dataset contains only 569 samples.
- The model has not been clinically validated.
- Incorrect predictions can occur, including missed malignant samples.
- Additional validation on independent clinical datasets is necessary.
- The application is intended for educational and experimental use only.

## Future Improvements

- Improve classification accuracy using ensemble models.
- Explore Random Forest and Gradient Boosting.
- Evaluate performance on larger datasets.
- Improve malignant-class recall.
- Introduce additional model explainability features.
- Develop a web-based interface.

## Conclusion

This project demonstrates how CART Decision Trees can classify breast tumour samples using numerical diagnostic features.

The depth-four CART benchmark achieved **93.86% accuracy**, demonstrating the usefulness of interpretable machine learning methods.

The project highlights the importance of balancing classification performance, model complexity, and interpretability.

**Disclaimer:** This application is an educational machine learning demonstration and must not be used for medical diagnosis or treatment decisions.

## Contributors

- Hemraj Bhirud
- Kunal Dayma
- Jayesh Dhupar

**Department of Computing Technologies**

SRM Institute of Science and Technology

**Course:** 21CSC305P – Machine Learning

**Academic Year:** 2026–2027
