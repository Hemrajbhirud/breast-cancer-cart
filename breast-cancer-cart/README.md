# Breast Cancer Classification Using CART

A BTech Computer Science Machine Learning mini-project. A Tkinter desktop app that classifies breast tumour samples as **Benign** or **Malignant** using the **CART (Classification and Regression Trees)** algorithm.

> **Disclaimer:** This is an educational project. Its predictions are NOT medical diagnoses and must never be used for real medical decisions.

## 1. Introduction
Breast cancer is one of the most common cancers worldwide, and early detection greatly improves outcomes. Machine learning can learn patterns from measurements of cell nuclei taken from a fine-needle aspirate (FNA) of a breast mass. This project uses a decision tree because it is easy to interpret: the tree can be drawn and read as a set of if/else rules.

## 2. Problem Statement
Given 30 numeric measurements describing the cell nuclei in a digitised image of a breast mass, classify the tumour as:
- **Benign** – non-cancerous; does not invade nearby tissue or spread to other parts of the body.
- **Malignant** – cancerous; can invade surrounding tissue and spread (metastasise).

## 3. Objectives
- Load and explore the Breast Cancer Wisconsin dataset.
- Perform basic preprocessing and an 80:20 train/test split.
- Build a CART classifier (`DecisionTreeClassifier`, Gini criterion, limited depth).
- Evaluate with accuracy, precision, recall, F1-score and a confusion matrix.
- Visualise the class distribution and the decision tree.
- Predict on new user-entered measurements through a GUI.

## 4. Dataset
Breast Cancer Wisconsin (Diagnostic), loaded with `sklearn.datasets.load_breast_cancer()`.
- **569** records, **30** numeric features, **2** classes
- 212 malignant (target 0), 357 benign (target 1)
- Features: for each of 10 properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) the **mean**, **standard error** and **worst** value.
- No missing values.

## 5. Algorithm: CART
CART builds a **binary tree**. At each node it tests every feature and every candidate threshold and chooses the split (`feature <= threshold`) that most reduces **Gini impurity**:

```
Gini = 1 - sum(p_i^2)
```
where `p_i` is the proportion of class `i` in the node. Gini = 0 means a pure node (one class only); 0.5 is the maximum for two classes. The split minimises the weighted average Gini of the two child nodes. Splitting repeats recursively until a node is pure or a stopping rule is met.

Unrestricted trees memorise the training data (overfitting), so `max_depth=4` is used. Feature scaling is not needed since splits depend only on thresholds.

Precision, recall and F1 treat **Malignant** as the positive class, because missing a cancer (false negative) is the more serious error, so **recall** is especially important.

## 6. Technologies
Python 3, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Tkinter (ships with Python on Windows).

## 7. Project Structure
```
breast-cancer-cart/
├── main.py            # entry point
├── model.py           # data loading, training, metrics, prediction
├── gui.py             # Tkinter interface
├── visualizations.py  # Matplotlib/Seaborn figures
├── requirements.txt
└── README.md
```

## 8. Installation (Windows)
```powershell
cd breast-cancer-cart
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Tkinter is bundled with the standard python.org installer (tick "tcl/tk and IDLE" if you customise the install).

## 9. How to Run
```powershell
python main.py
```
In PyCharm / VS Code, open the `breast-cancer-cart` folder and run `main.py`.

## 10. Expected Output
- **Dashboard:** 569 records, 30 features, 2 classes, 357 benign / 212 malignant, plus sample rows.
- **Model Training:** click *Train CART Model*; status changes to "training completed" and shows about 93–96% accuracy with similar precision, recall and F1 (exact values depend on the split).
- **Visualizations:** class distribution bar chart (available at once); confusion matrix and decision tree (after training). Use the toolbar to zoom the tree.
- **Prediction:** enter 8 measurements (or use *Load typical benign / malignant*) and click *Predict* to see BENIGN or MALIGNANT with a confidence value.

## 11. Conclusion
The CART decision tree reaches high accuracy on the Wisconsin dataset while remaining fully interpretable: the drawn tree shows exactly which measurements (such as worst perimeter, worst concave points, mean texture) drive each decision. Limiting the depth keeps the gap between training and test accuracy small. The model is a learning tool, not a clinical one.
