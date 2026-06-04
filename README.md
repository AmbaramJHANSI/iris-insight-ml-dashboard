# 🌸 IrisInsight: ML-Powered Species Classification Dashboard

IrisInsight is an interactive Machine Learning dashboard that predicts Iris flower species using trained classification models and provides real-time analytics, confidence scoring, and data visualizations through Streamlit and Plotly.

The project demonstrates the complete Machine Learning workflow, including data preprocessing, model training, model evaluation, model selection, deployment-ready model persistence, and interactive dashboard development.

---

## 📌 Project Overview

The Iris dataset is one of the most widely used datasets in Machine Learning and pattern recognition.

Using four flower measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

the system predicts the species of an Iris flower among:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

In addition to prediction, the application provides detailed analytics, probability distributions, dataset insights, and feature relationship visualizations.

---

## 🚀 Features

### Machine Learning

* Iris species prediction using trained classification models
* Multi-model comparison
* Logistic Regression implementation
* Decision Tree implementation
* Random Forest implementation
* Automatic best-model selection
* Model persistence using Joblib

### Interactive Dashboard

* Professional Streamlit dashboard
* Responsive user interface
* Interactive slider controls
* Real-time prediction updates
* Confidence score analysis
* Probability distribution visualization

### Data Analytics

* Species distribution analysis
* Correlation heatmap
* Feature relationship analysis
* Dataset preview
* Interactive Plotly visualizations

### Software Engineering

* Git version control
* GitHub repository integration
* Modular project structure
* Deployment-ready architecture


## 🚀 Live Demo

[Open IrisInsight](https://iris-insight-ml-dashboard.streamlit.app/)

---

## 🧠 Machine Learning Workflow

### 1. Dataset Loading

The project uses the Iris dataset available through Scikit-Learn.

Dataset Information:

| Attribute     | Value                      |
| ------------- | -------------------------- |
| Total Samples | 150                        |
| Features      | 4                          |
| Classes       | 3                          |
| Dataset Type  | Multi-Class Classification |

---

### 2. Data Splitting

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

This ensures reliable evaluation on unseen samples.

---

### 3. Model Training

The following classification algorithms were trained and evaluated:

#### Logistic Regression

A linear classification algorithm suitable for multi-class classification problems.

#### Decision Tree Classifier

A tree-based algorithm that learns decision boundaries from data.

#### Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees for improved robustness.

---

### 4. Model Evaluation

The models were evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

---

### 5. Model Selection

The best-performing model is automatically selected and saved for deployment.

Saved Model:

```text
models/iris_classifier.pkl
```

---

## 📈 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 96.67%   |
| Decision Tree       | 93.33%   |
| Random Forest       | 90.00%   |

### Best Performing Model

**Logistic Regression**

### Best Accuracy

**96.67%**

---

## 🖥️ Dashboard Features

### Prediction Dashboard

Users can input flower measurements using interactive sliders.

Input Parameters:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Output:

* Predicted Species
* Confidence Score
* Probability Distribution Chart

---

### Species Intelligence

Provides information about the predicted species, including:

* Key characteristics
* Classification insights
* Species-specific traits

---

### Dataset Analytics

Includes:

#### Species Distribution

Visual representation of class distribution in the dataset.

#### Correlation Heatmap

Shows relationships among numerical features.

#### Feature Relationship Analysis

Interactive scatter plot illustrating feature relationships and class separation.

---

### Dataset Preview

Interactive view of the Iris dataset used for training and analysis.

---

## 📂 Project Structure

```text
IrisInsight/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── iris_classifier.pkl
│
└── screenshots/
    ├── dashboard.png
    ├── analytics.png
    └── prediction.png
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-Learn

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Plotly
* Seaborn
* Matplotlib

### Dashboard Development

* Streamlit

### Model Persistence

* Joblib

### Version Control

* Git
* GitHub

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/AmbaramJHANSI/iris-insight-ml-dashboard.git
```

### Navigate to Project Directory

```bash
cd iris-insight-ml-dashboard
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### Train the Model

```bash
python train_model.py
```

### Launch the Dashboard

```bash
streamlit run app.py
```

---

## 📷 Application Screenshots

### Dashboard Overview

Add screenshot:

```text
screenshots/dashboard.png
```

### Prediction Example

Add screenshot:

```text
screenshots/prediction.png
```

### Analytics Dashboard

Add screenshot:

```text
screenshots/analytics.png
```

---

## 🔮 Future Improvements

* Model explainability using SHAP
* CSV file upload for batch predictions
* Dark mode / light mode toggle
* Advanced analytics dashboard
* Cloud database integration
* Docker containerization
* CI/CD automation
* Multi-model prediction comparison

---

## 📚 Learning Outcomes

This project demonstrates:

* Data preprocessing
* Machine Learning model training
* Model evaluation techniques
* Model persistence
* Interactive dashboard development
* Data visualization
* Git and GitHub workflow
* Deployment-ready application development

---

## 👨‍💻 Author
**Ambaram JHANSI**
Aspiring Machine Learning Engineer | Software Developer
GitHub:
https://github.com/AmbaramJHANSI

---

## 🙏 Acknowledgements

* Scikit-Learn
* UCI Machine Learning Repository
* Streamlit
* Plotly

---

### ⭐ If you found this project useful, consider giving it a star on GitHub.
