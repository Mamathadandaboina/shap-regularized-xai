# SHAP-Regularized Attention Model  
## XGBoost + SHAP + Neural Network Attribution Alignment  

This repository presents a reproducible implementation of a SHAP-regularized attention-based neural network.  
The model is trained to align its gradient × input attributions with SHAP values generated from a strong XGBoost baseline.  

The objective is to build a transparent, reliable, and interpretable Explainable AI (XAI) framework for **tabular classification**, suitable for sensitive domains such as healthcare, finance, and digital risk assessment.

---

## Highlights

### 1. XGBoost Baseline With SHAP
- Computes global and local SHAP values  
- Serves as the gold-standard attribution reference  

### 2. SHAP-Regularized Attention Network
- Uses gradient × input attribution  
- Regularization term forces neural network attributions to match SHAP  
- Alignment loss:
Loss_total = BCE + λ * | SHAP - NN_attr |


### 3. Agreement Metrics
- Jaccard similarity (top-k feature overlap)  
- Spearman rank correlation  

### 4. Performance Summary
- Neural Network AUC: ≈ **0.999**  
- Mean Jaccard (top-8): ≈ **0.34**  
- Mean Spearman: ≈ **0.38**

---

## Repository Structure

/notebooks/  # Colab notebooks for preprocessing, SHAP, NN training

/src/        # Reusable scripts (models, utilities, training functions)

/experiments/   # Saved SHAP values, NN attributions, model checkpoints

/results/figures/  # Jaccard & Spearman plots, comparison figures

/docs/             # Technical documentation and project report


---

## Quick Start

### 1. Install Dependencies
Use the requirements file included in the repository:

pip install -r requirements.txt


### 2. Run the Main Notebook
Open the pipeline notebook:

notebooks/full_pipeline.ipynb


### The notebook performs:
- Data preprocessing  
- XGBoost SHAP baseline training  
- SHAP-regularized attention model training  
- Attribution extraction  
- Evaluation and metric computation  
- Plot and figure generation  

### 3. Access Generated Results
Outputs are saved under:

- `results/figures/`  
- `experiments/`  

---

## Requirements

This project uses:  
- Python 3.9+  
- PyTorch  
- XGBoost  
- SHAP  
- NumPy, Pandas, Scikit-learn  
- Matplotlib, Seaborn  

Install everything using:

pip install -r requirements.txt



---

## Contact

**Mamatha Dandaboina**  
Email: mamathanma1@gmail.com
