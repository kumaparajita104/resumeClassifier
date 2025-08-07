# GenAI-Powered Resume Classification System  
## Intelligent Role Prediction using Transformers + Similarity Scoring  

**Presented by**: Aparajita Kumari

---

## 🧩 Problem Statement

- Manual resume screening is time-consuming
- Frequent skill-role mismatches during shortlisting
- Resume formats vary greatly, making automation difficult

---

## 🎯 Project Goals

- Automatically classify resumes into job roles
- Compute similarity between resumes and job descriptions
- Store predictions and similarity scores in a database
- Enable decision support for HR using accurate predictions

---

## ⚙️ End-to-End Architecture

- **Input**: Resume text (PDF/DOCX → Text)
- **Preprocessing**: Cleaning text, label encoding
- **Embedding**: Sentence-BERT (SBERT) for dense vectors
- **Training**: Classical ML + Calibration
- **Inference**: API exposes predictions + similarity scoring
- **Deployment**: FastAPI + React + Docker Compose

---

## 🧼 Resume Preprocessing

- **Text Extraction**: AWS Lambda + S3 (Initially)
- **Text Cleaning**: Regex (removing noise, emails, numbers)
- **Label Encoding**: Sklearn’s `LabelEncoder`

---

## 🤖 Model Training

### Embedding with SBERT:
- Captures **semantic meaning** of resume
- Outperforms traditional TF-IDF/BOW
- Useful for both classification and similarity

### Algorithms Trained:
- Logistic Regression *(baseline)*
- Random Forest *(non-linear features)*
-  SVM *(good for high-dimensional space)*
- MLP *(basic deep learning approach)*

---

## Best Model Selection

| Model           | Accuracy |
|----------------|----------|
| Logistic Reg.  | 70.42%   |
| SVM            | 69.82%   |
| MLP            | 65.59%   |
| Random Forest  | 62.78%   |

**Selected Model**: Logistic Regression (Balanced)  
Calibrated using `CalibratedClassifierCV` (Isotonic)

---

## Model Calibration

> "Model calibration aligns predicted confidence scores with actual accuracy."

**Problem**:  
Model was **underconfident** – it predicted with 60% confidence but was correct 90% of the time.

**Solution**:
- Used `CalibratedClassifierCV` to improve confidence estimation
- Helps in thresholding and improving trust in predictions

---

## Evaluation & Inference

### Inference Example:

```python
resume = "Experienced HR professional with 5+ years..."
predict_resume(resume) 
# Predicted Role: HR (Confidence: 84.25%)
```

- Supports thresholding (e.g., flag resumes with <30% confidence)
- Allows human-in-loop where needed

---

##  System Deployment

- Built **FastAPI backend** with endpoints to:
  - Classify resume text
  - Compute similarity with job descriptions
  - Store results in MySQL

- **Frontend**: React UI for uploads & visualization
- **Containerized**: via Docker Compose (backend + frontend)

---

## Challenges & Solutions

| Challenge                     | Solution                                                |
|------------------------------|---------------------------------------------------------|
| AWS Deployment Abandoned     | Switched to **Docker Compose** locally                  |
| Docker Networking Conflicts  | Fixed with **CORS setup** and clean Dockerfiles         |
| Underconfident Predictions   | Used **Model Calibration** + Confidence Thresholding    |

---

## Future Improvements

- Fine-tune **LLaMA-2** or **DeBERTa** on domain data
- Add **human-in-loop** for low-confidence resumes
- Reintroduce AWS deployment with logging/monitoring
- Explore **Bayesian calibration** for uncertainty estimation

