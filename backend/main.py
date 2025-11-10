import pandas as pd
import numpy as np
import joblib
import re
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util


# -------------------------------------
# FastAPI App
# -------------------------------------
app = FastAPI(title="Resume Classifier API")

# -------------------------------------
# Load Model & Encoder
# -------------------------------------
classifier = joblib.load("models/calibrated_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------
# Load JD CSV and Preprocess
# -------------------------------------
jd_df = pd.read_csv("job_descriptions.csv")  # Expecting columns: Role, JD_Text
if not {"Role", "JD_Text"}.issubset(jd_df.columns):
    raise Exception("Missing 'Role' or 'JD_Text' columns in JD file.")

jd_df.dropna(subset=["Role", "JD_Text"], inplace=True)

def clean_text(text: str) -> str:
    text = re.sub(r'\d{10}', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

jd_df["clean_jd"] = jd_df["JD_Text"].apply(clean_text)
jd_embeddings = model.encode(jd_df["clean_jd"].tolist(), show_progress_bar=False)

# -------------------------------------
# DB Connection
# -------------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="host.docker.internal",
        port=3307,
        user="root",
        password="",
        database="resume_db"
    )

# -------------------------------------
# Pydantic Input Schema
# -------------------------------------
class ResumeRequest(BaseModel):
    resume_text: str
    top_k: int = 3
    threshold: float = 0.3

# -------------------------------------
# Middleware
# -------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------
# API Endpoint
# -------------------------------------
@app.post("/predict")
def analyze_resume(request: ResumeRequest):
    resume_text = request.resume_text
    top_k = request.top_k
    

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text is empty.")

    # Clean and embed
    clean_resume = clean_text(resume_text)
    emb = model.encode([clean_resume])

    # Predict label
    probs = classifier.predict_proba(emb)[0]
    max_idx = np.argmax(probs)
    label = label_encoder.inverse_transform([max_idx])[0]
    confidence = probs[max_idx]

    # Store in DB
    conn = get_db_connection()
    cursor = conn.cursor()

    insert_resume = """
        INSERT INTO resumes (resume_text, predicted_category, confidence_score)
        VALUES (%s, %s, %s)
    """
    cursor.execute(insert_resume, (resume_text, label, float(confidence)))
    resume_id = cursor.lastrowid

    # Compute top K similarity scores
    sims = util.cos_sim(emb, jd_embeddings)[0]
    top_indices = sims.argsort(descending=True)[:top_k].cpu().numpy().tolist()  # ✅ Convert tensor to list of ints

    for idx in top_indices:
        idx = int(idx)  # Just in case
        role = jd_df.iloc[idx]["Role"]
        sim_score = float(sims[idx].item())

        insert_sim = """
            INSERT INTO similarity_scores (resume_id, job_role, similarity_score)
            VALUES (%s, %s, %s)
        """
        cursor.execute(insert_sim, (resume_id, role, sim_score))

    conn.commit()
    cursor.close()
    conn.close()

    similar_roles = [
        {
            "job_role": jd_df.iloc[idx]["Role"],
            "similarity_score": float(sims[idx])
        }
        for idx in top_indices
    ]

    return {
        "resume_id": resume_id,
        "predicted_category": label,
        "confidence_score": round(confidence, 4),
        "similar_roles": similar_roles
    }
