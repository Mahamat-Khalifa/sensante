# api/main.py
# SenSante API - Assistant pre-diagnostic medical
# Lab 3 → Lab 5 - Integration de Modeles IA - ESP/UCAD

import os
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq

# ─────────────────────────────────────────────
#  Variables d'environnement + client Groq
# ─────────────────────────────────────────────
load_dotenv()

groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. /explain sera desactive.")

# ─────────────────────────────────────────────
#  Application FastAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Chargement du modele ML (une seule fois)
# ─────────────────────────────────────────────
print("Chargement du modele...")
model       = joblib.load("models/model.pkl")
le_sexe     = joblib.load("models/encoder_sexe.pkl")
le_region   = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modele charge : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")

# ─────────────────────────────────────────────
#  Schemas Pydantic
# ─────────────────────────────────────────────
class PatientInput(BaseModel):
    age:         int   = Field(..., ge=0, le=120,   description="Age en annees")
    sexe:        str   = Field(...,                  description="Sexe : M ou F")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Temperature en Celsius")
    tension_sys: int   = Field(..., ge=60, le=250,  description="Tension systolique")
    toux:        bool  = Field(...,                  description="Presence de toux")
    fatigue:     bool  = Field(...,                  description="Presence de fatigue")
    maux_tete:   bool  = Field(...,                  description="Presence de maux de tete")
    region:      str   = Field(...,                  description="Region du Senegal")

class DiagnosticOutput(BaseModel):
    diagnostic:  str   = Field(..., description="Diagnostic predit")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    confiance:   str   = Field(..., description="Niveau de confiance")
    message:     str   = Field(..., description="Recommandation")

class ExplainInput(BaseModel):
    diagnostic:  str   = Field(..., description="Diagnostic predit par le modele")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    age:         int   = Field(..., description="Age du patient")
    sexe:        str   = Field(..., description="Sexe du patient")
    temperature: float = Field(..., description="Temperature en Celsius")
    region:      str   = Field(..., description="Region du Senegal")

class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en francais")
    modele_llm:  str = Field(default="llama-3.1-8b-instant", description="Modele LLM utilise")

# ─────────────────────────────────────────────
#  System prompt LLM
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Tu es un assistant medical senegalais.\n"
    "Tu recois un diagnostic et des donnees patient.\n"
    "Explique le resultat en francais simple,\n"
    "comme un medecin parlerait a son patient.\n"
    "Sois rassurant mais recommande toujours une consultation medicale.\n"
    "Maximum 3 phrases.\n"
    "Ne fais JAMAIS de diagnostic toi-meme.\n"
    "Tu expliques uniquement le diagnostic fourni."
)

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Verification de l'etat de l'API."""
    return {"status": "ok", "message": "SenSante API is running"}


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    """Predire un diagnostic a partir des symptomes d'un patient."""

    # 1. Encoder les variables categoriques
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0, confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F."
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0, confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # 2. Construire le vecteur de features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # 3. Predire
    diagnostic = model.predict(features)[0]
    probas     = model.predict_proba(features)[0]
    proba_max  = float(probas.max())

    # 4. Niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # 5. Recommandation
    messages = {
        "palu":   "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typh":   "Suspicion de typhoide. Consultation medicale necessaire.",
        "sain":   "Pas de pathologie detectee. Continuez a surveiller."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )


@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en francais avec Llama 3 via Groq."""

    # Degradation gracieuse si la cle API est absente
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Cle API non configuree.",
            modele_llm="aucun"
        )

    # Construire le user prompt avec les donnees reelles du patient
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content

    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"

    return ExplainOutput(explication=explication)
