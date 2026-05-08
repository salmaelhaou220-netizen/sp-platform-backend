from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os, json, time

app = FastAPI(title="SP Platform API v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# ── System prompts (lire depuis fichiers ou inline) ────────────
SYSTEM_PROMPT_GENERATION = open("system_prompt_generation.txt").read() if os.path.exists("system_prompt_generation.txt") else ""
SYSTEM_PROMPT_EVALUATION = open("system_prompt_evaluation.txt").read() if os.path.exists("system_prompt_evaluation.txt") else ""

# ── Modèles ────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    module: str
    contenu: str
    type_sp: str = "didactique"       # didactique | formative | sommative
    profils_vark: List[str] = ["K"]   # ["V"], ["K","R"], ["selon_contexte"]
    niveau_difficulte: str = "intermédiaire"
    contexte_souhaite: str = "vie quotidienne"
    langue: str = "français"

class EvaluateRequest(BaseModel):
    module: str
    seance: str
    situation_probleme: str

class SaveSPRequest(BaseModel):
    titre: str
    module: str
    type_sp: str
    contenu: dict
    user_id: Optional[str] = "anonymous"

# ── Retry helper ───────────────────────────────────────────────
def call_gemini(prompt: str, system: str, retries: int = 3):
    for attempt in range(retries):
        try:
            full = f"{system}\n\n{prompt}"
            response = model.generate_content(full)
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            if attempt == retries - 1:
                raise HTTPException(500, f"JSON invalide: {str(e)}")
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == retries - 1:
                raise HTTPException(503, f"Erreur Gemini: {str(e)}")
            time.sleep(2 ** attempt)

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0", "message": "SP Platform API is running"}

@app.post("/generate-sp")
def generate_sp(req: GenerateRequest):
    profils = req.profils_vark if req.profils_vark else ["selon_contexte"]
    prompt = json.dumps({
        "module": req.module,
        "contenu": req.contenu,
        "type_sp": req.type_sp,
        "profils_vark": profils,
        "niveau_difficulte": req.niveau_difficulte,
        "contexte_souhaite": req.contexte_souhaite,
        "langue": req.langue
    }, ensure_ascii=False)
    result = call_gemini(prompt, SYSTEM_PROMPT_GENERATION)
    return {"success": True, "data": result}

@app.post("/evaluate-sp")
def evaluate_sp(req: EvaluateRequest):
    prompt = f"MODULE : {req.module}\nSÉANCE : {req.seance}\nSITUATION-PROBLÈME :\n{req.situation_probleme}"
    result = call_gemini(prompt, SYSTEM_PROMPT_EVALUATION)
    return {"success": True, "data": result}

@app.get("/dashboard/sp")
def get_dashboard():
    # Retourne des données mockées — à connecter à Supabase plus tard
    return {
        "success": True,
        "data": {
            "total_generes": 0,
            "total_evalues": 0,
            "sp_list": []
        }
    }

