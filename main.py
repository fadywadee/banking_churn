from typing import Literal, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field, conint, confloat
from fastapi.middleware.cors import CORSMiddleware
import pickle
import threading

# ---------- Config ----------
MODEL_PATH = "Banking_Churn.sav"
FEATURE_ORDER = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Geography", "Gender"
]
GEO_MAP = {"France": 0, "Spain": 1, "Germany": 2}
GENDER_MAP = {"Male": 0, "Female": 1}

# ---------- API & CORS ----------
app = FastAPI(title="Banking Churn API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Model holder ----------
_model = {"clf": None}
_model_lock = threading.Lock()

def load_model():
    with _model_lock:
        if _model["clf"] is None:
            with open(MODEL_PATH, "rb") as f:
                _model["clf"] = pickle.load(f)
        return _model["clf"]

# ---------- Schemas ----------
class Payload(BaseModel):
    CreditScore: conint(ge=0, le=1000) = Field(..., description="0–1000")
    Age: conint(ge=0, le=120)
    Tenure: conint(ge=0, le=50)
    Balance: confloat(ge=0)
    NumOfProducts: conint(ge=1, le=10)
    HasCrCard: Literal[0,1]  # 1 yes, 0 no
    IsActiveMember: Literal[0,1]
    EstimatedSalary: confloat(ge=0)
    Geography: Literal["France", "Spain", "Germany"]
    Gender: Literal["Male", "Female"]

class Prediction(BaseModel):
    prediction: Literal[0,1]
    label: Literal["No Churn","Churn"]
    proba_churn: Optional[float] = None
    feature_order: list[str] = FEATURE_ORDER

# ---------- Startup ----------
@app.on_event("startup")
def _startup():
    # Fail fast if model cannot be loaded
    load_model()

# ---------- Endpoints ----------
@app.get("/health")
def health():
    try:
        _ = load_model()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/predict", response_model=Prediction)
def predict(payload: Payload):
    # Encode categorical
    geo = GEO_MAP[payload.Geography]
    gender = GENDER_MAP[payload.Gender]

    x = [
        int(payload.CreditScore),
        int(payload.Age),
        int(payload.Tenure),
        float(payload.Balance),
        int(payload.NumOfProducts),
        int(payload.HasCrCard),
        int(payload.IsActiveMember),
        float(payload.EstimatedSalary),
        int(geo),
        int(gender),
    ]

    clf = load_model()

    # shape: (1, n_features)
    y = clf.predict([x])[0]
    label = "Churn" if int(y) == 1 else "No Churn"

    proba = None
    if hasattr(clf, "predict_proba"):
        try:
            proba = float(clf.predict_proba([x])[0][1])
        except Exception:
            proba = None

    return Prediction(
        prediction=int(y),
        label=label,
        proba_churn=proba,
        feature_order=FEATURE_ORDER,
    )
