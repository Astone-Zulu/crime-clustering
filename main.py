from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------------------
# INIT APP
# ---------------------------
app = FastAPI()

# ---------------------------
# LOAD MODELS
# ---------------------------
crime_model = joblib.load("crime_model.pkl")
hotspot_model = joblib.load("hotspot_model.pkl")

danger_zones = hotspot_model.cluster_centers_

# ---------------------------
# INIT FIREBASE
# ---------------------------
firebase_config = json.loads(os.environ["FIREBASE_KEY"])

cred = credentials.Certificate(firebase_config)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
# ---------------------------
# REQUEST MODEL
# ---------------------------
class CrimeInput(BaseModel):
    lat: float
    lon: float
    hour: int
    day: int


# ---------------------------
# VALIDATION FUNCTION
# ---------------------------
def validate(lat, lon, hour, day):
    if lat < -90 or lat > 90:
        return False
    if lon < -180 or lon > 180:
        return False
    if hour < 0 or hour > 23:
        return False
    if day < 1 or day > 7:
        return False
    return True


# ---------------------------
# ROOT
# ---------------------------
@app.get("/")
def home():
    return {"message": "Smart Crime API Running "}


# ---------------------------


@app.post("/predict")
def predict(data: CrimeInput):

    lat = data.lat
    lon = data.lon
    hour = data.hour
    day = data.day

    # ---------------------------
    # VALIDATION
    # ---------------------------
    if not validate(lat, lon, hour, day):

        db.collection("crime_reports").add({
            "lat": lat,
            "lon": lon,
            "hour": hour,
            "day": day,
            "status": "error",
            "message": "Invalid input",
            "timestamp": datetime.utcnow()
        })

        return {"status": "error", "message": "Invalid input"}

    # ---------------------------
    # ML PREDICTION
    # ---------------------------
    risk = crime_model.predict([[lat, lon, hour, day]])[0]

    # ---------------------------
    # HOTSPOT CHECK
    # ---------------------------
    near_hotspot = False

    for zone in danger_zones:
        distance = np.sqrt((lat - zone[0])**2 + (lon - zone[1])**2)

        if distance < 0.03:
            near_hotspot = True
            break

    # ---------------------------
    # FINAL RESULT
    # ---------------------------
    if near_hotspot and risk == 1:
        result = "DANGER"

    elif risk == 1:
        result = "HIGH"

    else:
        result = "LOW"

    # ---------------------------
    # SAVE TO FIRESTORE
    # ---------------------------
    db.collection("crime_reports").add({
        "lat": lat,
        "lon": lon,
        "hour": hour,
        "day": day,
        "risk_level": result,
        "near_hotspot": near_hotspot,
        "status": "success",
        "timestamp": datetime.now().isoformat()
    })

    # ---------------------------
    # RESPONSE
    # ---------------------------
    return {
        "status": "success",
        "risk_level": result,
        "location": {
            "lat": lat,
            "lon": lon
        }
    }