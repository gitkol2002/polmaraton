import streamlit as st
import json
import datetime
import pandas as pd
import os
import tempfile
from dotenv import load_dotenv

# ====== AI / Langfuse ======
from langfuse import Langfuse
from langfuse.openai import OpenAI as LangfuseOpenAI

# ====== Model (scikit-learn + joblib zamiast PyCaret) ======
import joblib

# ====== S3 (DigitalOcean Spaces) ======
import boto3
from botocore.exceptions import NoCredentialsError, ClientError


# =======================================
# 0. ENV
# =======================================
load_dotenv()

# Langfuse
LANGFUSE_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC,
    secret_key=LANGFUSE_SECRET,
    host=LANGFUSE_HOST
)

# OpenAI (wrapped by Langfuse)
client = LangfuseOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =======================================
# 1. DigitalOcean Spaces (S3)
# =======================================
BUCKET_NAME = "maraton"
MODEL_S3_KEY = "models/model_polmaraton_splity.joblib"  # teraz .joblib

s3 = boto3.client(
    "s3",
    region_name="fra1",
    endpoint_url="https://fra1.digitaloceanspaces.com",
    aws_access_key_id=os.getenv("SPACES_KEY"),
    aws_secret_access_key=os.getenv("SPACES_SECRET")
)


# =======================================
# 2. Download Model from S3
# =======================================
def download_model_from_s3():
    """
    Pobiera model z S3 i zapisuje go lokalnie (format joblib).
    Zwraca pełną ścieżkę do pliku .joblib
    """
    try:
        temp_dir = tempfile.gettempdir()
        local_model_dir = os.path.join(temp_dir, "models")
        os.makedirs(local_model_dir, exist_ok=True)

        local_model_path = os.path.join(local_model_dir, "model_polmaraton_splity.joblib")

        # jeśli już istnieje — nie pobieraj drugi raz
        if os.path.exists(local_model_path):
            print("✔ Model już istnieje lokalnie")
            return local_model_path

        print(f"📥 Pobieram model z S3: {BUCKET_NAME}/{MODEL_S3_KEY}")
        s3.download_file(BUCKET_NAME, MODEL_S3_KEY, local_model_path)
        print(f"✔ Model pobrany do: {local_model_path}")

        return local_model_path

    except Exception as e:
        st.error(f"❌ Błąd pobierania modelu: {str(e)}")
        return None


# =======================================
# 3. Load model (joblib + sklearn)
# =======================================
@st.cache_resource
def load_model():
    """
    Wczytuje model w formacie .joblib
    """
    LOCAL_PATH = "models/model_polmaraton_splity.joblib"

    # 1. Najpierw próbuj wczytać lokalnie (repozytorium)
    if os.path.exists(LOCAL_PATH):
        try:
            return joblib.load(LOCAL_PATH)
        except Exception:
            st.warning("⚠️ Lokalny model uszkodzony — pobieram z S3...")

    # 2. Pobierz z S3
    model_path = download_model_from_s3()
    if model_path is None:
        return None

    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Błąd ładowania modelu: {str(e)}")
        return None


# =======================================
# 4. AI → Extract Running Data
# =======================================
def extract_data(text: str) -> dict:
    """
    Wyłuskuje płeć, wiek i czas 5 km za pomocą OpenAI + Langfuse.
    """
    if not text.strip():
        return {"sex": None, "age": None, "time_5km": None}

    system_prompt = """
    Extract running-related data from Polish text.
    Return ONLY JSON:

    {
      "sex": "M" | "K" | null,
      "age": number | null,
      "time_5km": number | null
    }
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        data = json.loads(response.choices[0].message.content)

        # Walidacja
        if data.get("age") and not (1 <= data["age"] <= 100):
            data["age"] = None
        if data.get("time_5km") and not (60 <= data["time_5km"] <= 5000):
            data["time_5km"] = None

        return data

    except Exception as e:
        st.error(f"❌ Błąd AI: {str(e)}")
        return {"sex": None, "age": None, "time_5km": None}


# =======================================
# 5. Prediction Logic (sklearn)
# =======================================
def predict_time(sex: str, age: int, t5: float):
    model = load_model()
    if model is None:
        raise ValueError("Model nie został wczytany")

    tempo = t5 / 5

    df = pd.DataFrame([{
        "Płeć": sex,
        "Wiek": int(age),
        "5 km Czas": float(t5),
        "10 km Czas": tempo * 10,
        "15 km Czas": tempo * 15,
        "20 km Czas": tempo * 20
    }])

    pred = model.predict(df)[0]
    return int(pred), tempo


def format_time(sec: float) -> str:
    return str(datetime.timedelta(seconds=int(sec)))


# =====================================
