import streamlit as st
import json
import datetime
import pandas as pd
import os
import tempfile
from dotenv import load_dotenv
import joblib
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# ================================
# 0. Streamlit config (MUST BE FIRST)
# ================================
st.set_page_config(
    page_title="Predykcja Półmaratonu",
    page_icon="🏃",
    layout="centered"
)

# ================================
# 1. ENV
# ================================
load_dotenv()

# Langfuse / OpenAI
from langfuse import Langfuse
from langfuse.openai import OpenAI as LangfuseOpenAI

LANGFUSE_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC,
    secret_key=LANGFUSE_SECRET,
    host=LANGFUSE_HOST
)

client = LangfuseOpenAI(api_key=OPENAI_KEY)

# ================================
# 2. DigitalOcean S3
# ================================
BUCKET_NAME = "maraton"
MODEL_S3_KEY = "models/model_polmaraton_splity.pkl"

s3 = boto3.client(
    "s3",
    region_name="fra1",
    endpoint_url="https://fra1.digitaloceanspaces.com",
    aws_access_key_id=os.getenv("SPACES_KEY"),
    aws_secret_access_key=os.getenv("SPACES_SECRET")
)


# ================================
# 3. BACKEND: S3 check + download
# ================================
def check_s3_model_exists():
    """Check if model exists in S3 — NO Streamlit here."""
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=MODEL_S3_KEY)
        print(f"✔ Model exists in S3: {MODEL_S3_KEY}")
        return True
    except Exception as e:
        print(f"❌ Model missing in S3: {MODEL_S3_KEY}")
        return False


def download_model_from_s3():
    """Download model from S3 — backend only (no Streamlit)."""
    try:
        tmp_dir = tempfile.gettempdir()
        local_dir = os.path.join(tmp_dir, "models")
        os.makedirs(local_dir, exist_ok=True)

        local_path = os.path.join(local_dir, "model_polmaraton_splity.pkl")

        if os.path.exists(local_path):
            print("✔ Model already downloaded.")
            return local_path

        if not check_s3_model_exists():
            return None

        print("📥 Downloading model from S3...")
        s3.download_file(BUCKET_NAME, MODEL_S3_KEY, local_path)
        print(f"✔ Downloaded to {local_path}")

        return local_path

    except Exception as e:
        print(f"❌ S3 download error: {e}")
        return None


# ================================
# 4. Load model (backend safe)
# ================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Safe load — never uses Streamlit printouts inside."""
    local_repo_path = "models/model_polmaraton_splity.pkl"

    # 1. Try local repo model
    if os.path.exists(local_repo_path):
        try:
            print("✔ Loading local model...")
            return joblib.load(local_repo_path)
        except Exception as e:
            print(f"⚠ Local model read error: {e}")

    # 2. Download from S3
    model_path = download_model_from_s3()
    if model_path is None:
        return None

    try:
        print("✔ Loading S3 model...")
        return joblib.load(model_path)
    except Exception as e:
        print(f"❌ Joblib load error: {e}")
        return None


# ================================
# 5. Preload
# ================================
@st.cache_resource(show_spinner=False)
def preload_model():
    return load_model()

_ = preload_model()


# ================================
# 6. AI Extraction
# ================================
def extract_data(text: str) -> dict:
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

    TIME rules:
    - MM:SS → convert to seconds
    - H:MM:SS
    - "25 min" → 1500
    - valid range 300–5000 sec
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

        if data.get("time_5km") and not (300 <= data["time_5km"] <= 5000):
            data["time_5km"] = None

        return data

    except Exception as e:
        st.error(f"❌ Błąd AI: {e}")
        return {"sex": None, "age": None, "time_5km": None}


# ================================
# 7. Prediction
# ================================
def predict_time(sex: str, age: int, t5: float):
    model = load_model()
    if model is None:
        raise ValueError("Model nie został wczytany.")

    tempo = t5 / 5

    df = pd.DataFrame([{
        "Płeć": sex,
        "Wiek": age,
        "5 km Czas": t5,
        "10 km Czas": tempo * 10,
        "15 km Czas": tempo * 15,
        "20 km Czas": tempo * 20
    }])

    pred = model.predict(df)[0]
    return int(pred), tempo


def format_time(sec):
    return str(datetime.timedelta(seconds=int(sec)))


# ================================
# 8. UI
# ================================
st.title("🏃‍♂️ Predykcja Półmaratonu przez AI")
st.write("Podaj dane lub pozwól AI je wyłuskać.")

# --- Session state ---
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {"sex": None, "age": None, "time_5km": None}
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# ----------------------------
# STEP 1 — AI EXTRACTION
# ----------------------------
st.subheader("📝 Krok 1: Opis biegania")

text = st.text_area("Wpisz opis:", height=100)

if st.button("🔎 Wyłuskaj AI"):
    with st.spinner("Analiza AI..."):
        st.session_state.extracted_data = extract_data(text)
    st.rerun()

if st.button("🔄 Reset"):
    st.session_state.extracted_data = {"sex": None, "age": None, "time_5km": None}
    st.session_state.prediction_result = None
    st.rerun()

# ----------------------------
# STEP 2 — MANUAL INPUT
# ----------------------------
st.subheader("✏️ Krok 2: Dane wejściowe")

col1, col2, col3 = st.columns(3)

with col1:
    sex = st.selectbox("Płeć", ["", "M", "K"],
                       index=["", "M", "K"].index(st.session_state.extracted_data["sex"] or ""))

with col2:
    age = st.number_input(
        "Wiek",
        min_value=1, max_value=100,
        value=st.session_state.extracted_data["age"] or 25
    )

with col3:
    t5 = st.number_input(
        "Czas 5 km (sekundy)",
        min_value=1, max_value=5000,
        value=st.session_state.extracted_data["time_5km"] or 1500
    )

if t5 > 0:
    st.caption(f"Czas 5 km: {format_time(t5)}")


# ----------------------------
# STEP 3 — PREDICTION
# ----------------------------
st.subheader("🏁 Krok 3: Wynik")

if st.button("🚀 Oblicz czas"):
    try:
        with st.spinner("Liczenie..."):
            pred, tempo = predict_time(sex, age, t5)
            st.session_state.prediction_result = {
                "time": format_time(pred),
                "seconds": pred,
                "tempo": tempo
            }
        st.balloons()
    except Exception as e:
        st.error(f"❌ Błąd predykcji: {e}")

# ----------------------------
# RESULT
# ----------------------------
if st.session_state.prediction_result:
    st.subheader("📊 Wynik")
    r = st.session_state.prediction_result

    col1, col2, col3 = st.columns(3)
    col1.metric("Czas półmaratonu", r["time"])
    col2.metric("Tempo", format_time(r["tempo"]))
    col3.metric("Prędkość [km/h]", f"{(21.0975 / r['seconds']) * 3600:.2f}")

