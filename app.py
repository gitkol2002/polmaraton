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
# 3. Load Model (scikit-learn + joblib)
# =======================================
@st.cache_resource
def load_model():
    """
    Wczytuje model:
    1. Najpierw próbuje z lokalnego katalogu ./models/
    2. Jeśli nie ma lokalnie, pobiera z S3 i ładuje z .joblib
    """
    LOCAL_MODEL_PATH = "models/model_polmaraton_splity.joblib"

    # 1. Sprawdź lokalny model
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            print(f"✔ Wczytuję model lokalnie: {LOCAL_MODEL_PATH}")
            return joblib.load(LOCAL_MODEL_PATH)
        except Exception as e:
            st.warning(f"⚠️ Błąd wczytywania lokalnego modelu: {str(e)}")
            st.info("📥 Próbuję pobrać model z S3...")

    # 2. Pobierz model z S3
    s3_path = download_model_from_s3()
    if s3_path is None:
        st.error("❌ Nie udało się pobrać modelu z S3")
        return None

    try:
        print(f"✔ Wczytuję model z S3: {s3_path}")
        return joblib.load(s3_path)
    except Exception as e:
        st.error(f"❌ Błąd ładowania modelu z S3: {str(e)}")
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


# =======================================
# 6. STREAMLIT UI
# =======================================
st.set_page_config(
    page_title="Predykcja Półmaratonu",
    page_icon="🏃",
    layout="centered"
)

st.title("🏃‍♂️ Predykcja Półmaratonu przez AI")
st.write("Aplikacja przewiduje Twój czas półmaratonu na podstawie danych treningowych.")


# ----- Session state -----
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {"sex": None, "age": None, "time_5km": None}

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# =======================================
# UI – Step 1: AI Extraction
# =======================================
st.subheader("📝 Krok 1: Wprowadź opis")

user_text = st.text_area(
    "Napisz coś o sobie:",
    placeholder="Np. Mam 33 lata, jestem mężczyzną, biegam 5 km w 22:15.",
    height=100,
    help="AI automatycznie wyłuska dane z Twojego opisu"
)

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🔎 Wyłuskaj dane AI", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analizuję tekst..."):
                extracted = extract_data(user_text)
                st.session_state.extracted_data = extracted
                
                # Sprawdź które dane zostały znalezione
                found_data = []
                missing_data = []
                
                if extracted.get("sex"):
                    found_data.append("płeć")
                else:
                    missing_data.append("płeć")
                
                if extracted.get("age"):
                    found_data.append("wiek")
                else:
                    missing_data.append("wiek")
                
                if extracted.get("time_5km"):
                    found_data.append("czas 5 km")
                else:
                    missing_data.append("czas 5 km")
                
                # Wyświetl odpowiedni komunikat
                if len(found_data) == 3:
                    st.success("✅ Wszystkie dane wyłuskane! Sprawdź i popraw poniżej jeśli trzeba.")
                elif len(found_data) > 0:
                    st.warning(f"⚠️ Znaleziono: **{', '.join(found_data)}**. Brakuje: **{', '.join(missing_data)}**. Uzupełnij ręcznie poniżej.")
                else:
                    st.error("❌ Nie znaleziono żadnych danych w tekście. Wprowadź je ręcznie poniżej.")
                    st.info("💡 Spróbuj podać więcej informacji, np. 'Mam 30 lat, jestem mężczyzną, mój czas na 5 km to 22:15'")
        else:
            st.warning("Wprowadź najpierw tekst do analizy.")

with col2:
    if st.button("🔄 Wyczyść wszystko", use_container_width=True):
        st.session_state.extracted_data = {"sex": None, "age": None, "time_5km": None}
        st.session_state.prediction_result = None
        st.rerun()


# =======================================
# UI – Step 2: Manual Input
# =======================================
st.divider()
st.subheader("✏️ Krok 2: Dane wejściowe")

col1, col2, col3 = st.columns(3)

with col1:
    sex_options = ["", "M", "K"]
    current_sex = st.session_state.extracted_data.get("sex") or ""
    sex_index = sex_options.index(current_sex) if current_sex in sex_options else 0
    
    sex = st.selectbox(
        "Płeć:",
        sex_options,
        index=sex_index,
        help="M - mężczyzna, K - kobieta"
    )

with col2:
    default_age = st.session_state.extracted_data.get("age")
    age = st.number_input(
        "Wiek:",
        min_value=0,
        max_value=100,
        value=int(default_age) if default_age else 0,
        help="Twój wiek w latach",
        placeholder="Podaj wiek"
    )

with col3:
    default_t5 = st.session_state.extracted_data.get("time_5km")
    t5 = st.number_input(
        "Czas 5 km (sekundy):",
        min_value=0,
        max_value=5000,
        value=int(default_t5) if default_t5 else 0,
        help="Twój najlepszy czas na 5 km w sekundach (np. 1335 = 22:15)",
        placeholder="Podaj czas w sekundach"
    )

if t5 > 0:
    st.caption(f"💡 Czas 5 km: **{format_time(t5)}** (tempo: **{format_time(t5/5)}/km**)")


# =======================================
# UI – Step 3: Prediction
# =======================================
st.divider()
st.subheader("🏁 Krok 3: Oblicz przewidywany czas")

if st.button("🚀 Oblicz czas półmaratonu", type="primary", use_container_width=True):
    if not sex or sex == "":
        st.error("❌ Wybierz płeć!")
    elif age <= 0:
        st.error("❌ Podaj wiek (musi być większy niż 0)!")
    elif t5 <= 0:
        st.error("❌ Podaj czas 5 km (musi być większy niż 0)!")
    elif t5 < 60:
        st.error("❌ Czas 5 km jest zbyt krótki (minimum 60 sekund = 1 minuta)!")
    else:
        try:
            with st.spinner("Pobieranie modelu i obliczanie predykcji..."):
                predicted, tempo = predict_time(sex, age, t5)
                st.session_state.prediction_result = {
                    "time": format_time(predicted),
                    "seconds": predicted,
                    "tempo": tempo
                }
                st.balloons()
        except Exception as e:
            st.error(f"❌ Błąd podczas predykcji: {str(e)}")


# =======================================
# UI – Results
# =======================================
if st.session_state.prediction_result:
    st.divider()
    st.subheader("📊 Twój przewidywany wynik")
    
    result = st.session_state.prediction_result
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Czas półmaratonu",
            value=result["time"]
        )
    
    with col2:
        st.metric(
            label="Tempo na km",
            value=format_time(result["tempo"])
        )
    
    with col3:
        distance_km = 21.0975
        avg_speed = (distance_km / result["seconds"]) * 3600
        st.metric(
            label="Średnia prędkość",
            value=f"{avg_speed:.2f} km/h"
        )
    
    st.info("💡 **Pamiętaj:** To tylko predykcja oparta na modelu. Rzeczywisty wynik może się różnić w zależności od treningu, warunków pogodowych i dnia startu!")


# =======================================
# Footer
# =======================================
st.divider()
st.caption("🔗 Aplikacja wykorzystuje OpenAI, Langfuse, PyCaret i DigitalOcean Spaces (S3).")