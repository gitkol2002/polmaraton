import streamlit as st
import json
import datetime
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from pycaret.regression import load_model as pyc_load


# =======================================
# 1. ENV + OpenAI
# =======================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =======================================
# 2. AI → Extract Data
# =======================================
def extract_data(text: str) -> dict:
    """
    AI wyciąga płeć, wiek i czas 5 km.
    Rozpoznaje płeć także z imienia (analiza końcówki).
    """
    if not text or not text.strip():
        return {"sex": None, "age": None, "time_5km": None}

    system_prompt = """
    You extract structured running data from Polish text.
    Return ONLY valid JSON.

    JSON structure:
    {
      "sex": "M" | "K" | null,
      "age": number | null,
      "time_5km": number | null
    }

    Rules:
    - If name ends with 'a' → female ("K"),
      except: ["Kuba", "Barnaba", "Bonawentura", "Kacper"]
    - If masculine name → "M"
    - If unclear → null
    - Time formats accepted: "22:15" (MM:SS), "1:22:15" (H:MM:SS)
    - Convert time to seconds (e.g., 22:15 = 1335 seconds)
    - Age must be reasonable (1-100)
    """

    user_prompt = f"""
    Extract JSON from this text:
    \"\"\"{text}\"\"\"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)
        
        # Walidacja danych
        if data.get("age") and (data["age"] < 1 or data["age"] > 100):
            data["age"] = None
        if data.get("time_5km") and (data["time_5km"] < 60 or data["time_5km"] > 5000):
            data["time_5km"] = None
            
        return data

    except Exception as e:
        st.error(f"Błąd podczas ekstrakcji danych: {str(e)}")
        return {"sex": None, "age": None, "time_5km": None}


# =======================================
# 3. Load PyCaret Model
# =======================================
@st.cache_resource
def load_model():
    """Cache model to avoid reloading on every interaction"""
    try:
        return pyc_load("models/model_polmaraton_splity")
    except Exception as e:
        st.error(f"Błąd wczytywania modelu: {str(e)}")
        return None


# =======================================
# 4. Prediction
# =======================================
def predict_time(sex: str, age: int, t5: float) -> tuple:
    """
    Tworzy sztuczne splity z tempa i przewiduje czas półmaratonu.
    Zwraca (predicted_time_seconds, tempo_per_km)
    """
    if not sex or not age or not t5:
        raise ValueError("Wszystkie dane są wymagane")
    
    tempo = t5 / 5  # tempo na kilometr

    df = pd.DataFrame([{
        "Płeć": sex,
        "Wiek": int(age),
        "5 km Czas": float(t5),
        "10 km Czas": tempo * 10,
        "15 km Czas": tempo * 15,
        "20 km Czas": tempo * 20
    }])

    model = load_model()
    if model is None:
        raise ValueError("Model nie został wczytany")
    
    pred = model.predict(df)[0]
    return int(pred), tempo


def format_time(sec: float) -> str:
    """Formatuje sekundy na czytelny czas HH:MM:SS"""
    return str(datetime.timedelta(seconds=int(sec)))


# =======================================
# 5. STREAMLIT UI
# =======================================
st.set_page_config(
    page_title="Predykcja Półmaratonu",
    page_icon="🏃",
    layout="centered"
)

st.title("🏃‍♂️ Predykcja Półmaratonu — AI")
st.write("Wpisz opis lub podaj dane ręcznie, aby przewidzieć swój czas na półmaratonie.")


# ---------------------------------------
# Inicjalizacja stanu sesji
# ---------------------------------------
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {"sex": None, "age": None, "time_5km": None}

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ---------------------------------------
# Pole tekstowe
# ---------------------------------------
with st.container():
    st.subheader("📝 Krok 1: Opis")
    user_text = st.text_area(
        "Opisz się (opcjonalnie):",
        placeholder="Np. Cześć, jestem Paweł, mam 33 lata, biegam 5 km w 22:15.",
        height=100,
        help="AI spróbuje automatycznie wyłuskać dane z Twojego opisu"
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


# ---------------------------------------
# Ręczna edycja danych
# ---------------------------------------
st.divider()
st.subheader("✏️ Krok 2: Dane wejściowe")

col1, col2, col3 = st.columns(3)

with col1:
    sex_options = ["", "M", "K"]
    sex_index = 0
    if st.session_state.extracted_data.get("sex") == "M":
        sex_index = 1
    elif st.session_state.extracted_data.get("sex") == "K":
        sex_index = 2
    
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

# Pokazuj czas w formacie czytelnym tylko jeśli t5 > 0
if t5 > 0:
    st.caption(f"💡 Czas 5 km: **{format_time(t5)}** (tempo: **{format_time(t5/5)}/km**)")


# ---------------------------------------
# Predykcja
# ---------------------------------------
st.divider()
st.subheader("🏁 Krok 3: Oblicz przewidywany czas")

if st.button("🚀 Oblicz czas półmaratonu", type="primary", use_container_width=True):
    if not sex or sex == "":
        st.error("❌ Wybierz płeć!")
    elif not age or age == 0:
        st.error("❌ Podaj wiek (musi być większy niż 0)!")
    elif not t5 or t5 == 0:
        st.error("❌ Podaj czas 5 km (musi być większy niż 0)!")
    elif t5 < 60:
        st.error("❌ Czas 5 km jest zbyt krótki (minimum 60 sekund = 1 minuta)!")
    else:
        try:
            with st.spinner("Obliczam predykcję..."):
                pred_seconds, tempo = predict_time(sex, age, t5)
                st.session_state.prediction_result = {
                    "time": format_time(pred_seconds),
                    "seconds": pred_seconds,
                    "tempo": tempo
                }
                st.balloons()
        except Exception as e:
            st.error(f"Błąd podczas predykcji: {str(e)}")


# ---------------------------------------
# Wyświetlenie wyniku
# ---------------------------------------
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


# ---------------------------------------
# Footer
# ---------------------------------------
st.divider()
st.caption("Aplikacja wykorzystuje model AI do predykcji czasu półmaratonu na podstawie danych treningowych.")