import streamlit as st
import tensorflow as tf
import tf_keras
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path
from datetime import date
import sqlite3
import uuid


# ============================================================
# GRUNDEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Farben
DUNKELBLAU = "#183447"
DUNKELBLAU_2 = "#21465D"
METALL_GRAU = "#68727A"
METALL_GRAU_DUNKEL = "#59636B"
WEISS = "#FFFFFF"
SCHWARZ = "#111111"
HELLBLAU = "#DCECF3"

MODEL_PATH = Path("keras_model.h5")
LABELS_PATH = Path("labels.txt")

UPLOAD_DIR = Path("fundgrube_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DB_PATH = Path("fundgrube.db")


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    f"""
    <style>

    /* --------------------------------------------------------
       Gesamte App
       -------------------------------------------------------- */

    .stApp {{
        background:
            linear-gradient(
                135deg,
                #102B3B 0%,
                #183447 35%,
                #1E4055 65%,
                #122D3E 100%
            );
        color: {WEISS};
    }}

    /* Hauptbereich etwas kompakter */
    .block-container {{
        max-width: 1150px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }}

    /* --------------------------------------------------------
       Normale Schrift
       -------------------------------------------------------- */

    html, body, [class*="css"] {{
        font-family: Arial, Helvetica, sans-serif;
    }}

    p, label, span {{
        color: {WEISS};
    }}

    /* --------------------------------------------------------
       Seitenüberschrift
       -------------------------------------------------------- */

    .page-title-box {{
        background:
            linear-gradient(
                145deg,
                #737D84,
                #5B666E
            );
        border: 2px solid #879198;
        border-radius: 20px;

        padding: 30px 36px;
        margin-bottom: 35px;

        box-shadow:
            0 8px 20px rgba(0,0,0,0.30),
            inset 0 1px 0 rgba(255,255,255,0.20);
    }}

    .page-title {{
        color: white;
        font-size: 3.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.5px;
    }}

    .page-subtitle {{
        color: #F3F5F6;
        font-size: 1.05rem;
        margin-top: 8px;
    }}


    /* ========================================================
       SUCHFELD
       ======================================================== */

    /*
       Anfangszustand:
       Graues Feld mit weißer Schrift.
    */

    div[data-testid="stTextInput"] {{
        background:
            linear-gradient(
                145deg,
                #747E85,
                #5C666E
            );

        border: 2px solid #89939A;
        border-radius: 18px;

        padding: 25px 30px;

        min-height: 90px;

        transition:
            all 0.25s ease;

        box-shadow:
            0 7px 18px rgba(0,0,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.18);
    }}

    /*
       Wenn die Maus über dem Feld ist:
       Feld wird größer.
    */

    div[data-testid="stTextInput"]:hover {{
        padding: 30px 35px;

        min-height: 145px;

        transform: scale(1.015);

        background:
            linear-gradient(
                145deg,
                #7D878E,
                #626D75
            );

        box-shadow:
            0 12px 30px rgba(0,0,0,0.38),
            inset 0 1px 0 rgba(255,255,255,0.22);
    }}

    /*
       Beschriftung
    */

    div[data-testid="stTextInput"] label {{
        color: white !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        margin-bottom: 12px !important;

        transition: all 0.2s ease;
    }}

    div[data-testid="stTextInput"]:hover label {{
        font-size: 1.6rem !important;
    }}

    /*
       Echtes Eingabefeld.
       Anfangs zurückhaltend.
    */

    div[data-testid="stTextInput"] input {{
        background: transparent !important;

        color: transparent !important;

        border: 2px solid transparent !important;

        border-radius: 12px !important;

        height: 0px !important;

        padding: 0px 12px !important;

        opacity: 0;

        transition:
            all 0.25s ease;
    }}

    /*
       Beim Darüberfahren:
       Weißes Suchfeld erscheint.
    */

    div[data-testid="stTextInput"]:hover input {{
        background: white !important;

        color: #111111 !important;

        border: 2px solid #D6DCE0 !important;

        height: 52px !important;

        padding: 10px 15px !important;

        opacity: 1;

        font-size: 1.05rem !important;
    }}

    div[data-testid="stTextInput"] input::placeholder {{
        color: #777777 !important;
        opacity: 1 !important;
    }}

    /*
       Fokus des Suchfeldes
    */

    div[data-testid="stTextInput"] input:focus {{
        background: white !important;
        color: black !important;

        border: 2px solid #AAB5BC !important;

        box-shadow:
            0 0 0 3px rgba(255,255,255,0.15) !important;
    }}


    /* ========================================================
       HINWEISE
       ======================================================== */

    div[data-testid="stAlert"] {{
        background:
            linear-gradient(
                145deg,
                #647078,
                #555F66
            ) !important;

        border: 1px solid #879198 !important;

        color: white !important;

        border-radius: 14px !important;
    }}

    div[data-testid="stAlert"] p {{
        color: white !important;
    }}


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {{
        width: 100%;

        background:
            linear-gradient(
                145deg,
                #747E85,
                #59636A
            );

        color: white;

        border: 2px solid #89939A;

        border-radius: 14px;

        padding: 12px 20px;

        font-size: 1rem;
        font-weight: 700;

        transition:
            all 0.2s ease;

        box-shadow:
            0 5px 14px rgba(0,0,0,0.25);
    }}

    .stButton > button:hover {{
        transform: scale(1.025);

        background:
            linear-gradient(
                145deg,
                #879197,
                #68737B
            );

        color: white;

        border-color: white;

        box-shadow:
            0 8px 20px rgba(0,0,0,0.35);
    }}


    /* ========================================================
       DATE / SELECT / FILE UPLOADER
       ======================================================== */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background: #FFFFFF !important;
        color: #111111 !important;
        border-radius: 10px !important;
    }}

    div[data-testid="stFileUploader"] {{
        background:
            linear-gradient(
                145deg,
                #68737A,
                #58636A
            );

        border: 2px solid #89939A;
        border-radius: 16px;

        padding: 15px;

        color: white;
    }}

    div[data-testid="stFileUploader"] * {{
        color: white !important;
    }}


    /* ========================================================
       CARDS FÜR FUNDSTÜCKE
       ======================================================== */

    .item-card {{
        background:
            linear-gradient(
                145deg,
                #727C83,
                #59636B
            );

        border: 2px solid #89939A;
        border-radius: 16px;

        padding: 15px;

        margin-bottom: 15px;

        box-shadow:
            0 7px 18px rgba(0,0,0,0.30);
    }}

    .item-title {{
        color: white;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 8px;
    }}

    .item-info {{
        color: #F0F2F3;
        font-size: 0.95rem;
        margin-top: 4px;
    }}


    /* ========================================================
       TRENNLINIE
       ======================================================== */

    hr {{
        border-color: #667780;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATENBANK
# ============================================================

def init_database():
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS found_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            item_type TEXT NOT NULL,
            location TEXT NOT NULL,
            found_date TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


init_database()


# ============================================================
# LABELS LADEN
# ============================================================

def load_labels():
    if not LABELS_PATH.exists():
        return []

    labels = []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # z.B. "0 Hose" -> "Hose"
            parts = line.split(" ", 1)

            if len(parts) == 2:
                labels.append(parts[1].strip())
            else:
                labels.append(line)

    return labels


LABELS = load_labels()


# ============================================================
# KI-MODELL LADEN
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    try:
        model = tf_keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model

    except Exception as error:
        st.error(
            f"Das KI-Modell konnte nicht geladen werden: {error}"
        )

        return None


model = load_model()


# ============================================================
# BILD KLASSIFIZIEREN
# ============================================================

def classify_image(image):
    if model is None:
        return "Unbekannt", 0.0

    image = image.convert("RGB")
    image = image.resize((224, 224))

    array = np.asarray(image).astype(np.float32)

    # Teachable-Machine-Normalisierung
    array = array / 127.5 - 1.0

    array = np.expand_dims(array, axis=0)

    prediction = model.predict(
        array,
        verbose=0
    )

    probabilities = prediction[0]

    index = int(np.argmax(probabilities))

    confidence = float(probabilities[index])

    if index < len(LABELS):
        label = LABELS[index]
    else:
        label = "Unbekannt"

    return label, confidence


# ============================================================
# BILDER QUADRATISCH ZUSCHNEIDEN
# ============================================================

def square_thumbnail(image, size=500):
    image = image.convert("RGB")

    return ImageOps.fit(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS
    )


# ============================================================
# DATENBANK FUNKTIONEN
# ============================================================

def save_item(image, item_type, location, found_date):
    filename = (
        f"{uuid.uuid4().hex}.jpg"
    )

    path = UPLOAD_DIR / filename

    image.convert("RGB").save(
        path,
        "JPEG",
        quality=90
    )

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO found_items
        (image_path, item_type, location, found_date)
        VALUES (?, ?, ?, ?)
        """,
        (
            str(path),
            item_type,
            location,
            found_date
        )
    )

    conn.commit()
    conn.close()


def get_oldest_items(limit=9):
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            image_path,
            item_type,
            location,
            found_date
        FROM found_items
        ORDER BY found_date ASC, id ASC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def search_items(search_text):
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    search = f"%{search_text.lower()}%"

    cursor.execute(
        """
        SELECT
            id,
            image_path,
            item_type,
            location,
            found_date
        FROM found_items
        WHERE
            LOWER(item_type) LIKE ?
            OR LOWER(location) LIKE ?
        ORDER BY found_date ASC
        """,
        (search, search)
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


# ============================================================
# MENÜ
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Suche"


with st.sidebar:
    st.markdown(
        """
        <h2 style="color:white; margin-top:10px;">
        Fundgrube
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.button("🔎 Suche"):
        st.session_state.page = "Suche"
        st.rerun()

    if st.button("📷 Bild hochladen"):
        st.session_state.page = "Upload"
        st.rerun()

    if st.button("🕘 Älteste Fundstücke"):
        st.session_state.page = "Älteste"
        st.rerun()

    st.markdown("---")

    st.caption("Fundgrube")


# ============================================================
# SEITEN-TITEL
# ============================================================

def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-title-box">
            <div class="page-title">
                {title}
            </div>

            <div class="page-subtitle">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SEITE 1 – SUCHE
# ============================================================

if st.session_state.page == "Suche":

    page_header(
        "Fundgrube",
        "Verlorene Kleidungsstücke einfach wiederfinden."
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DAS NEUE INTERAKTIVE SUCHFELD
    # --------------------------------------------------------

    search_text = st.text_input(
        "Etwas Verloren? Suche es.",
        placeholder="Suchbegriff eingeben, z. B. Hoodie, Hose oder Schuh ...",
        key="search"
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SUCHE AUSFÜHREN
    # --------------------------------------------------------

    if search_text.strip():

        results = search_items(
            search_text.strip()
        )

        if results:

            st.subheader(
                f"Gefundene Fundstücke ({len(results)})"
            )

            columns = st.columns(3)

            for index, item in enumerate(results):

                item_id = item[0]
                image_path = item[1]
                item_type = item[2]
                location = item[3]
                found_date = item[4]

                column = columns[index % 3]

                with column:

                    st.markdown(
                        '<div class="item-card">',
                        unsafe_allow_html=True
                    )

                    if Path(image_path).exists():

                        image = Image.open(
                            image_path
                        )

                        st.image(
                            square_thumbnail(image),
                            width="stretch"
                        )

                    st.markdown(
                        f"""
                        <div class="item-title">
                            {item_type}
                        </div>

                        <div class="item-info">
                            📍 {location}
                        </div>

                        <div class="item-info">
                            📅 {found_date}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

        else:

            st.info(
                "Leider wurde noch kein passendes Fundstück gefunden."
            )

    else:

        st.info(
            "Fahre mit der Maus über das graue Feld. "
            "Dann erscheint die Suchleiste."
        )


# ============================================================
# SEITE 2 – BILD HOCHLADEN
# ============================================================

elif st.session_state.page == "Upload":

    page_header(
        "📷 Bild hochladen",
        "Lade ein Foto hoch und lass die KI das Kleidungsstück erkennen."
    )

    uploaded_file = st.file_uploader(
        "Foto aus deiner Mediathek auswählen",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.subheader("Ausgewähltes Bild")

        st.image(
            image,
            width="stretch"
        )

        st.markdown("---")

        # ----------------------------------------------------
        # KI
        # ----------------------------------------------------

        st.subheader(
            "🤖 KI-Erkennung"
        )

        with st.spinner(
            "Die KI analysiert das Bild ..."
        ):

            label, confidence = classify_image(
                image
            )

        st.success(
            f"Erkannt: {label}"
        )

        st.write(
            f"Erkennungswahrscheinlichkeit: "
            f"{confidence * 100:.1f}%"
        )

        st.progress(
            min(max(confidence, 0.0), 1.0)
        )

        st.markdown("---")

        # ----------------------------------------------------
        # FUNDSTÜCK INFORMATIONEN
        # ----------------------------------------------------

        st.subheader(
            "📍 Wo wurde das Kleidungsstück gefunden?"
        )

        location = st.text_input(
            "Fundort",
            placeholder="z. B. Sporthalle, Schulhof oder Raum 204"
        )

        found_date = st.date_input(
            "Funddatum",
            value=date.today()
        )

        # ----------------------------------------------------
        # SPEICHERN
        # ----------------------------------------------------

        if st.button(
            "💾 Fundstück speichern"
        ):

            if not location.strip():

                st.warning(
                    "Bitte gib zuerst den Fundort ein."
                )

            else:

                save_item(
                    image,
                    label,
                    location.strip(),
                    found_date.isoformat()
                )

                st.success(
                    "Das Fundstück wurde erfolgreich gespeichert!"
                )

                st.balloons()


# ============================================================
# SEITE 3 – ÄLTESTE FUNDSTÜCKE
# ============================================================

elif st.session_state.page == "Älteste":

    page_header(
        "🕘 Älteste Fundstücke",
        "Hier siehst du die neun ältesten gespeicherten Fundstücke."
    )

    items = get_oldest_items(9)

    if not items:

        st.info(
            "Es wurden bisher noch keine Fundstücke gespeichert."
        )

    else:

        columns = st.columns(3)

        for index, item in enumerate(items):

            item_id = item[0]
            image_path = item[1]
            item_type = item[2]
            location = item[3]
            found_date = item[4]

            column = columns[index % 3]

            with column:

                st.markdown(
                    '<div class="item-card">',
                    unsafe_allow_html=True
                )

                if Path(image_path).exists():

                    image = Image.open(
                        image_path
                    )

                    st.image(
                        square_thumbnail(image),
                        width="stretch"
                    )

                st.markdown(
                    f"""
                    <div class="item-title">
                        {item_type}
                    </div>

                    <div class="item-info">
                        📍 {location}
                    </div>

                    <div class="item-info">
                        📅 {found_date}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )
