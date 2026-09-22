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
# KONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FARBEN
# ============================================================

DARK_BLUE = "#102F43"
DARK_BLUE_2 = "#173F55"
METAL_GREY = "#68747C"
METAL_GREY_DARK = "#58636B"
METAL_GREY_LIGHT = "#7B858C"

WHITE = "#FFFFFF"
BLACK = "#111111"

INPUT_WHITE = "#F7F8F9"

INFO_BLUE = "#244D66"


# ============================================================
# DATEIEN / ORDNER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"

DB_PATH = BASE_DIR / "fundgrube.db"
UPLOAD_DIR = BASE_DIR / "fundgrube_uploads"

UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# CSS
#
# Wichtig:
# Hier wird KEIN sichtbares HTML erzeugt.
# Das ist ausschließlich Styling für Streamlit.
# ============================================================

st.markdown(
    f"""
    <style>

    /* --------------------------------------------------------
       GESAMTE APP
       -------------------------------------------------------- */

    .stApp {{
        background:
            linear-gradient(
                135deg,
                {DARK_BLUE} 0%,
                {DARK_BLUE_2} 45%,
                #0D293A 100%
            );
        color: {WHITE};
    }}


    /* Hauptbereich */

    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }}


    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                #53616A 0%,
                #465159 100%
            );
        border-right: 1px solid #7D878D;
    }}

    section[data-testid="stSidebar"] * {{
        color: {WHITE};
    }}

    section[data-testid="stSidebar"] button {{
        background: #69757D;
        color: white;
        border: 1px solid #7F8A91;
        border-radius: 12px;
        min-height: 48px;
        font-weight: 600;
        transition: all 0.18s ease;
    }}

    section[data-testid="stSidebar"] button:hover {{
        background: #7D888F;
        transform: scale(1.025);
        border-color: #AAB2B7;
    }}


    /* --------------------------------------------------------
       NORMALE BUTTONS
       -------------------------------------------------------- */

    .stButton > button {{
        background: {METAL_GREY};
        color: white;
        border: 1px solid #89939A;
        border-radius: 12px;
        font-weight: 700;
        min-height: 45px;
        transition: all 0.18s ease;
    }}

    .stButton > button:hover {{
        background: {METAL_GREY_LIGHT};
        color: white;
        transform: scale(1.025);
        border-color: #B2BABE;
    }}


    /* --------------------------------------------------------
       SEITENÜBERSCHRIFT
       -------------------------------------------------------- */

    .page-header {{
        background:
            linear-gradient(
                145deg,
                #707B82,
                #5E6971
            );
        border: 1px solid #89939A;
        border-radius: 20px;
        padding: 28px 32px 24px 32px;
        box-shadow:
            0 8px 22px rgba(0,0,0,0.25),
            inset 0 1px 0 rgba(255,255,255,0.10);
        margin-bottom: 28px;
    }}


    /* --------------------------------------------------------
       STREAMLIT CONTAINER MIT KEY
       -------------------------------------------------------- */

    .st-key-page_header {{
        background:
            linear-gradient(
                145deg,
                #707B82,
                #5E6971
            );
        border: 1px solid #89939A;
        border-radius: 20px;
        padding: 20px 28px;
        box-shadow:
            0 8px 22px rgba(0,0,0,0.25),
            inset 0 1px 0 rgba(255,255,255,0.10);
        margin-bottom: 30px;
    }}

    .st-key-page_header h1 {{
        color: white !important;
        font-size: 3rem !important;
        font-weight: 900 !important;
        margin-bottom: 5px !important;
    }}

    .st-key-page_header p {{
        color: white !important;
    }}


    /* --------------------------------------------------------
       HAUPTTITEL FUNDGRUBE
       -------------------------------------------------------- */

    .st-key-main_title {{
        background:
            linear-gradient(
                145deg,
                #727D84,
                #5D6870
            );
        border: 1px solid #8C969C;
        border-radius: 22px;
        padding: 28px 35px 25px 35px;
        text-align: center;
        box-shadow:
            0 10px 28px rgba(0,0,0,0.30),
            inset 0 1px 0 rgba(255,255,255,0.12);
        margin-bottom: 25px;
    }}

    .st-key-main_title h1 {{
        color: white !important;
        font-size: 4.2rem !important;
        font-weight: 950 !important;
        letter-spacing: 1px;
        margin: 0 !important;
    }}

    .st-key-main_title p {{
        color: white !important;
        font-size: 1.15rem !important;
    }}


    /* --------------------------------------------------------
       SUCHFELD
       -------------------------------------------------------- */

    .st-key-search_card {{
        background:
            linear-gradient(
                145deg,
                #737E85,
                #5C676F
            );
        border: 1px solid #8C969C;
        border-radius: 22px;
        padding: 32px 34px;
        box-shadow:
            0 9px 24px rgba(0,0,0,0.30),
            inset 0 1px 0 rgba(255,255,255,0.10);
        transition:
            transform 0.22s ease,
            box-shadow 0.22s ease,
            padding 0.22s ease;
        overflow: hidden;
        margin-top: 15px;
        margin-bottom: 25px;
    }}


    /* Das komplette Suchfeld wird größer */

    .st-key-search_card:hover {{
        transform: scale(1.025);
        box-shadow:
            0 14px 32px rgba(0,0,0,0.38),
            inset 0 1px 0 rgba(255,255,255,0.15);
        padding-top: 38px;
        padding-bottom: 38px;
    }}


    /* Überschrift des Suchfeldes */

    .st-key-search_card [data-testid="stMarkdownContainer"] p {{
        color: white !important;
        font-size: 1.55rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
    }}


    /* --------------------------------------------------------
       SUCHINPUT ZUNÄCHST VERSTECKEN
       -------------------------------------------------------- */

    .st-key-search_card [data-testid="stTextInput"] {{
        max-height: 0;
        opacity: 0;
        overflow: hidden;
        transform: translateY(-8px);
        transition:
            max-height 0.25s ease,
            opacity 0.20s ease,
            transform 0.25s ease,
            margin-top 0.25s ease;
        margin-top: 0;
    }}


    /* --------------------------------------------------------
       SUCHINPUT BEIM HOVER EINBLENDEN
       -------------------------------------------------------- */

    .st-key-search_card:hover [data-testid="stTextInput"] {{
        max-height: 100px;
        opacity: 1;
        transform: translateY(0);
        margin-top: 18px;
    }}


    /* Label des Suchinputs */

    .st-key-search_card [data-testid="stTextInput"] label {{
        color: white !important;
    }}


    /* Weiße Eingabeleiste */

    .st-key-search_card [data-testid="stTextInput"] input {{
        background: {INPUT_WHITE} !important;
        color: {BLACK} !important;
        border: 2px solid white !important;
        border-radius: 10px !important;
        font-size: 1.05rem !important;
        padding: 13px 15px !important;
    }}

    .st-key-search_card [data-testid="stTextInput"] input::placeholder {{
        color: #707070 !important;
        opacity: 1 !important;
    }}

    .st-key-search_card [data-testid="stTextInput"] input:focus {{
        border: 2px solid white !important;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.25) !important;
    }}


    /* --------------------------------------------------------
       INFOBOXEN
       -------------------------------------------------------- */

    div[data-testid="stAlert"] {{
        background: #31566E;
        border: 1px solid #59798D;
        color: white;
        border-radius: 12px;
    }}

    div[data-testid="stAlert"] p {{
        color: white !important;
    }}


    /* --------------------------------------------------------
       UPLOAD-BEREICH
       -------------------------------------------------------- */

    .st-key-upload_card {{
        background:
            linear-gradient(
                145deg,
                #707B82,
                #5C676F
            );
        border: 1px solid #8A959B;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 9px 24px rgba(0,0,0,0.28);
    }}

    .st-key-upload_card label {{
        color: white !important;
    }}


    /* --------------------------------------------------------
       FILE UPLOADER
       -------------------------------------------------------- */

    [data-testid="stFileUploader"] {{
        background: #707B82;
        border: 1px solid #90999F;
        border-radius: 15px;
        padding: 15px;
    }}

    [data-testid="stFileUploader"] section {{
        background: #707B82;
        border: 1px dashed #B9C0C4;
    }}

    [data-testid="stFileUploader"] section * {{
        color: white !important;
    }}


    /* --------------------------------------------------------
       TEXT INPUTS AUF ANDEREN SEITEN
       -------------------------------------------------------- */

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {{
        background: {INPUT_WHITE} !important;
        color: {BLACK} !important;
        border-radius: 10px !important;
        border: 2px solid #8A949A !important;
    }}

    [data-testid="stTextInput"] label,
    [data-testid="stTextArea"] label {{
        color: white !important;
        font-weight: 600 !important;
    }}


    /* --------------------------------------------------------
       DATUM
       -------------------------------------------------------- */

    [data-testid="stDateInput"] label {{
        color: white !important;
        font-weight: 600 !important;
    }}


    /* --------------------------------------------------------
       SELECTBOX
       -------------------------------------------------------- */

    [data-testid="stSelectbox"] label {{
        color: white !important;
    }}


    /* --------------------------------------------------------
       CARDS FÜR FUNDSTÜCKE
       -------------------------------------------------------- */

    .st-key-item_card {{
        background:
            linear-gradient(
                145deg,
                #707B82,
                #5C676F
            );
        border: 1px solid #8B959B;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        transition: all 0.18s ease;
    }}

    .st-key-item_card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 13px 28px rgba(0,0,0,0.35);
    }}


    /* --------------------------------------------------------
       ALLGEMEINE SCHRIFT
       -------------------------------------------------------- */

    p, label, span {{
        color: white;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATENBANK
# ============================================================

def init_database():
    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS found_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            item_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            location TEXT NOT NULL,
            found_date TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


init_database()


# ============================================================
# LABELS LADEN
# ============================================================

def load_labels():
    if not LABELS_PATH.exists():
        return ["Hose", "Schuh", "T-Shirt", "Hoodie"]

    labels = []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # Beispiel:
            # "0 Hose"
            # "1 Schuh"

            parts = line.split(maxsplit=1)

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
            "Das KI-Modell konnte nicht geladen werden."
        )
        st.code(str(error))

        return None


model = load_model()


# ============================================================
# KI-KLASSIFIKATION
# ============================================================

def classify_image(image):

    if model is None:
        return "Unbekannt", 0.0

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Teachable-Machine-Normalisierung
    image_array = image_array / 127.5 - 1.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )

    prediction = np.asarray(prediction)

    if prediction.ndim > 1:
        prediction = prediction[0]

    index = int(np.argmax(prediction))

    confidence = float(prediction[index])

    if index < len(LABELS):
        label = LABELS[index]
    else:
        label = "Unbekannt"

    return label, confidence


# ============================================================
# DATENBANK FUNKTIONEN
# ============================================================

def save_found_item(
    image_path,
    item_type,
    confidence,
    location,
    found_date
):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO found_items
        (
            image_path,
            item_type,
            confidence,
            location,
            found_date
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            str(image_path),
            item_type,
            float(confidence),
            location,
            str(found_date)
        )
    )

    connection.commit()
    connection.close()


def get_oldest_items(limit=9):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            image_path,
            item_type,
            confidence,
            location,
            found_date
        FROM found_items
        ORDER BY found_date ASC, id ASC
        LIMIT ?
        """,
        (limit,)
    )

    items = cursor.fetchall()

    connection.close()

    return items


def search_items(search_term):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    term = f"%{search_term.lower()}%"

    cursor.execute(
        """
        SELECT
            id,
            image_path,
            item_type,
            confidence,
            location,
            found_date
        FROM found_items
        WHERE
            LOWER(item_type) LIKE ?
            OR LOWER(location) LIKE ?
        ORDER BY found_date ASC
        """,
        (term, term)
    )

    results = cursor.fetchall()

    connection.close()

    return results


# ============================================================
# BILDER
# ============================================================

def make_square_image(image, size=500):

    image = image.convert("RGB")

    return ImageOps.fit(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    )


# ============================================================
# SEITEN-HEADER
# ============================================================

def page_header(title, subtitle=None):

    with st.container(key="page_header"):

        st.title(title)

        if subtitle:
            st.write(subtitle)


# ============================================================
# SEITEN-NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Suche"


with st.sidebar:

    st.markdown(
        """
        # Fundgrube
        """
    )

    st.divider()

    if st.button(
        "🔎 Suche",
        key="nav_search",
        width="stretch"
    ):
        st.session_state.page = "Suche"
        st.rerun()

    if st.button(
        "📷 Bild hochladen",
        key="nav_upload",
        width="stretch"
    ):
        st.session_state.page = "Bild hochladen"
        st.rerun()

    if st.button(
        "◷ Älteste Fundstücke",
        key="nav_oldest",
        width="stretch"
    ):
        st.session_state.page = "Älteste Fundstücke"
        st.rerun()

    st.divider()

    st.caption("Fundgrube")


# ============================================================
# SEITE 1 – SUCHE
# ============================================================

if st.session_state.page == "Suche":

    # --------------------------------------------------------
    # GROSSER TITEL
    # --------------------------------------------------------

    with st.container(key="main_title"):

        st.title("Fundgrube")

        st.write(
            "Schul-Fundstücke einfach wiederfinden"
        )


    # --------------------------------------------------------
    # HOVER-SUCHFELD
    # --------------------------------------------------------

    left, center, right = st.columns(
        [1, 6, 1]
    )

    with center:

        with st.container(key="search_card"):

            st.markdown(
                "Etwas verloren? Suche es."
            )

            search_term = st.text_input(
                "Suchbegriff",
                placeholder=(
                    "Suchbegriff eingeben, "
                    "z. B. Hoodie, Hose oder Schuh ..."
                ),
                label_visibility="collapsed",
                key="search_input"
            )


    # --------------------------------------------------------
    # SUCHERGEBNISSE
    # --------------------------------------------------------

    if search_term.strip():

        results = search_items(
            search_term.strip()
        )

        st.subheader(
            f"Suchergebnisse für „{search_term.strip()}“"
        )

        if not results:

            st.info(
                "Keine passenden Fundstücke gefunden."
            )

        else:

            for row_start in range(
                0,
                len(results),
                3
            ):

                row = results[
                    row_start:row_start + 3
                ]

                columns = st.columns(3)

                for column, item in zip(
                    columns,
                    row
                ):

                    (
                        item_id,
                        image_path,
                        item_type,
                        confidence,
                        location,
                        found_date
                    ) = item

                    with column:

                        with st.container(
                            key=f"item_card_{item_id}"
                        ):

                            image_file = Path(
                                image_path
                            )

                            if image_file.exists():

                                image = Image.open(
                                    image_file
                                )

                                image = make_square_image(
                                    image,
                                    500
                                )

                                st.image(
                                    image,
                                    width="stretch"
                                )

                            st.markdown(
                                f"**{item_type}**"
                            )

                            st.write(
                                f"📍 {location}"
                            )

                            st.write(
                                f"📅 {found_date}"
                            )


# ============================================================
# SEITE 2 – BILD HOCHLADEN
# ============================================================

elif st.session_state.page == "Bild hochladen":

    page_header(
        "📷 Bild hochladen",
        "Lade ein Foto hoch und lass die KI das Kleidungsstück erkennen."
    )


    with st.container(key="upload_card"):

        uploaded_file = st.file_uploader(
            "Foto aus deiner Mediathek auswählen",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],
            key="uploaded_image"
        )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        st.subheader(
            "Vorschau"
        )

        preview_left, preview_center, preview_right = st.columns(
            [1, 2, 1]
        )

        with preview_center:

            st.image(
                image,
                width="stretch"
            )


        # ----------------------------------------------------
        # KI ERKENNUNG
        # ----------------------------------------------------

        st.subheader(
            "🤖 KI-Erkennung"
        )

        with st.spinner(
            "Die KI analysiert das Bild ..."
        ):

            predicted_label, confidence = classify_image(
                image
            )


        st.success(
            f"Erkannt: {predicted_label}"
        )

        st.progress(
            min(max(confidence, 0.0), 1.0)
        )

        st.write(
            f"Erkennungswahrscheinlichkeit: "
            f"**{confidence * 100:.1f} %**"
        )


        # ----------------------------------------------------
        # DATEN DES FUNDSTÜCKS
        # ----------------------------------------------------

        st.subheader(
            "📍 Angaben zum Fundstück"
        )

        location = st.text_input(
            "Wo wurde das Kleidungsstück gefunden?",
            placeholder="z. B. Sporthalle, Klassenraum 203 ..."
        )

        found_date = st.date_input(
            "Wann wurde es gefunden?",
            value=date.today()
        )


        # ----------------------------------------------------
        # SPEICHERN
        # ----------------------------------------------------

        if st.button(
            "💾 Fundstück speichern",
            key="save_item",
            width="stretch"
        ):

            if not location.strip():

                st.warning(
                    "Bitte gib zuerst den Fundort ein."
                )

            else:

                unique_name = (
                    f"{uuid.uuid4().hex}.jpg"
                )

                image_path = (
                    UPLOAD_DIR / unique_name
                )

                image.save(
                    image_path,
                    format="JPEG",
                    quality=92
                )

                save_found_item(
                    image_path=image_path,
                    item_type=predicted_label,
                    confidence=confidence,
                    location=location.strip(),
                    found_date=found_date
                )

                st.success(
                    "Das Fundstück wurde erfolgreich gespeichert."
                )

                st.balloons()


# ============================================================
# SEITE 3 – ÄLTESTE FUNDSTÜCKE
# ============================================================

elif st.session_state.page == "Älteste Fundstücke":

    page_header(
        "◷ Älteste Fundstücke",
        "Die neun ältesten gespeicherten Fundstücke."
    )


    items = get_oldest_items(9)


    if not items:

        st.info(
            "Es wurden bisher noch keine Fundstücke gespeichert."
        )

    else:

        for row_start in range(
            0,
            len(items),
            3
        ):

            row = items[
                row_start:row_start + 3
            ]

            columns = st.columns(3)

            for column, item in zip(
                columns,
                row
            ):

                (
                    item_id,
                    image_path,
                    item_type,
                    confidence,
                    location,
                    found_date
                ) = item


                with column:

                    with st.container(
                        key=f"item_card_oldest_{item_id}"
                    ):

                        image_file = Path(
                            image_path
                        )

                        if image_file.exists():

                            image = Image.open(
                                image_file
                            )

                            image = make_square_image(
                                image,
                                500
                            )

                            st.image(
                                image,
                                width="stretch"
                            )

                        st.markdown(
                            f"**{item_type}**"
                        )

                        st.write(
                            f"📍 {location}"
                        )

                        st.write(
                            f"📅 {found_date}"
                        )
