import os
import re
import sqlite3
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import streamlit as st
import tf_keras

from PIL import Image, ImageOps, ImageDraw


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DATEIEN
# ============================================================

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

DB_PATH = "fundgrube.db"
UPLOAD_DIR = Path("fundgrube_uploads")

UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# FARBEN
# ============================================================

# Dunkleres Blau
BACKGROUND_BLUE = "#79B8D0"

# Metallisches Grau
METAL_GREY = "#687279"

# Dunkleres metallisches Grau
DARK_GREY = "#4F585E"

# Sehr dunkles Grau
DARK_TEXT = "#20272C"

# Weiß
WHITE = "#FFFFFF"


# ============================================================
# DESIGN
#
# WICHTIG:
# Hier wird nur CSS für das Aussehen verwendet.
# Es werden KEINE HTML-DIVs für sichtbare Inhalte verwendet.
# ============================================================

st.markdown(
    f"""
    <style>

    /* ======================================================
       GESAMTE APP
       ====================================================== */

    .stApp {{
        background-color: {BACKGROUND_BLUE};
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {BACKGROUND_BLUE};
    }}

    [data-testid="stHeader"] {{
        background-color: {BACKGROUND_BLUE};
    }}

    .main .block-container {{
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}


    /* ======================================================
       TITEL
       ====================================================== */

    .stMarkdown h1,
    h1 {{
        color: {WHITE} !important;

        font-size: 5.5rem !important;

        font-weight: 900 !important;

        text-align: center !important;

        line-height: 1.0 !important;

        margin-top: 0.2rem !important;
        margin-bottom: 0.3rem !important;
    }}

    .stMarkdown h2,
    h2 {{
        color: {WHITE} !important;

        font-weight: 800 !important;
    }}

    .stMarkdown h3,
    h3 {{
        color: {WHITE} !important;

        font-weight: 750 !important;
    }}


    /* ======================================================
       GRAUE CONTAINER
       ====================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {METAL_GREY} !important;

        border: 2px solid {DARK_GREY} !important;

        border-radius: 18px !important;

        padding: 1rem !important;

        transition:
            transform 0.20s ease,
            box-shadow 0.20s ease;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        transform: scale(1.01);

        box-shadow:
            0 7px 18px rgba(0, 0, 0, 0.20);
    }}

    [data-testid="stVerticalBlockBorderWrapper"] p {{
        color: {WHITE} !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] label {{
        color: {WHITE} !important;
    }}


    /* ======================================================
       NORMALE TEXTE
       ====================================================== */

    .stMarkdown p {{
        color: {DARK_TEXT};
    }}

    .stCaption {{
        color: {WHITE} !important;
    }}

    label {{
        color: {DARK_TEXT} !important;
        font-weight: 700 !important;
    }}


    /* ======================================================
       TEXT INPUT
       ====================================================== */

    div[data-testid="stTextInput"] {{
        transition:
            transform 0.20s ease,
            box-shadow 0.20s ease;
    }}

    div[data-testid="stTextInput"]:hover {{
        transform: scale(1.015);
    }}

    div[data-testid="stTextInput"]:focus-within {{
        transform: scale(1.025);
    }}

    div[data-testid="stTextInput"]
    div[data-baseweb="input"] {{
        background-color: {METAL_GREY} !important;

        border: 2px solid {DARK_GREY} !important;

        border-radius: 13px !important;

        min-height: 54px !important;

        transition:
            background-color 0.20s ease,
            border-color 0.20s ease,
            box-shadow 0.20s ease;
    }}

    div[data-testid="stTextInput"]:focus-within
    div[data-baseweb="input"] {{
        background-color: {DARK_GREY} !important;

        border-color: {WHITE} !important;

        box-shadow:
            0 0 0 3px rgba(255,255,255,0.25) !important;
    }}

    div[data-testid="stTextInput"] input {{
        color: {WHITE} !important;

        background-color: transparent !important;

        font-size: 1.05rem !important;

        font-weight: 500 !important;

        caret-color: {WHITE} !important;
    }}

    div[data-testid="stTextInput"]
    input::placeholder {{
        color: {WHITE} !important;

        opacity: 0.85 !important;
    }}


    /* ======================================================
       DATE INPUT
       ====================================================== */

    div[data-testid="stDateInput"] {{
        transition:
            transform 0.20s ease;
    }}

    div[data-testid="stDateInput"]:hover {{
        transform: scale(1.015);
    }}

    div[data-testid="stDateInput"]:focus-within {{
        transform: scale(1.025);
    }}

    div[data-testid="stDateInput"]
    div[data-baseweb="input"] {{
        background-color: {METAL_GREY} !important;

        border: 2px solid {DARK_GREY} !important;

        border-radius: 13px !important;
    }}

    div[data-testid="stDateInput"] input {{
        color: {WHITE} !important;

        background-color: transparent !important;
    }}


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton {{
        transition:
            transform 0.20s ease;
    }}

    .stButton:hover {{
        transform: scale(1.025);
    }}

    .stButton > button {{
        background-color: {METAL_GREY} !important;

        color: {WHITE} !important;

        border: 2px solid {DARK_GREY} !important;

        border-radius: 13px !important;

        min-height: 50px !important;

        font-weight: 700 !important;

        transition:
            background-color 0.20s ease,
            box-shadow 0.20s ease;
    }}

    .stButton > button:hover {{
        background-color: {DARK_GREY} !important;

        color: {WHITE} !important;

        border-color: {WHITE} !important;

        box-shadow:
            0 5px 14px rgba(0,0,0,0.22);
    }}


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {{
        background-color: {METAL_GREY} !important;

        border: 2px solid {DARK_GREY} !important;

        border-radius: 16px !important;

        padding: 1rem !important;

        transition:
            transform 0.20s ease,
            box-shadow 0.20s ease;
    }}

    [data-testid="stFileUploader"]:hover {{
        transform: scale(1.015);

        box-shadow:
            0 6px 15px rgba(0,0,0,0.20);
    }}

    [data-testid="stFileUploader"] * {{
        color: {WHITE} !important;
    }}


    /* ======================================================
       INFO / WARNING / SUCCESS
       ====================================================== */

    [data-testid="stAlert"] {{
        border-radius: 13px !important;

        border: 2px solid {DARK_GREY} !important;

        background-color: {METAL_GREY} !important;
    }}

    [data-testid="stAlert"] * {{
        color: {WHITE} !important;
    }}


    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {{
        background-color: {DARK_GREY};
    }}

    [data-testid="stSidebar"] * {{
        color: {WHITE} !important;
    }}

    [data-testid="stSidebar"] .stButton > button {{
        background-color: {METAL_GREY} !important;

        border-color: {METAL_GREY} !important;

        color: {WHITE} !important;
    }}

    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {BACKGROUND_BLUE} !important;

        color: {DARK_TEXT} !important;
    }}


    /* ======================================================
       TRENNLINIEN
       ====================================================== */

    hr {{
        border-color: {DARK_GREY} !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Suche"


# ============================================================
# LABELS
# ============================================================

def load_labels():

    fallback = [
        "Hose",
        "Schuh",
        "T-Shirt",
        "Hoodie",
    ]

    if not os.path.exists(LABELS_PATH):
        return fallback

    labels = []

    try:

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                match = re.match(
                    r"^\s*\d+\s+(.*)$",
                    line,
                )

                if match:
                    labels.append(
                        match.group(1).strip()
                    )
                else:
                    labels.append(line)

    except Exception:

        return fallback

    if not labels:
        return fallback

    return labels


LABELS = load_labels()


# ============================================================
# DATENBANK
# ============================================================

def get_connection():

    return sqlite3.connect(
        DB_PATH,
        timeout=10,
    )


def init_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL,
            location TEXT NOT NULL,
            found_date TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
        """
    )

    connection.commit()

    connection.close()


init_database()


# ============================================================
# KI-MODELL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "keras_model.h5 wurde nicht gefunden."
        )

    return tf_keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


# ============================================================
# KI-KLASSIFIKATION
# ============================================================

def classify_image(image):

    model = load_model()

    image = image.convert("RGB")

    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(
        image
    ).astype(np.float32)

    array = (
        array / 127.5
    ) - 1.0

    array = np.expand_dims(
        array,
        axis=0,
    )

    prediction = model.predict(
        array,
        verbose=0,
    )

    probabilities = prediction[0]

    index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[index]
    )

    if index < len(LABELS):
        label = LABELS[index]
    else:
        label = f"Klasse {index}"

    return label, confidence


# ============================================================
# QUADRATISCHES BILD
# ============================================================

def square_image(
    image,
    size=500,
):

    image = image.convert("RGB")

    return ImageOps.fit(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


# ============================================================
# LUPE ZEICHNEN
# ============================================================

@st.cache_data
def create_loupe():

    width = 500
    height = 320

    image = Image.new(
        "RGBA",
        (width, height),
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(image)

    center_x = 215
    center_y = 125

    outer_radius = 82
    inner_radius = 65

    # äußerer Ring
    draw.ellipse(
        (
            center_x - outer_radius,
            center_y - outer_radius,
            center_x + outer_radius,
            center_y + outer_radius,
        ),
        outline=DARK_TEXT,
        width=9,
    )

    # innerer Ring
    draw.ellipse(
        (
            center_x - inner_radius,
            center_y - inner_radius,
            center_x + inner_radius,
            center_y + inner_radius,
        ),
        outline=DARK_TEXT,
        width=7,
    )

    # Griff
    draw.line(
        [
            (
                center_x + 58,
                center_y + 58,
            ),
            (
                center_x + 145,
                center_y + 145,
            ),
        ],
        fill=DARK_TEXT,
        width=23,
    )

    return image


# ============================================================
# NAVIGATION
# ============================================================

def navigation():

    with st.sidebar:

        st.markdown(
            "## Fundgrube"
        )

        st.divider()

        if st.button(
            "🔎 Suche",
            width="stretch",
        ):

            st.session_state.page = "Suche"

            st.rerun()

        if st.button(
            "📷 Bild hochladen",
            width="stretch",
        ):

            st.session_state.page = "Hochladen"

            st.rerun()

        if st.button(
            "🕐 Älteste Fundstücke",
            width="stretch",
        ):

            st.session_state.page = "Älteste"

            st.rerun()

        st.divider()

        st.caption(
            "Fundgrube"
        )


# ============================================================
# GROSSER TITEL
# ============================================================

def main_title():

    # Der Titel liegt jetzt selbst in einem grauen,
    # gerahmten Streamlit-Container.

    with st.container(border=True):

        st.title(
            "Fundgrube"
        )

        st.markdown(
            "Schul-Fundstücke einfach wiederfinden"
        )


# ============================================================
# LUPE
# ============================================================

def centered_loupe():

    left, middle, right = st.columns(
        [1, 1, 1]
    )

    with middle:

        st.image(
            create_loupe(),
            width="stretch",
        )


# ============================================================
# SEITENTITEL
# ============================================================

def section_title(
    title,
    description=None,
):

    with st.container(border=True):

        st.subheader(
            title
        )

        if description:

            st.write(
                description
            )


# ============================================================
# SUCHE
# ============================================================

def search_page():

    navigation()

    main_title()

    centered_loupe()

    st.write("")

    search = st.text_input(
        "Was hast du verloren?",
        placeholder=(
            "z. B. Hoodie, schwarze Hose, Schuh ..."
        ),
    )

    if not search.strip():

        st.info(
            "Gib oben ein Kleidungsstück ein, "
            "um passende Fundstücke zu finden."
        )

        return

    words = [
        word.lower()
        for word in search.split()
        if word.strip()
    ]

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            item_type,
            location,
            found_date,
            image_path
        FROM items
        ORDER BY
            found_date ASC,
            id ASC
        """
    )

    items = cursor.fetchall()

    connection.close()

    results = []

    for item in items:

        (
            item_id,
            item_type,
            location,
            found_date,
            image_path,
        ) = item

        text = (
            f"{item_type} {location}"
        ).lower()

        if any(
            word in text
            for word in words
        ):

            results.append(item)

    if not results:

        st.warning(
            "Kein passendes Fundstück gefunden."
        )

        return

    st.subheader(
        f"{len(results)} Fundstück"
        + (
            "" if len(results) == 1
            else "e"
        )
        + " gefunden"
    )

    columns = st.columns(3)

    for index, item in enumerate(results):

        (
            item_id,
            item_type,
            location,
            found_date,
            image_path,
        ) = item

        with columns[index % 3]:

            with st.container(
                border=True
            ):

                if os.path.exists(
                    image_path
                ):

                    try:

                        image = Image.open(
                            image_path
                        )

                        st.image(
                            square_image(
                                image
                            ),
                            width="stretch",
                        )

                    except Exception:

                        st.warning(
                            "Bild konnte nicht geladen werden."
                        )

                st.subheader(
                    item_type
                )

                st.write(
                    f"📍 Fundort: {location}"
                )

                st.write(
                    f"📅 Gefunden: {found_date}"
                )


# ============================================================
# HOCHLADEN
# ============================================================

def upload_page():

    navigation()

    section_title(
        "📷 Bild hochladen",
        "Lade ein Foto hoch und lass es von der KI erkennen.",
    )

    uploaded_file = st.file_uploader(
        "Foto aus deiner Mediathek auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded_file is None:

        st.info(
            "Wähle ein Bild aus, um zu beginnen."
        )

        return

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Das Bild konnte nicht geöffnet werden."
        )

        return

    st.subheader(
        "Hochgeladenes Bild"
    )

    image_column, result_column = st.columns(
        [2, 1]
    )

    with image_column:

        st.image(
            image,
            width="stretch",
        )

    with result_column:

        with st.container(
            border=True
        ):

            st.subheader(
                "🤖 KI-Erkennung"
            )

            progress = st.progress(
                0
            )

            status = st.empty()

            try:

                status.write(
                    "Bild wird vorbereitet ..."
                )

                progress.progress(
                    25
                )

                status.write(
                    "KI analysiert das Bild ..."
                )

                progress.progress(
                    60
                )

                label, confidence = classify_image(
                    image
                )

                progress.progress(
                    100
                )

                status.empty()

                st.write(
                    f"Erkannt: **{label}**"
                )

                st.write(
                    f"Sicherheit: **{confidence * 100:.1f} %**"
                )

                if confidence < 0.50:

                    st.warning(
                        "Die KI ist bei dieser "
                        "Erkennung nicht besonders sicher."
                    )

            except Exception as error:

                progress.empty()

                status.empty()

                st.error(
                    "Bei der KI-Erkennung ist ein Fehler aufgetreten."
                )

                st.exception(
                    error
                )

                return

    st.write("")

    section_title(
        "📍 Fundinformationen"
    )

    location = st.text_input(
        "Wo wurde der Gegenstand gefunden?",
        placeholder=(
            "z. B. Sporthalle, Aula oder Raum 203"
        ),
    )

    found_date = st.date_input(
        "Wann wurde der Gegenstand gefunden?",
        value=date.today(),
    )

    save = st.button(
        "💾 Fundstück speichern",
        type="primary",
        width="stretch",
    )

    if not save:
        return

    if not location.strip():

        st.warning(
            "Bitte gib zuerst den Fundort ein."
        )

        return

    filename = (
        uuid.uuid4().hex
        + ".jpg"
    )

    image_path = (
        UPLOAD_DIR
        / filename
    )

    try:

        image.save(
            image_path,
            format="JPEG",
            quality=90,
        )

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO items
            (
                item_type,
                location,
                found_date,
                image_path
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                label,
                location.strip(),
                found_date.isoformat(),
                str(image_path),
            ),
        )

        connection.commit()

        connection.close()

        st.success(
            "✅ Das Fundstück wurde erfolgreich gespeichert!"
        )

    except Exception as error:

        st.error(
            "Das Fundstück konnte nicht gespeichert werden."
        )

        st.exception(
            error
        )


# ============================================================
# ÄLTESTE FUNDSTÜCKE
# ============================================================

def oldest_page():

    navigation()

    section_title(
        "🕐 Älteste Fundstücke",
        "Die neun ältesten gespeicherten Fundstücke.",
    )

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            item_type,
            location,
            found_date,
            image_path
        FROM items
        ORDER BY
            found_date ASC,
            id ASC
        LIMIT 9
        """
    )

    items = cursor.fetchall()

    connection.close()

    if not items:

        st.info(
            "Es wurden bisher noch keine Fundstücke gespeichert."
        )

        return

    columns = st.columns(3)

    for index, item in enumerate(items):

        (
            item_id,
            item_type,
            location,
            found_date,
            image_path,
        ) = item

        with columns[index % 3]:

            with st.container(
                border=True
            ):

                if os.path.exists(
                    image_path
                ):

                    try:

                        image = Image.open(
                            image_path
                        )

                        st.image(
                            square_image(
                                image
                            ),
                            width="stretch",
                        )

                    except Exception:

                        st.warning(
                            "Bild konnte nicht geladen werden."
                        )

                st.subheader(
                    item_type
                )

                st.write(
                    f"📍 Fundort: {location}"
                )

                st.write(
                    f"📅 Gefunden: {found_date}"
                )


# ============================================================
# APP
# ============================================================

if st.session_state.page == "Suche":

    search_page()

elif st.session_state.page == "Hochladen":

    upload_page()

elif st.session_state.page == "Älteste":

    oldest_page()

else:

    st.session_state.page = "Suche"

    st.rerun()
