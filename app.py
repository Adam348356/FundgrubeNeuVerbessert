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
# GRUNDEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

DB_PATH = "fundgrube.db"
UPLOAD_DIR = Path("fundgrube_uploads")

UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# FARBEN
# ============================================================

HELLBLAU = "#BFE8F7"
METALL_GRAU = "#6F777D"
METALL_GRAU_DUNKEL = "#596167"
WEISS = "#FFFFFF"
SCHWARZ = "#22272B"
HELL_GRAU = "#E8EDF0"


# ============================================================
# STREAMLIT-DESIGN
# ============================================================

st.markdown(
    f"""
    <style>

    /* -------------------------------------------------------
       GESAMTE APP
       ------------------------------------------------------- */

    .stApp {{
        background-color: {HELLBLAU};
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {HELLBLAU};
    }}

    [data-testid="stHeader"] {{
        background-color: {HELLBLAU};
    }}

    .main .block-container {{
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}


    /* -------------------------------------------------------
       SIDEBAR
       ------------------------------------------------------- */

    [data-testid="stSidebar"] {{
        background-color: {METALL_GRAU_DUNKEL};
    }}

    [data-testid="stSidebar"] * {{
        color: {WEISS} !important;
    }}


    /* -------------------------------------------------------
       TITEL
       ------------------------------------------------------- */

    .fundgrube-title {{
        color: {SCHWARZ};
        text-align: center;
        font-size: 4.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }}

    .fundgrube-subtitle {{
        color: {SCHWARZ};
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }}


    /* -------------------------------------------------------
       NORMALE TEXTE
       ------------------------------------------------------- */

    label {{
        color: {SCHWARZ} !important;
        font-weight: 600 !important;
    }}

    .stMarkdown p {{
        color: {SCHWARZ};
    }}


    /* -------------------------------------------------------
       TEXTFELDER
       ------------------------------------------------------- */

    div[data-baseweb="input"] {{
        background-color: {METALL_GRAU} !important;
        border-radius: 12px !important;
        border: 2px solid {METALL_GRAU_DUNKEL} !important;
    }}

    div[data-baseweb="input"] input {{
        color: {WEISS} !important;
        background-color: transparent !important;
    }}

    div[data-baseweb="input"] input::placeholder {{
        color: #E8E8E8 !important;
        opacity: 1 !important;
    }}


    /* -------------------------------------------------------
       TEXTAREA
       ------------------------------------------------------- */

    textarea {{
        background-color: {METALL_GRAU} !important;
        color: {WEISS} !important;
        border: 2px solid {METALL_GRAU_DUNKEL} !important;
        border-radius: 12px !important;
    }}


    /* -------------------------------------------------------
       DATE INPUT
       ------------------------------------------------------- */

    div[data-baseweb="select"] > div {{
        background-color: {METALL_GRAU} !important;
        color: {WEISS} !important;
        border: 2px solid {METALL_GRAU_DUNKEL} !important;
        border-radius: 12px !important;
    }}

    div[data-baseweb="select"] * {{
        color: {WEISS} !important;
    }}


    /* -------------------------------------------------------
       BUTTONS
       ------------------------------------------------------- */

    .stButton > button {{
        background-color: {METALL_GRAU} !important;
        color: {WEISS} !important;
        border: 2px solid {METALL_GRAU_DUNKEL} !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        min-height: 48px;
    }}

    .stButton > button:hover {{
        background-color: {METALL_GRAU_DUNKEL} !important;
        color: {WEISS} !important;
        border-color: {SCHWARZ} !important;
    }}


    /* -------------------------------------------------------
       FILE UPLOADER
       ------------------------------------------------------- */

    [data-testid="stFileUploader"] {{
        background-color: {METALL_GRAU} !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        border: 2px solid {METALL_GRAU_DUNKEL} !important;
    }}

    [data-testid="stFileUploader"] * {{
        color: {WEISS} !important;
    }}


    /* -------------------------------------------------------
       INFO / SUCCESS / WARNING
       ------------------------------------------------------- */

    [data-testid="stAlert"] {{
        border-radius: 12px !important;
    }}


    /* -------------------------------------------------------
       KARTEN
       ------------------------------------------------------- */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {METALL_GRAU} !important;
        border-radius: 16px !important;
        border: 2px solid {METALL_GRAU_DUNKEL} !important;
        padding: 0.5rem !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] * {{
        color: {WEISS};
    }}


    /* -------------------------------------------------------
       TRENNLINIEN
       ------------------------------------------------------- */

    hr {{
        border-color: {METALL_GRAU} !important;
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
                    label = match.group(1).strip()
                else:
                    label = line

                labels.append(label)

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
            f"Die Datei '{MODEL_PATH}' wurde nicht gefunden."
        )

    model = tf_keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    return model


# ============================================================
# BILD KLASSIFIZIEREN
# ============================================================

def classify_image(image):

    model = load_model()

    image = image.convert("RGB")

    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    image_array = np.asarray(
        image
    ).astype(np.float32)

    image_array = (
        image_array / 127.5
    ) - 1.0

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    prediction = model.predict(
        image_array,
        verbose=0,
    )

    probabilities = prediction[0]

    class_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_index]
    )

    if class_index < len(LABELS):
        label = LABELS[class_index]
    else:
        label = f"Klasse {class_index}"

    return label, confidence


# ============================================================
# QUADRATISCHE BILDER
# ============================================================

def make_square_image(
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
# LUPE ERZEUGEN
# ============================================================

@st.cache_data
def create_magnifying_glass():

    width = 420
    height = 330

    image = Image.new(
        "RGBA",
        (width, height),
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(image)

    # Kreis
    circle_box = (
        70,
        25,
        285,
        240,
    )

    # äußerer Kreis
    draw.ellipse(
        circle_box,
        outline=SCHWARZ,
        width=12,
    )

    # innerer Kreis
    inner_box = (
        91,
        46,
        264,
        219,
    )

    draw.ellipse(
        inner_box,
        outline=SCHWARZ,
        width=8,
    )

    # Griff
    draw.line(
        [
            (245, 215),
            (340, 300),
        ],
        fill=SCHWARZ,
        width=25,
    )

    return image


# ============================================================
# NAVIGATION
# ============================================================

def show_navigation():

    with st.sidebar:

        st.markdown(
            "## Fundgrube"
        )

        st.divider()

        if st.button(
            "🔎 Suche",
            use_container_width=True,
        ):

            st.session_state.page = "Suche"
            st.rerun()

        if st.button(
            "📷 Bild hochladen",
            use_container_width=True,
        ):

            st.session_state.page = "Hochladen"
            st.rerun()

        if st.button(
            "🕐 Älteste Fundstücke",
            use_container_width=True,
        ):

            st.session_state.page = "Älteste"
            st.rerun()

        st.divider()

        st.caption(
            "Fundgrube"
        )


# ============================================================
# HAUPTTITEL
# ============================================================

def show_main_title():

    st.markdown(
        '<div class="fundgrube-title">Fundgrube</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="fundgrube-subtitle">'
        "Schul-Fundstücke einfach wiederfinden"
        "</div>",
        unsafe_allow_html=True,
    )

    # LUPE DIREKT UNTER DEM TITEL
    left, center, right = st.columns(
        [1, 2, 1]
    )

    with center:

        st.image(
            create_magnifying_glass(),
            width=190,
        )


# ============================================================
# SEITENTITEL
# ============================================================

def page_header(
    title,
    description=None,
):

    st.markdown(
        f"""
        <div style="
            background-color: {METALL_GRAU};
            padding: 18px 24px;
            border-radius: 16px;
            margin-top: 15px;
            margin-bottom: 25px;
        ">
            <div style="
                color: white;
                font-size: 2rem;
                font-weight: 700;
            ">
                {title}
            </div>
        """,
        unsafe_allow_html=True,
    )

    if description:

        st.markdown(
            f"""
            <div style="
                color: white;
                font-size: 1rem;
                margin-top: 5px;
            ">
                {description}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# SUCHBEGRIFFE
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zäöüß0-9\- ]",
        " ",
        text,
    )

    words = [
        word.strip()
        for word in text.split()
        if word.strip()
    ]

    return words


# ============================================================
# SUCHE
# ============================================================

def search_page():

    show_navigation()

    show_main_title()

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    search_text = st.text_input(
        "Was hast du verloren?",
        placeholder=(
            "z. B. Hoodie, schwarze Hose, Schuh ..."
        ),
    )

    if not search_text.strip():

        st.info(
            "Gib oben ein Kleidungsstück ein, "
            "um passende Fundstücke zu finden."
        )

        return

    words = normalize_text(
        search_text
    )

    if not words:
        return

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

    all_items = cursor.fetchall()

    connection.close()

    results = []

    for item in all_items:

        (
            item_id,
            item_type,
            location,
            found_date,
            image_path,
        ) = item

        searchable_text = (
            f"{item_type} {location}"
        ).lower()

        if any(
            word in searchable_text
            for word in words
        ):

            results.append(item)

    if not results:

        st.warning(
            "Kein passendes Fundstück gefunden."
        )

        return

    st.markdown(
        f"""
        <div style="
            color: {SCHWARZ};
            font-size: 1.4rem;
            font-weight: 700;
            margin: 20px 0;
        ">
            {len(results)} Fundstück
            {"gefunden" if len(results) == 1 else "e gefunden"}
        </div>
        """,
        unsafe_allow_html=True,
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

                if os.path.exists(image_path):

                    try:

                        image = Image.open(
                            image_path
                        )

                        st.image(
                            make_square_image(
                                image
                            ),
                            use_container_width=True,
                        )

                    except Exception:

                        st.warning(
                            "Bild konnte nicht geladen werden."
                        )

                st.markdown(
                    f"### {item_type}"
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

    show_navigation()

    page_header(
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
            use_container_width=True,
        )

    with result_column:

        st.markdown(
            f"""
            <div style="
                background-color: {METALL_GRAU};
                color: white;
                padding: 20px;
                border-radius: 16px;
            ">
                <h3 style="color:white;">
                    🤖 KI-Erkennung
                </h3>
            """,
            unsafe_allow_html=True,
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

            confidence_percent = (
                confidence * 100
            )

            st.markdown(
                f"""
                <div style="
                    background-color: {METALL_GRAU_DUNKEL};
                    color: white;
                    padding: 15px;
                    border-radius: 12px;
                    margin-top: 10px;
                ">
                    <b>Erkannt:</b><br>
                    <span style="font-size: 1.5rem;">
                        {label}
                    </span>
                    <br><br>
                    <b>Sicherheit:</b>
                    {confidence_percent:.1f} %
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confidence < 0.50:

                st.warning(
                    "Die KI ist bei dieser Erkennung "
                    "nicht besonders sicher."
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

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    page_header(
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
        use_container_width=True,
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

    show_navigation()

    page_header(
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

                if os.path.exists(image_path):

                    try:

                        image = Image.open(
                            image_path
                        )

                        st.image(
                            make_square_image(
                                image
                            ),
                            use_container_width=True,
                        )

                    except Exception:

                        st.warning(
                            "Bild konnte nicht geladen werden."
                        )

                st.markdown(
                    f"### {item_type}"
                )

                st.write(
                    f"📍 Fundort: {location}"
                )

                st.write(
                    f"📅 Gefunden: {found_date}"
                )


# ============================================================
# APP STARTEN
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
