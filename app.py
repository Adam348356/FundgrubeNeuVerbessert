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

HELLBLAU = "#BFE8F7"
METALL_GRAU = "#70787E"
METALL_GRAU_DUNKEL = "#555D63"
WEISS = "#FFFFFF"
SCHWARZ = "#20262A"


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    f"""
    <style>

    /* ======================================================
       HINTERGRUND
       ====================================================== */

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


    /* ======================================================
       TITEL
       ====================================================== */

    .fundgrube-title {{
        color: {SCHWARZ};
        text-align: center;

        font-size: 5.5rem;
        font-weight: 900;

        line-height: 1.05;

        margin-top: 0.5rem;
        margin-bottom: 0.2rem;

        letter-spacing: -2px;
    }}

    .fundgrube-subtitle {{
        color: {SCHWARZ};

        text-align: center;

        font-size: 1.15rem;
        font-weight: 500;

        margin-bottom: 0.5rem;
    }}


    /* ======================================================
       LUPE
       ====================================================== */

    .loupe-container {{
        display: flex;
        justify-content: center;
        align-items: center;

        width: 100%;

        margin-top: 5px;
        margin-bottom: 25px;

        transition: transform 0.25s ease;
    }}

    .loupe-container:hover {{
        transform: scale(1.10);
    }}


    /* ======================================================
       TEXTE
       ====================================================== */

    label {{
        color: {SCHWARZ} !important;
        font-weight: 700 !important;
    }}

    .stMarkdown p {{
        color: {SCHWARZ};
    }}


    /* ======================================================
       SUCHFELD
       ====================================================== */

    div[data-testid="stTextInput"] {{
        transition:
            transform 0.20s ease,
            filter 0.20s ease;
    }}

    div[data-testid="stTextInput"]:hover {{
        transform: scale(1.015);
    }}

    div[data-testid="stTextInput"]:focus-within {{
        transform: scale(1.025);
    }}

    div[data-testid="stTextInput"]
    div[data-baseweb="input"] {{
        background-color: {METALL_GRAU} !important;

        border: 2px solid {METALL_GRAU_DUNKEL} !important;

        border-radius: 13px !important;

        min-height: 52px;

        transition:
            background-color 0.2s ease,
            border-color 0.2s ease,
            box-shadow 0.2s ease;
    }}

    div[data-testid="stTextInput"]:focus-within
    div[data-baseweb="input"] {{
        background-color: {METALL_GRAU_DUNKEL} !important;

        border-color: {SCHWARZ} !important;

        box-shadow:
            0 0 0 3px rgba(255,255,255,0.30) !important;
    }}

    div[data-testid="stTextInput"] input {{
        color: {WEISS} !important;

        background-color: transparent !important;

        font-size: 1.05rem !important;

        font-weight: 500 !important;

        caret-color: white !important;
    }}

    div[data-testid="stTextInput"]
    input::placeholder {{
        color: {WEISS} !important;

        opacity: 0.85 !important;
    }}


    /* ======================================================
       DATUMSFELD
       ====================================================== */

    div[data-testid="stDateInput"] {{
        transition: transform 0.20s ease;
    }}

    div[data-testid="stDateInput"]:hover {{
        transform: scale(1.015);
    }}

    div[data-testid="stDateInput"]:focus-within {{
        transform: scale(1.02);
    }}

    div[data-testid="stDateInput"]
    div[data-baseweb="input"] {{
        background-color: {METALL_GRAU} !important;

        border: 2px solid {METALL_GRAU_DUNKEL} !important;

        border-radius: 13px !important;
    }}

    div[data-testid="stDateInput"] input {{
        color: {WEISS} !important;

        background-color: transparent !important;
    }}


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton {{
        transition: transform 0.20s ease;
    }}

    .stButton:hover {{
        transform: scale(1.02);
    }}

    .stButton > button {{
        background-color: {METALL_GRAU} !important;

        color: {WEISS} !important;

        border: 2px solid {METALL_GRAU_DUNKEL} !important;

        border-radius: 13px !important;

        min-height: 50px;

        font-weight: 700 !important;

        transition:
            background-color 0.2s ease,
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }}

    .stButton > button:hover {{
        background-color: {METALL_GRAU_DUNKEL} !important;

        color: {WEISS} !important;

        border-color: {SCHWARZ} !important;

        box-shadow:
            0 5px 12px rgba(0,0,0,0.20);
    }}


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {{
        background-color: {METALL_GRAU} !important;

        border-radius: 15px !important;

        padding: 1rem !important;

        border: 2px solid {METALL_GRAU_DUNKEL} !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }}

    [data-testid="stFileUploader"]:hover {{
        transform: scale(1.015);

        box-shadow:
            0 5px 12px rgba(0,0,0,0.15);
    }}

    [data-testid="stFileUploader"] * {{
        color: {WEISS} !important;
    }}


    /* ======================================================
       KARTEN
       ====================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {METALL_GRAU} !important;

        border-radius: 16px !important;

        border: 2px solid {METALL_GRAU_DUNKEL} !important;

        padding: 0.5rem !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        transform: scale(1.015);

        box-shadow:
            0 7px 15px rgba(0,0,0,0.20);
    }}

    [data-testid="stVerticalBlockBorderWrapper"] * {{
        color: {WEISS};
    }}


    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {{
        background-color: {METALL_GRAU_DUNKEL};
    }}

    [data-testid="stSidebar"] * {{
        color: {WEISS} !important;
    }}


    /* ======================================================
       TRENNLINIEN
       ====================================================== */

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
# LABELS LADEN
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
# QUADRATISCHES BILD
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
# LUPE
# ============================================================

@st.cache_data
def create_magnifying_glass():

    width = 700
    height = 330

    image = Image.new(
        "RGBA",
        (width, height),
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(image)

    center_x = 300
    center_y = 130

    radius_outer = 92
    radius_inner = 72

    # Äußerer Kreis
    draw.ellipse(
        (
            center_x - radius_outer,
            center_y - radius_outer,
            center_x + radius_outer,
            center_y + radius_outer,
        ),
        outline=SCHWARZ,
        width=10,
    )

    # Innerer Kreis
    draw.ellipse(
        (
            center_x - radius_inner,
            center_y - radius_inner,
            center_x + radius_inner,
            center_y + radius_inner,
        ),
        outline=SCHWARZ,
        width=7,
    )

    # Griff
    draw.line(
        [
            (
                center_x + 65,
                center_y + 65,
            ),
            (
                center_x + 155,
                center_y + 155,
            ),
        ],
        fill=SCHWARZ,
        width=24,
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
# HAUPTTITEL
# ============================================================

def show_main_title():

    st.markdown(
        '<div class="fundgrube-title">Fundgrube</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="fundgrube-subtitle">'
        'Schul-Fundstücke einfach wiederfinden'
        '</div>',
        unsafe_allow_html=True,
    )

    # Lupe mittig
    st.markdown(
        '<div class="loupe-container">',
        unsafe_allow_html=True,
    )

    st.image(
        create_magnifying_glass(),
        width="content",
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# SEITENTITEL
# ============================================================

def page_header(
    title,
    description=None,
):

    if description:

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
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TEXT NORMALISIEREN
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zäöüß0-9\- ]",
        " ",
        text,
    )

    return [
        word.strip()
        for word in text.split()
        if word.strip()
    ]


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
            {len(results)}
            {"Fundstück gefunden"
             if len(results) == 1
             else "Fundstücke gefunden"}
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
                            width="stretch",
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
            width="stretch",
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
            </div>
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

                    <span style="
                        font-size: 1.5rem;
                    ">
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
                            width="stretch",
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
