import sqlite3
import uuid
from pathlib import Path
from datetime import date, datetime

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
import tf_keras


# ============================================================
# GRUNDEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    layout="wide"
)


# ============================================================
# DATEIPFADE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
DB_PATH = BASE_DIR / "fundgrube.db"

UPLOAD_DIR = BASE_DIR / "fundgrube_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GESAMTE APP
    -------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(110, 140, 155, 0.22),
                transparent 32%
            ),
            radial-gradient(
                circle at 85% 90%,
                rgba(70, 100, 120, 0.20),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #081e2d 0%,
                #0d2c40 45%,
                #0a2232 100%
            );

        color: white;
    }


    /* --------------------------------------------------------
       HAUPTBEREICH BREITER MACHEN
    -------------------------------------------------------- */

    .block-container {
        max-width: 1500px !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #59636a 0%,
                #444e55 50%,
                #343e45 100%
            ) !important;

        border-right: 1px solid rgba(255,255,255,0.15);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        min-height: 58px !important;

        font-size: 1.15rem !important;
        font-weight: 700 !important;

        border-radius: 12px !important;
        border: 1px solid #aeb7bc !important;

        background:
            linear-gradient(
                145deg,
                #707a80,
                #505a61
            ) !important;

        color: white !important;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.20),
            0 5px 12px rgba(0,0,0,0.20) !important;

        transition:
            transform 0.15s ease,
            filter 0.15s ease !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        filter: brightness(1.10) !important;
    }


    /* --------------------------------------------------------
       ALLE GRAUEN CONTAINER / KARTEN
    -------------------------------------------------------- */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(
                145deg,
                #8d969b 0%,
                #747e84 45%,
                #636d73 100%
            ) !important;

        border: 2px solid #aeb7bb !important;

        border-radius: 20px !important;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.22),
            inset 0 -2px 5px rgba(0,0,0,0.12),
            0 12px 30px rgba(0,0,0,0.28) !important;

        padding: 1.8rem !important;

        min-height: 120px;
    }


    /* --------------------------------------------------------
       TITELKARTEN
    -------------------------------------------------------- */

    .st-key-page_title {
        width: 100% !important;

        min-height: 145px !important;

        display: flex !important;
        align-items: center !important;

        margin-bottom: 2rem !important;

        background:
            linear-gradient(
                145deg,
                #929b9f,
                #737d83
            ) !important;

        border: 2px solid #b6bdc1 !important;

        border-radius: 22px !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.25),
            0 14px 35px rgba(0,0,0,0.30) !important;
    }

    .st-key-page_title h1 {
        color: white !important;

        font-size: 3.4rem !important;
        line-height: 1.1 !important;

        font-weight: 900 !important;

        margin: 0 !important;
    }

    .st-key-page_title h2,
    .st-key-page_title h3,
    .st-key-page_title p {
        color: white !important;
    }


    /* --------------------------------------------------------
       NORMALE ÜBERSCHRIFTEN
    -------------------------------------------------------- */

    h1, h2, h3 {
        color: white !important;
    }

    h2 {
        font-size: 2.2rem !important;
        font-weight: 850 !important;
    }

    h3 {
        font-size: 1.7rem !important;
        font-weight: 800 !important;
    }


    /* --------------------------------------------------------
       SUCHE - GROSSE GRAUE KARTE
    -------------------------------------------------------- */

    .st-key-search_card {
        width: 100% !important;

        min-height: 330px !important;

        padding: 2.8rem !important;

        background:
            linear-gradient(
                145deg,
                #929b9f 0%,
                #737d83 50%,
                #646e74 100%
            ) !important;

        border: 2px solid #b5bdc1 !important;

        border-radius: 24px !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.25),
            inset 0 -3px 8px rgba(0,0,0,0.12),
            0 15px 35px rgba(0,0,0,0.30) !important;

        transition:
            transform 0.20s ease,
            box-shadow 0.20s ease !important;
    }

    .st-key-search_card:hover {
        transform: scale(1.025) !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.30),
            0 20px 45px rgba(0,0,0,0.38) !important;
    }

    .st-key-search_card h2 {
        color: white !important;

        font-size: 2.8rem !important;

        font-weight: 900 !important;

        margin-bottom: 1rem !important;
    }

    .st-key-search_card p {
        color: white !important;

        font-size: 1.25rem !important;
    }


    /* --------------------------------------------------------
       SUCHFELD
       Standardmäßig versteckt
    -------------------------------------------------------- */

    .st-key-search_card [data-testid="stTextInput"] {
        max-height: 0 !important;

        opacity: 0 !important;

        overflow: hidden !important;

        margin-top: 0 !important;

        transition:
            max-height 0.25s ease,
            opacity 0.20s ease,
            margin-top 0.25s ease !important;
    }

    .st-key-search_card:hover [data-testid="stTextInput"] {
        max-height: 120px !important;

        opacity: 1 !important;

        overflow: visible !important;

        margin-top: 1.5rem !important;
    }


    /* --------------------------------------------------------
       TEXT INPUTS
    -------------------------------------------------------- */

    div[data-testid="stTextInput"] input {
        min-height: 62px !important;

        background: #f5f6f7 !important;

        color: #111111 !important;

        border: 2px solid #c4c9cc !important;

        border-radius: 13px !important;

        font-size: 1.25rem !important;

        padding-left: 1rem !important;

        box-shadow:
            inset 0 2px 5px rgba(0,0,0,0.12) !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border: 2px solid #ffffff !important;

        box-shadow:
            0 0 0 2px rgba(255,255,255,0.25),
            inset 0 2px 5px rgba(0,0,0,0.12) !important;
    }

    div[data-testid="stTextInput"] label {
        color: white !important;

        font-size: 1.15rem !important;

        font-weight: 700 !important;
    }


    /* --------------------------------------------------------
       DATE INPUT
    -------------------------------------------------------- */

    div[data-testid="stDateInput"] input {
        min-height: 58px !important;

        background: #f5f6f7 !important;

        color: #111111 !important;

        border: 2px solid #c4c9cc !important;

        border-radius: 12px !important;

        font-size: 1.15rem !important;
    }

    div[data-testid="stDateInput"] label {
        color: white !important;

        font-size: 1.15rem !important;

        font-weight: 700 !important;
    }


    /* --------------------------------------------------------
       FILE UPLOADER
    -------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background:
            linear-gradient(
                145deg,
                #858f94,
                #697379
            ) !important;

        border: 2px solid #adb5b9 !important;

        border-radius: 18px !important;

        padding: 1.2rem !important;

        min-height: 150px !important;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.20),
            0 8px 20px rgba(0,0,0,0.20) !important;
    }

    [data-testid="stFileUploader"] label {
        color: white !important;

        font-size: 1.25rem !important;

        font-weight: 800 !important;
    }

    [data-testid="stFileUploader"] small {
        color: white !important;

        font-size: 1rem !important;
    }

    [data-testid="stFileUploader"] button {
        min-height: 48px !important;

        background: #f0f1f2 !important;

        color: #222222 !important;

        border: 1px solid #c4c7c9 !important;

        border-radius: 10px !important;

        font-size: 1rem !important;

        font-weight: 700 !important;
    }


    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    div.stButton > button {
        min-height: 58px !important;

        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;

        border-radius: 13px !important;

        border: 2px solid #aeb7bb !important;

        background:
            linear-gradient(
                145deg,
                #8e989d,
                #667177
            ) !important;

        color: white !important;

        font-size: 1.15rem !important;

        font-weight: 800 !important;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.22),
            0 7px 15px rgba(0,0,0,0.22) !important;

        transition:
            transform 0.15s ease,
            filter 0.15s ease !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;

        filter: brightness(1.10) !important;
    }


    /* --------------------------------------------------------
       SPEICHER-BUTTON
    -------------------------------------------------------- */

    .st-key-save_button button {
        min-height: 70px !important;

        font-size: 1.3rem !important;

        width: 100% !important;
    }


    /* --------------------------------------------------------
       ALLGEMEINE TEXTE
    -------------------------------------------------------- */

    .stMarkdown p {
        font-size: 1.12rem;
    }

    .stCaption {
        font-size: 1rem !important;

        color: #f2f2f2 !important;
    }


    /* --------------------------------------------------------
       INFO / SUCCESS / ERROR
    -------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 14px !important;

        font-size: 1.1rem !important;

        padding: 1rem !important;
    }


    /* --------------------------------------------------------
       BILDER
    -------------------------------------------------------- */

    [data-testid="stImage"] img {
        border-radius: 14px !important;
    }


    /* --------------------------------------------------------
       FUNDSTÜCK-KARTEN
    -------------------------------------------------------- */

    .st-key-item_card {
        min-height: 450px !important;

        background:
            linear-gradient(
                145deg,
                #8b9499,
                #697379
            ) !important;

        border: 2px solid #adb5b9 !important;

        border-radius: 18px !important;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.20),
            0 10px 25px rgba(0,0,0,0.25) !important;
    }

    .st-key-item_card p {
        color: white !important;

        font-size: 1.08rem !important;
    }

    .st-key-item_card strong {
        color: white !important;
    }


    /* --------------------------------------------------------
       TRENNLINIEN
    -------------------------------------------------------- */

    hr {
        border-color: rgba(255,255,255,0.20) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LABELS LADEN
# ============================================================

def load_labels():
    labels = []

    if LABELS_PATH.exists():
        try:
            lines = LABELS_PATH.read_text(
                encoding="utf-8"
            ).splitlines()

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                parts = line.split(maxsplit=1)

                if (
                    len(parts) == 2
                    and parts[0].rstrip(".").isdigit()
                ):
                    labels.append(parts[1].strip())
                else:
                    labels.append(line)

        except Exception:
            pass

    if not labels:
        labels = [
            "Hose",
            "Schuh",
            "T-Shirt",
            "Hoodie"
        ]

    return labels


LABELS = load_labels()


# ============================================================
# DATENBANK
# ============================================================

def get_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=10
    )

    return conn


def init_database():
    """
    Erstellt die Datenbank und repariert ältere Versionen
    automatisch.
    """

    conn = get_connection()

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT DEFAULT '',
                label TEXT DEFAULT '',
                location TEXT DEFAULT '',
                found_date TEXT DEFAULT '',
                created_at TEXT DEFAULT ''
            )
            """
        )

        conn.commit()

        # Vorhandene Spalten überprüfen
        columns = set()

        cursor = conn.execute(
            "PRAGMA table_info(items)"
        )

        for row in cursor.fetchall():
            columns.add(row[1])

        required_columns = {
            "filename": "TEXT DEFAULT ''",
            "label": "TEXT DEFAULT ''",
            "location": "TEXT DEFAULT ''",
            "found_date": "TEXT DEFAULT ''",
            "created_at": "TEXT DEFAULT ''"
        }

        # Fehlende Spalten automatisch hinzufügen
        for column, column_type in required_columns.items():

            if column not in columns:

                conn.execute(
                    f"""
                    ALTER TABLE items
                    ADD COLUMN {column} {column_type}
                    """
                )

        conn.commit()

    finally:
        conn.close()


init_database()


# ============================================================
# DATENBANK-FUNKTIONEN
# ============================================================

def save_item(
    image,
    label,
    location,
    found_date
):
    filename = (
        f"{uuid.uuid4().hex}.jpg"
    )

    image_path = UPLOAD_DIR / filename

    image_rgb = image.convert("RGB")

    image_rgb.save(
        image_path,
        "JPEG",
        quality=92
    )

    conn = get_connection()

    try:
        conn.execute(
            """
            INSERT INTO items
            (
                filename,
                label,
                location,
                found_date,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename,
                label,
                location,
                found_date.isoformat(),
                datetime.now().isoformat()
            )
        )

        conn.commit()

    finally:
        conn.close()


def get_oldest_items(limit=9):
    conn = get_connection()

    try:
        cursor = conn.execute(
            """
            SELECT
                id,
                filename,
                label,
                location,
                found_date,
                created_at
            FROM items
            ORDER BY
                CASE
                    WHEN found_date = ''
                    THEN 1
                    ELSE 0
                END,
                found_date ASC,
                created_at ASC,
                id ASC
            LIMIT ?
            """,
            (limit,)
        )

        return cursor.fetchall()

    finally:
        conn.close()


def search_items(query):
    conn = get_connection()

    try:
        search_value = f"%{query.lower()}%"

        cursor = conn.execute(
            """
            SELECT
                id,
                filename,
                label,
                location,
                found_date,
                created_at
            FROM items
            WHERE
                LOWER(label) LIKE ?
                OR LOWER(location) LIKE ?
            ORDER BY
                found_date DESC,
                created_at DESC
            """,
            (
                search_value,
                search_value
            )
        )

        return cursor.fetchall()

    finally:
        conn.close()


# ============================================================
# KI-MODELL LADEN
# ============================================================

@st.cache_resource
def load_model():
    return tf_keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# BILD KLASSIFIZIEREN
# ============================================================

def predict_image(image):

    model = load_model()

    prepared = image.convert("RGB")

    prepared = prepared.resize(
        (224, 224)
    )

    array = np.asarray(
        prepared
    ).astype(
        np.float32
    )

    array = (
        array / 127.5
    ) - 1.0

    array = np.expand_dims(
        array,
        axis=0
    )

    prediction = model.predict(
        array,
        verbose=0
    )[0]

    index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[index]
    )

    if index < len(LABELS):
        label = LABELS[index]
    else:
        label = str(index)

    return label, confidence


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Suche"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## Fundgrube"
    )

    st.divider()

    if st.button(
        "Suche",
        key="nav_search",
        width="stretch"
    ):
        st.session_state.page = "Suche"
        st.rerun()

    if st.button(
        "Bild hochladen",
        key="nav_upload",
        width="stretch"
    ):
        st.session_state.page = "Bild hochladen"
        st.rerun()

    if st.button(
        "Älteste Fundstücke",
        key="nav_oldest",
        width="stretch"
    ):
        st.session_state.page = "Älteste Fundstücke"
        st.rerun()


# ============================================================
# SEITENTITEL
# ============================================================

def page_title(title):

    with st.container(
        border=True,
        key="page_title"
    ):
        st.title(title)


# ============================================================
# SUCHE
# ============================================================

def show_search_page():

    page_title(
        "Fundgrube"
    )

    with st.container(
        border=True,
        key="search_card"
    ):

        st.subheader(
            "Etwas verloren? Suche es."
        )

        st.write(
            "Bewege den Mauszeiger über dieses Feld, "
            "um die Suche zu öffnen."
        )

        query = st.text_input(
            "Suchbegriff",
            placeholder=(
                "Zum Beispiel: Hoodie, Hose oder Schuh"
            ),
            label_visibility="collapsed",
            key="search_input"
        )

    if query.strip():

        results = search_items(
            query.strip()
        )

        st.write("")

        st.subheader(
            "Suchergebnisse"
        )

        if not results:

            with st.container(
                border=True
            ):
                st.write(
                    "Keine passenden Fundstücke gefunden."
                )

        else:

            for start in range(
                0,
                len(results),
                3
            ):

                row = results[
                    start:start + 3
                ]

                columns = st.columns(
                    len(row),
                    gap="large"
                )

                for column, item in zip(
                    columns,
                    row
                ):

                    (
                        item_id,
                        filename,
                        label,
                        location,
                        found_date,
                        created_at
                    ) = item

                    with column:

                        with st.container(
                            border=True
                        ):

                            image_path = (
                                UPLOAD_DIR
                                / filename
                            )

                            if image_path.exists():

                                try:
                                    image = Image.open(
                                        image_path
                                    )

                                    thumbnail = ImageOps.fit(
                                        image,
                                        (500, 500)
                                    )

                                    st.image(
                                        thumbnail,
                                        width="stretch"
                                    )

                                except Exception:
                                    st.write(
                                        "Bild konnte nicht geladen werden."
                                    )

                            st.subheader(
                                label or "Unbekannter Gegenstand"
                            )

                            st.write(
                                f"Fundort: {location or 'Nicht angegeben'}"
                            )

                            st.write(
                                f"Gefunden am: {found_date or 'Nicht angegeben'}"
                            )


# ============================================================
# UPLOAD
# ============================================================

def show_upload_page():

    page_title(
        "Bild hochladen"
    )

    with st.container(
        border=True
    ):

        st.subheader(
            "Foto des Fundstücks"
        )

        st.write(
            "Lade ein Foto hoch. Die KI erkennt automatisch, "
            "um welche Art von Kleidungsstück oder Schuh es sich handelt."
        )

        uploaded_file = st.file_uploader(
            "Foto aus deiner Mediathek auswählen",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],
            key="upload_file"
        )

    if uploaded_file is None:
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

    st.write("")

    with st.container(
        border=True
    ):

        st.subheader(
            "Hochgeladenes Bild"
        )

        st.image(
            image,
            width="stretch"
        )

    st.write("")

    # --------------------------------------------------------
    # KI
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "KI-Erkennung"
        )

        try:

            with st.spinner(
                "Das Bild wird analysiert..."
            ):

                label, confidence = predict_image(
                    image
                )

            st.success(
                f"Erkannt: {label} "
                f"mit {confidence:.0%} Wahrscheinlichkeit."
            )

        except Exception as error:

            st.error(
                "Das KI-Modell konnte nicht geladen "
                "oder ausgeführt werden."
            )

            st.caption(
                str(error)
            )

            return

    st.write("")

    # --------------------------------------------------------
    # INFORMATIONEN
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "Informationen zum Fundstück"
        )

        location = st.text_input(
            "Wo wurde der Gegenstand gefunden?",
            placeholder="Zum Beispiel: Sporthalle",
            key="found_location"
        )

        found_date = st.date_input(
            "Wann wurde der Gegenstand gefunden?",
            value=date.today(),
            key="found_date"
        )

    st.write("")

    # --------------------------------------------------------
    # SPEICHERN
    # --------------------------------------------------------

    if st.button(
        "Fundstück speichern",
        key="save_button",
        width="stretch"
    ):

        if not location.strip():

            st.warning(
                "Bitte gib zuerst den Fundort ein."
            )

            return

        try:

            save_item(
                image=image,
                label=label,
                location=location.strip(),
                found_date=found_date
            )

            st.success(
                "Das Fundstück wurde erfolgreich gespeichert."
            )

        except Exception as error:

            st.error(
                "Das Fundstück konnte nicht gespeichert werden."
            )

            st.caption(
                str(error)
            )


# ============================================================
# ÄLTESTE FUNDSTÜCKE
# ============================================================

def show_oldest_page():

    page_title(
        "Älteste Fundstücke"
    )

    with st.container(
        border=True
    ):

        st.subheader(
            "Die neun ältesten Fundstücke"
        )

        st.write(
            "Hier werden die ältesten gespeicherten "
            "Fundstücke angezeigt."
        )

    st.write("")

    try:

        items = get_oldest_items(
            9
        )

    except sqlite3.Error as error:

        st.error(
            "Die Datenbank konnte nicht gelesen werden."
        )

        st.caption(
            str(error)
        )

        return

    if not items:

        with st.container(
            border=True
        ):

            st.write(
                "Es wurden noch keine Fundstücke gespeichert."
            )

        return

    for start in range(
        0,
        len(items),
        3
    ):

        row = items[
            start:start + 3
        ]

        columns = st.columns(
            len(row),
            gap="large"
        )

        for column, item in zip(
            columns,
            row
        ):

            (
                item_id,
                filename,
                label,
                location,
                found_date,
                created_at
            ) = item

            with column:

                with st.container(
                    border=True,
                    key="item_card"
                ):

                    image_path = (
                        UPLOAD_DIR
                        / filename
                    )

                    if image_path.exists():

                        try:

                            image = Image.open(
                                image_path
                            ).convert(
                                "RGB"
                            )

                            thumbnail = ImageOps.fit(
                                image,
                                (600, 600)
                            )

                            st.image(
                                thumbnail,
                                width="stretch"
                            )

                        except Exception:

                            st.write(
                                "Bild konnte nicht geladen werden."
                            )

                    else:

                        st.write(
                            "Kein Bild vorhanden."
                        )

                    st.subheader(
                        label or "Unbekannter Gegenstand"
                    )

                    st.write(
                        f"Fundort: {location or 'Nicht angegeben'}"
                    )

                    st.write(
                        f"Gefunden am: {found_date or 'Nicht angegeben'}"
                    )


# ============================================================
# SEITE ANZEIGEN
# ============================================================

if st.session_state.page == "Suche":

    show_search_page()

elif st.session_state.page == "Bild hochladen":

    show_upload_page()

elif st.session_state.page == "Älteste Fundstücke":

    show_oldest_page()

else:

    st.session_state.page = "Suche"

    show_search_page()
