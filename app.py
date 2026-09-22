import os
import re
import sqlite3
import uuid
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image

import tf_keras


# =========================================================
# GRUNDEINSTELLUNGEN
# =========================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
DB_PATH = os.path.join(BASE_DIR, "fundgrube.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "fundgrube_uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================================
# DESIGN
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       HINTERGRUND - NICHT VERÄNDERN
       ===================================================== */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #07111f 0%,
                #0b1830 30%,
                #101d3a 60%,
                #07111f 100%
            ) !important;
        overflow-x: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        width: 900px;
        height: 900px;
        left: -300px;
        top: -300px;
        background: rgba(92, 55, 180, 0.22);
        border-radius: 50%;
        filter: blur(100px);
        z-index: 0;
        pointer-events: none;
        animation: purpleCloud 18s ease-in-out infinite alternate;
    }

    @keyframes purpleCloud {
        0% {
            transform: translate(0, 0) scale(1);
        }
        50% {
            transform: translate(180px, 120px) scale(1.18);
        }
        100% {
            transform: translate(60px, 220px) scale(1.05);
        }
    }

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        width: 1000px;
        height: 800px;
        right: -350px;
        top: 100px;
        background: rgba(105, 58, 190, 0.18);
        border-radius: 50%;
        filter: blur(110px);
        z-index: 0;
        pointer-events: none;
        animation: violetCloud 22s ease-in-out infinite alternate;
    }

    @keyframes violetCloud {
        0% {
            transform: translate(0, 0) scale(1);
        }
        50% {
            transform: translate(-160px, 100px) scale(1.2) rotate(8deg);
        }
        100% {
            transform: translate(-60px, -80px) scale(1.05);
        }
    }

    [data-testid="stAppViewContainer"]::after {
        content: "";
        position: fixed;
        width: 850px;
        height: 700px;
        left: 25%;
        bottom: -400px;
        background: rgba(38, 98, 190, 0.18);
        border-radius: 50%;
        filter: blur(120px);
        z-index: 0;
        pointer-events: none;
        animation: blueCloud 25s ease-in-out infinite alternate;
    }

    @keyframes blueCloud {
        0% {
            transform: translateX(-100px) scale(1);
        }
        50% {
            transform: translateX(180px) scale(1.25);
        }
        100% {
            transform: translateX(-50px) scale(1.05);
        }
    }

    .main::before {
        content: "";
        position: fixed;
        width: 700px;
        height: 600px;
        left: 35%;
        top: 35%;
        background: rgba(75, 45, 155, 0.12);
        border-radius: 50%;
        filter: blur(120px);
        z-index: 0;
        pointer-events: none;
        animation: bluePurpleCloud 20s ease-in-out infinite alternate;
    }

    @keyframes bluePurpleCloud {
        0% {
            transform: translate(-120px, -60px) scale(1);
        }
        50% {
            transform: translate(120px, 80px) scale(1.25) rotate(-10deg);
        }
        100% {
            transform: translate(-40px, 130px) scale(1.05);
        }
    }


    /* =====================================================
       ALLGEMEIN
       ===================================================== */

    .main,
    [data-testid="stAppViewContainer"] {
        position: relative;
    }

    .block-container {
        max-width: 1500px !important;
        padding-top: 5.5rem !important;
        padding-bottom: 4rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        position: relative;
        z-index: 1;
    }

    h1, h2, h3, h4, p, label {
        color: white !important;
    }

    .page-title {
        font-size: 4.2rem;
        font-weight: 900;
        letter-spacing: -2px;
        color: white;
        margin-bottom: 2.5rem;
        line-height: 1;
    }

    .section-title {
        font-size: 2.2rem;
        font-weight: 850;
        color: white;
        margin-bottom: 1.5rem;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background: rgba(5, 12, 25, 0.92) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1.05rem !important;
        font-weight: 650 !important;
    }


    /* =====================================================
       ÜBERSCHRIFTEN DER GRAUEN FELDER
       ===================================================== */

    .hover-title {
        color: white;
        font-size: 3.35rem;
        line-height: 0.9;
        font-weight: 950;
        letter-spacing: -2.5px;
        margin: 0;
        padding: 0;
        max-width: 620px;
    }

    .st-key-media_hover .hover-title,
    .st-key-camera_hover .hover-title {
        font-size: 3.0rem;
        line-height: 0.88;
        letter-spacing: -2.2px;
        max-width: 450px;
    }

    .st-key-latest_selector .hover-title,
    .st-key-oldest_selector .hover-title {
        font-size: 3.2rem;
        line-height: 0.88;
        letter-spacing: -2.3px;
        max-width: 400px;
    }


    /* =====================================================
       SUCHE - GESCHLOSSEN
       ===================================================== */

    .st-key-search_hover {
        background:
            linear-gradient(
                145deg,
                rgba(105, 110, 120, 0.94),
                rgba(64, 68, 76, 0.97)
            );
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 28px;
        padding: 2rem;
        min-height: 125px;
        max-height: 125px;
        overflow: hidden;
        box-shadow:
            0 18px 45px rgba(0,0,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.10);
        transition:
            max-height 0.55s ease,
            transform 0.45s ease,
            box-shadow 0.45s ease;
    }

    .st-key-search_hover:hover {
        max-height: 700px;
        transform: translateY(-5px);
        box-shadow:
            0 28px 60px rgba(0,0,0,0.38),
            inset 0 1px 0 rgba(255,255,255,0.14);
    }

    /* Alles außer dem Titel bleibt geschlossen */
    .st-key-search_hover > div:not(:first-child) {
        opacity: 0;
        transform: translateY(18px);
        pointer-events: none;
        transition:
            opacity 0.35s ease,
            transform 0.35s ease;
    }

    .st-key-search_hover:hover > div:not(:first-child) {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
    }


    /* =====================================================
       UPLOAD-KARTE
       ===================================================== */

    .st-key-media_hover {
        background:
            linear-gradient(
                145deg,
                rgba(105, 110, 120, 0.94),
                rgba(64, 68, 76, 0.97)
            );
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 28px;
        padding: 2rem;
        min-height: 125px;
        max-height: 125px;
        overflow: hidden;
        box-shadow:
            0 18px 45px rgba(0,0,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.10);
        transition:
            max-height 0.55s ease,
            transform 0.45s ease,
            box-shadow 0.45s ease;
    }

    .st-key-media_hover:hover {
        max-height: 950px;
        transform: translateY(-5px);
        box-shadow:
            0 28px 60px rgba(0,0,0,0.38),
            inset 0 1px 0 rgba(255,255,255,0.14);
    }

    .st-key-media_hover > div:not(:first-child) {
        opacity: 0;
        transform: translateY(20px);
        pointer-events: none;
        transition:
            opacity 0.35s ease,
            transform 0.35s ease;
    }

    .st-key-media_hover:hover > div:not(:first-child) {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
    }


    /* =====================================================
       KAMERA-KARTE
       ===================================================== */

    .st-key-camera_hover {
        background:
            linear-gradient(
                145deg,
                rgba(105, 110, 120, 0.94),
                rgba(64, 68, 76, 0.97)
            );
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 28px;
        padding: 2rem;
        min-height: 125px;
        max-height: 125px;
        overflow: hidden;
        box-shadow:
            0 18px 45px rgba(0,0,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.10);
        transition:
            max-height 0.55s ease,
            transform 0.45s ease,
            box-shadow 0.45s ease;
    }

    .st-key-camera_hover:hover {
        max-height: 950px;
        transform: translateY(-5px);
        box-shadow:
            0 28px 60px rgba(0,0,0,0.38),
            inset 0 1px 0 rgba(255,255,255,0.14);
    }

    .st-key-camera_hover > div:not(:first-child) {
        opacity: 0;
        transform: translateY(20px);
        pointer-events: none;
        transition:
            opacity 0.35s ease,
            transform 0.35s ease;
    }

    .st-key-camera_hover:hover > div:not(:first-child) {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
    }


    /* =====================================================
       FUNDSTÜCKE - AUSWAHLKARTEN
       ===================================================== */

    .st-key-latest_selector,
    .st-key-oldest_selector {
        background:
            linear-gradient(
                145deg,
                rgba(105, 110, 120, 0.94),
                rgba(64, 68, 76, 0.97)
            );
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 28px;
        padding: 2rem;
        min-height: 125px;
        max-height: 125px;
        overflow: hidden;
        box-shadow:
            0 18px 45px rgba(0,0,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.10);
        transition:
            max-height 0.55s ease,
            transform 0.45s ease,
            box-shadow 0.45s ease;
    }

    .st-key-latest_selector:hover,
    .st-key-oldest_selector:hover {
        max-height: 550px;
        transform: translateY(-5px);
        box-shadow:
            0 28px 60px rgba(0,0,0,0.38),
            inset 0 1px 0 rgba(255,255,255,0.14);
    }

    .st-key-latest_selector > div:not(:first-child),
    .st-key-oldest_selector > div:not(:first-child) {
        opacity: 0;
        transform: translateY(20px);
        pointer-events: none;
        transition:
            opacity 0.35s ease,
            transform 0.35s ease;
    }

    .st-key-latest_selector:hover > div:not(:first-child),
    .st-key-oldest_selector:hover > div:not(:first-child) {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
    }


    /* =====================================================
       TEXTFELDER
       ===================================================== */

    .stTextArea textarea,
    .stTextInput input {
        background: rgba(255,255,255,0.97) !important;
        color: #111 !important;
        border-radius: 16px !important;
        border: none !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }

    .stTextArea textarea {
        min-height: 150px !important;
    }

    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #555 !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        border-radius: 15px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        background: rgba(20, 30, 48, 0.92) !important;
        color: white !important;
        font-weight: 800 !important;
        min-height: 48px !important;
        transition:
            transform 0.2s ease,
            background 0.2s ease,
            box-shadow 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        background: rgba(35, 48, 70, 0.98) !important;
        box-shadow: 0 8px 22px rgba(0,0,0,0.25) !important;
    }


    /* =====================================================
       UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background: rgba(20, 27, 39, 0.55);
        border-radius: 18px;
        padding: 0.7rem;
        margin-top: 1.5rem;
    }

    [data-testid="stFileUploader"] * {
        color: white !important;
    }

    [data-testid="stCameraInput"] {
        margin-top: 1.5rem;
    }


    /* =====================================================
       BILD-VORSCHAU
       ===================================================== */

    .preview-window {
        margin-top: 1.5rem;
        height: 145px;
        overflow: hidden;
        border-radius: 18px;
    }

    .preview-strip {
        display: flex;
        gap: 12px;
        width: max-content;
        animation: previewMove 18s linear infinite;
    }

    .preview-strip img {
        width: 180px;
        height: 130px;
        object-fit: cover;
        border-radius: 15px;
        flex-shrink: 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }

    @keyframes previewMove {
        0% {
            transform: translateX(0);
        }
        100% {
            transform: translateX(-35%);
        }
    }


    /* =====================================================
       SUCHERGEBNISSE
       ===================================================== */

    .result-card {
        background: rgba(70, 76, 86, 0.92);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 22px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }

    .result-card img {
        width: 100%;
        height: 190px;
        object-fit: cover;
        border-radius: 16px;
    }

    .result-title {
        font-size: 1.35rem;
        font-weight: 850;
        margin-top: 0.8rem;
        color: white;
    }

    .result-info {
        color: rgba(255,255,255,0.78);
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }


    /* =====================================================
       GALLERY
       ===================================================== */

    .gallery-card {
        position: relative;
        background: rgba(62, 68, 78, 0.95);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 22px;
        padding: 10px;
        transition:
            transform 0.3s ease,
            box-shadow 0.3s ease;
        overflow: hidden;
        margin-bottom: 1rem;
    }

    .gallery-card:hover {
        transform: scale(1.035);
        box-shadow: 0 20px 45px rgba(0,0,0,0.4);
        z-index: 10;
    }

    .gallery-card img {
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border-radius: 15px;
        display: block;
    }

    .gallery-info {
        position: absolute;
        left: 10px;
        right: 10px;
        bottom: 10px;
        padding: 14px;
        border-radius: 14px;
        background: rgba(8, 13, 23, 0.92);
        color: white;
        opacity: 0;
        transform: translateY(12px);
        transition:
            opacity 0.25s ease,
            transform 0.25s ease;
        pointer-events: none;
    }

    .gallery-card:hover .gallery-info {
        opacity: 1;
        transform: translateY(0);
    }

    .gallery-info strong {
        font-size: 1rem;
        color: white;
    }

    .gallery-info span {
        color: rgba(255,255,255,0.78);
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LABELS
# =========================================================

def load_labels():
    labels = []

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                match = re.match(r"^\s*\d+\s+(.*)$", line)

                if match:
                    labels.append(match.group(1).strip())
                else:
                    labels.append(line)

    return labels


LABELS = load_labels()


# =========================================================
# DATENBANK
# =========================================================

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            location TEXT,
            found_date TEXT,
            image_path TEXT,
            created_at TEXT,
            found INTEGER DEFAULT 0
        )
        """
    )

    columns = [
        row["name"]
        for row in connection.execute(
            "PRAGMA table_info(items)"
        ).fetchall()
    ]

    if "found" not in columns:
        connection.execute(
            "ALTER TABLE items ADD COLUMN found INTEGER DEFAULT 0"
        )

    connection.commit()
    connection.close()


init_database()


# =========================================================
# DATENBANK-FUNKTIONEN
# =========================================================

def save_item(label, location, found_date, image):
    file_name = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(UPLOAD_DIR, file_name)

    image.save(
        image_path,
        format="JPEG",
        quality=92,
    )

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO items
        (
            label,
            location,
            found_date,
            image_path,
            created_at,
            found
        )
        VALUES (?, ?, ?, ?, ?, 0)
        """,
        (
            label,
            location,
            found_date,
            image_path,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )

    connection.commit()
    connection.close()


def mark_item_as_found(item_id):
    connection = get_connection()

    row = connection.execute(
        """
        SELECT image_path
        FROM items
        WHERE id = ?
        """,
        (item_id,),
    ).fetchone()

    if row:
        image_path = row["image_path"]

        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass

    connection.execute(
        """
        UPDATE items
        SET found = 1
        WHERE id = ?
        """,
        (item_id,),
    )

    connection.commit()
    connection.close()


def get_latest_items(limit=20):
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM items
        WHERE COALESCE(found, 0) = 0
        ORDER BY datetime(created_at) DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    connection.close()

    return rows


def get_oldest_items(limit=20):
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM items
        WHERE COALESCE(found, 0) = 0
        ORDER BY
            CASE
                WHEN found_date IS NULL
                  OR found_date = ''
                THEN 1
                ELSE 0
            END,
            date(found_date) ASC,
            datetime(created_at) ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    connection.close()

    return rows


# =========================================================
# DATUM
# =========================================================

def parse_date(date_text):
    if not date_text:
        return None

    formats = [
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
    ]

    for date_format in formats:
        try:
            return datetime.strptime(
                date_text.strip(),
                date_format,
            ).date()
        except ValueError:
            pass

    return None


# =========================================================
# SUCHE
# =========================================================

def search_items(query="", loss_date_text=""):
    connection = get_connection()

    query = query.strip()
    loss_date_text = loss_date_text.strip()

    if query:
        rows = connection.execute(
            """
            SELECT *
            FROM items
            WHERE COALESCE(found, 0) = 0
            AND (
                LOWER(label) LIKE LOWER(?)
                OR LOWER(location) LIKE LOWER(?)
            )
            """,
            (
                f"%{query}%",
                f"%{query}%",
            ),
        ).fetchall()

    else:
        rows = connection.execute(
            """
            SELECT *
            FROM items
            WHERE COALESCE(found, 0) = 0
            """
        ).fetchall()

    connection.close()

    search_date = parse_date(loss_date_text)

    if search_date:

        def date_difference(row):
            stored_date = parse_date(
                row["found_date"]
            )

            if stored_date is None:
                return 999999999

            return abs(
                (stored_date - search_date).days
            )

        rows = sorted(
            rows,
            key=date_difference,
        )

    else:
        rows = sorted(
            rows,
            key=lambda row: row["created_at"] or "",
            reverse=True,
        )

    return rows[:5]


# =========================================================
# KI-MODELL
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return tf_keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


def predict_image(image):
    model = load_model()

    if model is None:
        return "Unbekannt"

    image = image.convert("RGB")
    image = image.resize((224, 224))

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

    index = int(
        np.argmax(prediction[0])
    )

    if 0 <= index < len(LABELS):
        return LABELS[index]

    return "Unbekannt"


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:2rem;
            font-weight:900;
            margin-bottom:2rem;
            color:white;
        ">
            Fundgrube
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Suche",
            "Bild hochladen",
            "Fundstücke",
        ],
        label_visibility="collapsed",
    )


# =========================================================
# SESSION STATE
# =========================================================

if "fundstuecke_page" not in st.session_state:
    st.session_state.fundstuecke_page = "selection"


# =========================================================
# SEITE: SUCHE
# =========================================================

if page == "Suche":

    st.markdown(
        '<div class="page-title">Fundgrube</div>',
        unsafe_allow_html=True,
    )

    with st.container(key="search_hover"):

        st.markdown(
            """
            <div class="hover-title">
                Etwas<br>
                verloren?<br>
                Suche es.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        search_query = st.text_area(
            "Suche",
            placeholder="Beschreibe den verlorenen Gegenstand...",
            height=150,
            label_visibility="collapsed",
        )

        loss_date = st.text_input(
            "Verlustdatum",
            placeholder="Ungefähr wann verloren? z. B. 12.03.2026",
            label_visibility="collapsed",
        )

        search_button = st.button(
            "Suchen",
            key="search_button",
            width="stretch",
        )

    if search_button:

        results = search_items(
            search_query,
            loss_date,
        )

        st.write("")

        if not results:

            st.markdown(
                """
                <div class="gray-card">
                    <div style="
                        font-size:1.2rem;
                        font-weight:700;
                        color:white;
                    ">
                        Keine passenden Fundstücke gefunden.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<div class="section-title">Suchergebnisse</div>',
                unsafe_allow_html=True,
            )

            columns = st.columns(3)

            for index, item in enumerate(results):

                with columns[index % 3]:

                    image_path = item["image_path"]

                    if (
                        image_path
                        and os.path.exists(image_path)
                    ):

                        st.markdown(
                            '<div class="result-card">',
                            unsafe_allow_html=True,
                        )

                        st.image(
                            image_path,
                            use_container_width=True,
                        )

                        st.markdown(
                            f"""
                            <div class="result-title">
                                {item["label"] or "Unbekannt"}
                            </div>

                            <div class="result-info">
                                Fundort:
                                {item["location"] or "Nicht angegeben"}
                            </div>

                            <div class="result-info">
                                Gefunden am:
                                {item["found_date"] or "Nicht angegeben"}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True,
                        )


# =========================================================
# SEITE: BILD HOCHLADEN
# =========================================================

elif page == "Bild hochladen":

    st.markdown(
        '<div class="page-title">Bild hochladen</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(
        2,
        gap="large",
    )


    # =====================================================
    # MEDIATHEK
    # =====================================================

    with col1:

        with st.container(key="media_hover"):

            st.markdown(
                """
                <div class="hover-title">
                    Lade ein<br>
                    Bild aus<br>
                    deiner Mediathek
                </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Bild auswählen",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp",
                ],
                label_visibility="collapsed",
            )

            if uploaded_file is not None:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    image,
                    use_container_width=True,
                )

                prediction = predict_image(
                    image
                )

                st.markdown(
                    f"""
                    <div style="
                        background:rgba(10,17,29,0.75);
                        border-radius:16px;
                        padding:1rem;
                        margin-top:1rem;
                        color:white;
                    ">
                        <div style="
                            font-size:0.85rem;
                            opacity:0.7;
                        ">
                            KI-Erkennung
                        </div>

                        <div style="
                            font-size:1.4rem;
                            font-weight:850;
                        ">
                            {prediction}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                upload_location = st.text_input(
                    "Fundort",
                    placeholder="Wo wurde der Gegenstand gefunden?",
                    key="upload_location",
                )

                upload_date = st.text_input(
                    "Funddatum",
                    placeholder="z. B. 12.03.2026",
                    key="upload_date",
                )

                save_upload = st.button(
                    "Fundstück speichern",
                    key="save_upload",
                    width="stretch",
                )

                if save_upload:

                    if not upload_location.strip():

                        st.warning(
                            "Bitte gib einen Fundort ein."
                        )

                    elif not upload_date.strip():

                        st.warning(
                            "Bitte gib ein Funddatum ein."
                        )

                    elif parse_date(upload_date) is None:

                        st.warning(
                            "Bitte verwende ein gültiges Datum, z. B. 12.03.2026."
                        )

                    else:

                        save_item(
                            prediction,
                            upload_location.strip(),
                            upload_date.strip(),
                            image,
                        )

                        st.success(
                            "Das Fundstück wurde gespeichert."
                        )


    # =====================================================
    # KAMERA
    # =====================================================

    with col2:

        with st.container(key="camera_hover"):

            st.markdown(
                """
                <div class="hover-title">
                    Mach ein<br>
                    Foto in<br>
                    der App
                </div>
                """,
                unsafe_allow_html=True,
            )

            camera_file = st.camera_input(
                "Foto aufnehmen",
                label_visibility="collapsed",
            )

            if camera_file is not None:

                image = Image.open(
                    camera_file
                ).convert("RGB")

                st.image(
                    image,
                    use_container_width=True,
                )

                prediction = predict_image(
                    image
                )

                st.markdown(
                    f"""
                    <div style="
                        background:rgba(10,17,29,0.75);
                        border-radius:16px;
                        padding:1rem;
                        margin-top:1rem;
                        color:white;
                    ">
                        <div style="
                            font-size:0.85rem;
                            opacity:0.7;
                        ">
                            KI-Erkennung
                        </div>

                        <div style="
                            font-size:1.4rem;
                            font-weight:850;
                        ">
                            {prediction}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                camera_location = st.text_input(
                    "Fundort",
                    placeholder="Wo wurde der Gegenstand gefunden?",
                    key="camera_location",
                )

                camera_date = st.text_input(
                    "Funddatum",
                    placeholder="z. B. 12.03.2026",
                    key="camera_date",
                )

                save_camera = st.button(
                    "Fundstück speichern",
                    key="save_camera",
                    width="stretch",
                )

                if save_camera:

                    if not camera_location.strip():

                        st.warning(
                            "Bitte gib einen Fundort ein."
                        )

                    elif not camera_date.strip():

                        st.warning(
                            "Bitte gib ein Funddatum ein."
                        )

                    elif parse_date(camera_date) is None:

                        st.warning(
                            "Bitte verwende ein gültiges Datum, z. B. 12.03.2026."
                        )

                    else:

                        save_item(
                            prediction,
                            camera_location.strip(),
                            camera_date.strip(),
                            image,
                        )

                        st.success(
                            "Das Fundstück wurde gespeichert."
                        )


# =========================================================
# VORSCHAU FÜR FUNDSTÜCKE
# =========================================================

def create_preview_html(items):

    valid_items = []

    for item in items:

        path = item["image_path"]

        if path and os.path.exists(path):
            valid_items.append(path)

    if not valid_items:

        return """
        <div class="preview-window">
            <div style="
                color:rgba(255,255,255,0.65);
                padding:1rem;
            ">
                Noch keine Bilder vorhanden.
            </div>
        </div>
        """

    repeated = valid_items + valid_items

    images = ""

    for path in repeated:

        file_url = (
            "file://"
            + os.path.abspath(path)
        )

        images += f"""
            <img src="{file_url}">
        """

    return f"""
    <div class="preview-window">
        <div class="preview-strip">
            {images}
        </div>
    </div>
    """


# =========================================================
# SEITE: FUNDSTÜCKE
# =========================================================

elif page == "Fundstücke":

    if st.session_state.fundstuecke_page == "selection":

        st.markdown(
            '<div class="page-title">Fundstücke</div>',
            unsafe_allow_html=True,
        )

        latest_items = get_latest_items(10)
        oldest_items = get_oldest_items(10)

        col1, col2 = st.columns(
            2,
            gap="large",
        )


        # =================================================
        # NEUESTE
        # =================================================

        with col1:

            with st.container(
                key="latest_selector"
            ):

                st.markdown(
                    """
                    <div class="hover-title">
                        Neueste<br>
                        Bilder
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    create_preview_html(
                        latest_items
                    ),
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Neueste Bilder öffnen",
                    key="open_latest",
                    width="stretch",
                ):

                    st.session_state.fundstuecke_page = (
                        "latest"
                    )

                    st.rerun()


        # =================================================
        # ÄLTESTE
        # =================================================

        with col2:

            with st.container(
                key="oldest_selector"
            ):

                st.markdown(
                    """
                    <div class="hover-title">
                        Älteste<br>
                        Bilder
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    create_preview_html(
                        oldest_items
                    ),
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Älteste Bilder öffnen",
                    key="open_oldest",
                    width="stretch",
                ):

                    st.session_state.fundstuecke_page = (
                        "oldest"
                    )

                    st.rerun()


# =========================================================
# GALERIE
# =========================================================

if (
    page == "Fundstücke"
    and st.session_state.fundstuecke_page
    in ["latest", "oldest"]
):

    if st.session_state.fundstuecke_page == "latest":

        title = "Neueste Bilder"
        items = get_latest_items(100)

    else:

        title = "Älteste Bilder"
        items = get_oldest_items(100)

    st.markdown(
        f'<div class="page-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Zurück",
        key="back_to_fundstuecke",
    ):

        st.session_state.fundstuecke_page = (
            "selection"
        )

        st.rerun()

    st.write("")

    if not items:

        st.markdown(
            """
            <div class="gray-card">

                <div style="
                    font-size:1.3rem;
                    font-weight:750;
                    color:white;
                ">
                    Hier gibt es momentan keine Fundstücke.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        columns = st.columns(
            3,
            gap="large",
        )

        for index, item in enumerate(items):

            image_path = item["image_path"]

            if (
                not image_path
                or not os.path.exists(image_path)
            ):
                continue

            with columns[index % 3]:

                st.markdown(
                    f"""
                    <div class="gallery-card">

                        <img
                            src="file://{os.path.abspath(image_path)}"
                        >

                        <div class="gallery-info">

                            <strong>
                                Gegenstand:
                            </strong>
                            <br>

                            <span>
                                {item["label"] or "Nicht erkannt"}
                            </span>

                            <br><br>

                            <strong>
                                Fundort:
                            </strong>
                            <br>

                            <span>
                                {item["location"] or "Nicht angegeben"}
                            </span>

                            <br><br>

                            <strong>
                                Gefunden am:
                            </strong>
                            <br>

                            <span>
                                {item["found_date"] or "Nicht angegeben"}
                            </span>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Fundstück gefunden",
                    key=f"found_{item['id']}",
                    width="stretch",
                ):

                    mark_item_as_found(
                        item["id"]
                    )

                    st.rerun()
