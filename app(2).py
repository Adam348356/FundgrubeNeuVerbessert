import sqlite3
import uuid
import base64
from io import BytesIO
from pathlib import Path
from datetime import date, datetime

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tf_keras


st.set_page_config(
    page_title="Fundgrube",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
DB_PATH = BASE_DIR / "fundgrube.db"
UPLOAD_DIR = BASE_DIR / "fundgrube_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

    /* =====================================================
       ANIMIERTER HINTERGRUND
       NICHT VERÄNDERN
       ===================================================== */

    .stApp {
        position: relative;
        overflow-x: hidden !important;

        background:
            linear-gradient(
                135deg,
                #02030c 0%,
                #050719 30%,
                #070a20 55%,
                #040615 78%,
                #010208 100%
            ) !important;

        color: white;
    }


    .stApp::before {
        content: "";

        position: fixed;

        width: 115vw;
        height: 115vh;

        left: -35vw;
        top: -35vh;

        pointer-events: none;

        z-index: 0;

        border-radius: 50%;

        background:
            radial-gradient(
                ellipse at center,
                rgba(190, 45, 255, 0.88) 0%,
                rgba(155, 45, 245, 0.68) 16%,
                rgba(115, 40, 220, 0.44) 31%,
                rgba(75, 35, 165, 0.20) 48%,
                transparent 70%
            );

        filter: blur(55px);

        opacity: 0.95;

        transform:
            translate3d(-5vw, -4vh, 0)
            scale(1.0)
            rotate(0deg);

        animation:
            purpleCloud 18s ease-in-out infinite alternate;
    }


    [data-testid="stAppViewContainer"]::before {
        content: "";

        position: fixed;

        width: 95vw;
        height: 95vh;

        right: -35vw;
        top: 5vh;

        pointer-events: none;

        z-index: 0;

        border-radius: 50%;

        background:
            radial-gradient(
                ellipse at center,
                rgba(125, 35, 255, 0.82) 0%,
                rgba(100, 40, 230, 0.60) 17%,
                rgba(70, 40, 185, 0.34) 35%,
                rgba(45, 35, 130, 0.16) 50%,
                transparent 72%
            );

        filter: blur(65px);

        opacity: 0.9;

        transform:
            translate3d(5vw, 0, 0)
            scale(1.05)
            rotate(0deg);

        animation:
            violetCloud 23s ease-in-out infinite alternate;
    }


    [data-testid="stAppViewContainer"]::after {
        content: "";

        position: fixed;

        width: 110vw;
        height: 105vh;

        right: -40vw;
        bottom: -35vh;

        pointer-events: none;

        z-index: 0;

        border-radius: 50%;

        background:
            radial-gradient(
                ellipse at center,
                rgba(45, 100, 255, 0.88) 0%,
                rgba(45, 85, 245, 0.65) 17%,
                rgba(50, 65, 205, 0.42) 32%,
                rgba(45, 45, 150, 0.20) 49%,
                transparent 72%
            );

        filter: blur(60px);

        opacity: 0.90;

        transform:
            translate3d(4vw, 4vh, 0)
            scale(1.0)
            rotate(0deg);

        animation:
            blueCloud 21s ease-in-out infinite alternate;
    }


    .main::before {
        content: "";

        position: fixed;

        width: 90vw;
        height: 90vh;

        left: -30vw;
        bottom: -25vh;

        pointer-events: none;

        z-index: 0;

        border-radius: 50%;

        background:
            radial-gradient(
                ellipse at center,
                rgba(40, 90, 255, 0.72) 0%,
                rgba(65, 70, 235, 0.52) 18%,
                rgba(90, 45, 205, 0.30) 36%,
                rgba(55, 40, 150, 0.13) 52%,
                transparent 72%
            );

        filter: blur(70px);

        opacity: 0.82;

        transform:
            translate3d(-5vw, 5vh, 0)
            scale(1.05)
            rotate(0deg);

        animation:
            bluePurpleCloud 26s ease-in-out infinite alternate;
    }


    @keyframes purpleCloud {

        0% {
            transform:
                translate3d(-8vw, -6vh, 0)
                scale(0.92)
                rotate(-4deg);
        }

        20% {
            transform:
                translate3d(2vw, 3vh, 0)
                scale(1.04)
                rotate(2deg);
        }

        40% {
            transform:
                translate3d(14vw, 10vh, 0)
                scale(1.12)
                rotate(6deg);
        }

        60% {
            transform:
                translate3d(7vw, 20vh, 0)
                scale(1.07)
                rotate(2deg);
        }

        80% {
            transform:
                translate3d(-5vw, 13vh, 0)
                scale(1.16)
                rotate(-5deg);
        }

        100% {
            transform:
                translate3d(-15vw, 2vh, 0)
                scale(0.98)
                rotate(-8deg);
        }
    }


    @keyframes violetCloud {

        0% {
            transform:
                translate3d(9vw, -8vh, 0)
                scale(0.90)
                rotate(5deg);
        }

        20% {
            transform:
                translate3d(-3vw, 4vh, 0)
                scale(1.04)
                rotate(-2deg);
        }

        40% {
            transform:
                translate3d(-15vw, 15vh, 0)
                scale(1.14)
                rotate(-7deg);
        }

        60% {
            transform:
                translate3d(-8vw, 27vh, 0)
                scale(1.08)
                rotate(-3deg);
        }

        80% {
            transform:
                translate3d(7vw, 18vh, 0)
                scale(1.17)
                rotate(5deg);
        }

        100% {
            transform:
                translate3d(17vw, 5vh, 0)
                scale(0.96)
                rotate(9deg);
        }
    }


    @keyframes blueCloud {

        0% {
            transform:
                translate3d(10vw, 9vh, 0)
                scale(0.91)
                rotate(4deg);
        }

        20% {
            transform:
                translate3d(-2vw, -3vh, 0)
                scale(1.03)
                rotate(-2deg);
        }

        40% {
            transform:
                translate3d(-16vw, -13vh, 0)
                scale(1.13)
                rotate(-7deg);
        }

        60% {
            transform:
                translate3d(-24vw, -4vh, 0)
                scale(1.07)
                rotate(-3deg);
        }

        80% {
            transform:
                translate3d(-12vw, 12vh, 0)
                scale(1.16)
                rotate(5deg);
        }

        100% {
            transform:
                translate3d(4vw, 18vh, 0)
                scale(0.97)
                rotate(9deg);
        }
    }


    @keyframes bluePurpleCloud {

        0% {
            transform:
                translate3d(-8vw, 12vh, 0)
                scale(0.92)
                rotate(-5deg);
        }

        20% {
            transform:
                translate3d(5vw, 2vh, 0)
                scale(1.03)
                rotate(1deg);
        }

        40% {
            transform:
                translate3d(17vw, -8vh, 0)
                scale(1.12)
                rotate(7deg);
        }

        60% {
            transform:
                translate3d(25vw, 4vh, 0)
                scale(1.07)
                rotate(3deg);
        }

        80% {
            transform:
                translate3d(13vw, 17vh, 0)
                scale(1.15)
                rotate(-4deg);
        }

        100% {
            transform:
                translate3d(-1vw, 25vh, 0)
                scale(0.97)
                rotate(-9deg);
        }
    }


    /* =====================================================
       INHALT ÜBER DEM HINTERGRUND
       ===================================================== */

    [data-testid="stAppViewContainer"],
    .main,
    .block-container {
        position: relative;
        z-index: 1;
    }

    .block-container {
        max-width: 1500px !important;
        padding-top: 5.5rem !important;
        padding-bottom: 4rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(14, 17, 39, 0.98),
                rgba(5, 7, 20, 0.98)
            ) !important;

        border-right:
            1px solid rgba(255, 255, 255, 0.12);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }


    /* =====================================================
       SCHRIFT
       ===================================================== */

    h1,
    h2,
    h3,
    p,
    label,
    span,
    div {
        color: white;
    }


    /* =====================================================
       SEITENTITEL
       ===================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 22px !important;
    }

    .st-key-page_title {
        background:
            linear-gradient(
                145deg,
                rgba(70, 76, 92, 0.94),
                rgba(38, 42, 53, 0.96)
            );

        border:
            1px solid rgba(255, 255, 255, 0.13);

        box-shadow:
            0 18px 50px rgba(0, 0, 0, 0.30),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);

        padding: 1.3rem 1.6rem;
        margin-bottom: 2rem;
    }

    .st-key-page_title h1 {
        margin: 0;
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -0.04em;
    }


    /* =====================================================
       SUCHKARTE
       ===================================================== */

    .st-key-search_hover {
        background:
            linear-gradient(
                145deg,
                rgba(72, 77, 89, 0.95),
                rgba(39, 43, 53, 0.96)
            );

        border:
            1px solid rgba(255, 255, 255, 0.13);

        border-radius: 24px;

        padding: 2rem;

        min-height: 125px;
        max-height: 125px;

        overflow: hidden;

        transition:
            max-height 0.55s ease,
            transform 0.45s ease,
            box-shadow 0.45s ease;

        box-shadow:
            0 20px 55px rgba(0, 0, 0, 0.25);
    }

    .st-key-search_hover:hover {
        max-height: 600px;

        transform:
            translateY(-4px);

        box-shadow:
            0 28px 70px rgba(0, 0, 0, 0.35);
    }

    .search-title {
        font-size: 3.2rem;
        font-weight: 900;
        line-height: 0.98;
        letter-spacing: -0.045em;
        color: white;
        margin-bottom: 1rem;
    }

    .loss-date-label {
        font-size: 1rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }


    /* Suchinhalt unsichtbar, bis die Karte aufgeht */

    .st-key-search_hover textarea,
    .st-key-search_hover input,
    .st-key-search_hover .stButton {
        opacity: 0;

        transform:
            translateY(18px);

        pointer-events: none;

        transition:
            opacity 0.30s ease,
            transform 0.35s ease;
    }

    .st-key-search_hover:hover textarea,
    .st-key-search_hover:hover input,
    .st-key-search_hover:hover .stButton {
        opacity: 1;

        transform:
            translateY(0);

        pointer-events: auto;
    }

    .st-key-search_hover textarea {
        background: white !important;
        color: black !important;

        border-radius: 16px !important;
        border: none !important;

        font-size: 1.15rem !important;
        font-weight: 700 !important;

        padding: 1rem !important;
    }

    .st-key-search_hover input {
        background: white !important;
        color: black !important;

        border-radius: 14px !important;
        border: none !important;

        font-weight: 700 !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        border-radius: 14px !important;

        border:
            1px solid rgba(255, 255, 255, 0.16) !important;

        background:
            linear-gradient(
                145deg,
                rgba(95, 101, 116, 0.95),
                rgba(50, 54, 67, 0.98)
            ) !important;

        color: white !important;

        font-weight: 800 !important;

        min-height: 48px;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease,
            background 0.2s ease;
    }

    .stButton > button:hover {
        transform:
            translateY(-2px);

        box-shadow:
            0 10px 28px rgba(0, 0, 0, 0.30);

        background:
            linear-gradient(
                145deg,
                rgba(115, 121, 137, 0.98),
                rgba(61, 66, 80, 1)
            ) !important;
    }


    /* =====================================================
       UPLOAD-KARTEN
       ===================================================== */

    .st-key-media_hover,
    .st-key-camera_hover {
        background:
            linear-gradient(
                145deg,
                rgba(72, 77, 89, 0.95),
                rgba(39, 43, 53, 0.96)
            );

        border:
            1px solid rgba(255, 255, 255, 0.13);

        border-radius: 24px;

        padding: 2rem;

        min-height: 340px;
        max-height: 340px;

        overflow: hidden;

        transition:
            max-height 0.55s ease,
            transform 0.45s ease,
            box-shadow 0.45s ease;

        box-shadow:
            0 20px 55px rgba(0, 0, 0, 0.25);
    }

    .st-key-media_hover:hover,
    .st-key-camera_hover:hover {
        max-height: 900px;

        transform:
            translateY(-4px);

        box-shadow:
            0 28px 70px rgba(0, 0, 0, 0.35);
    }

    .upload-title,
    .camera-title {
        font-size: 3.6rem;
        font-weight: 900;
        line-height: 1.0;
        letter-spacing: -0.04em;

        margin-bottom: 1.3rem;
    }


    /* Upload-Inhalte verstecken */

    .st-key-media_hover
    [data-testid="stFileUploader"],
    .st-key-media_hover
    .stButton,
    .st-key-media_hover
    [data-testid="stImage"],
    .st-key-camera_hover
    [data-testid="stCameraInput"],
    .st-key-camera_hover
    .stButton,
    .st-key-camera_hover
    [data-testid="stImage"] {

        opacity: 0;

        transform:
            translateY(20px);

        pointer-events: none;

        transition:
            opacity 0.30s ease,
            transform 0.35s ease;
    }

    .st-key-media_hover:hover
    [data-testid="stFileUploader"],
    .st-key-media_hover:hover
    .stButton,
    .st-key-media_hover:hover
    [data-testid="stImage"],
    .st-key-camera_hover:hover
    [data-testid="stCameraInput"],
    .st-key-camera_hover:hover
    .stButton,
    .st-key-camera_hover:hover
    [data-testid="stImage"] {

        opacity: 1;

        transform:
            translateY(0);

        pointer-events: auto;
    }


    /* =====================================================
       FUNDSTÜCKE AUSWAHL
       ===================================================== */

    .st-key-latest_selector,
    .st-key-oldest_selector {

        background:
            linear-gradient(
                145deg,
                rgba(72, 77, 89, 0.95),
                rgba(39, 43, 53, 0.96)
            );

        border:
            1px solid rgba(255, 255, 255, 0.13);

        border-radius: 24px;

        padding: 1.6rem;

        min-height: 245px;
        max-height: 245px;

        overflow: hidden;

        transition:
            max-height 0.55s ease,
            transform 0.45s ease,
            box-shadow 0.45s ease;

        box-shadow:
            0 20px 55px rgba(0, 0, 0, 0.25);
    }

    .st-key-latest_selector:hover,
    .st-key-oldest_selector:hover {

        max-height: 500px;

        transform:
            translateY(-5px)
            scale(1.015);

        box-shadow:
            0 30px 70px rgba(0, 0, 0, 0.35);
    }

    .selector-title {
        font-size: 4.2rem;
        font-weight: 800;
        line-height: 0.92;
        letter-spacing: -0.04em;
        margin-bottom: 1rem;
    }

    .selector-title span {
        display: block;
    }


    /* Vorschau und Button zunächst unsichtbar */

    .st-key-latest_selector
    .preview-window,
    .st-key-oldest_selector
    .preview-window,
    .st-key-latest_selector
    .stButton,
    .st-key-oldest_selector
    .stButton {

        opacity: 0;

        transform:
            translateY(20px);

        pointer-events: none;

        transition:
            opacity 0.30s ease,
            transform 0.35s ease;
    }

    .st-key-latest_selector:hover
    .preview-window,
    .st-key-oldest_selector:hover
    .preview-window,
    .st-key-latest_selector:hover
    .stButton,
    .st-key-oldest_selector:hover
    .stButton {

        opacity: 1;

        transform:
            translateY(0);

        pointer-events: auto;
    }


    /* =====================================================
       VORSCHAUBILDER
       ===================================================== */

    .preview-window {
        width: 100%;
        height: auto;
        min-height: 320px;

        overflow: hidden;

        border-radius: 18px;

        margin-bottom: 1rem;

        background:
            rgba(0, 0, 0, 0.20);
    }

    .preview-track {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 18px;
        width: 100%;
    }

    .preview-image {
        width: min(100%, 420px);
        height: 300px;

        object-fit: cover;

        border-radius: 16px;

        margin-top: 0;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.30);
    }

    @keyframes previewMove {

        0% {
            transform:
                translateX(0);
        }

        100% {
            transform:
                translateX(-50%);
        }
    }


    /* =====================================================
       GALERIE
       ===================================================== */

    [class*="gallery_card_"] {

        transition:
            transform 0.35s ease,
            box-shadow 0.35s ease;

        overflow: hidden;
    }

    [class*="gallery_card_"]:hover {

        transform:
            scale(1.035)
            translateY(-5px);

        box-shadow:
            0 30px 65px rgba(0, 0, 0, 0.38);
    }

    [class*="gallery_card_"] img {

        border-radius: 14px;

        transition:
            transform 0.4s ease;
    }

    [class*="gallery_card_"]:hover img {

        transform:
            scale(1.025);
    }


    /* =====================================================
       GALERIE-INFOS
       ===================================================== */

    .gallery-info {

        max-height: 0;

        overflow: hidden;

        opacity: 0;

        transition:
            max-height 0.4s ease,
            opacity 0.35s ease;

        padding-top: 0;
    }

    [class*="gallery_card_"]:hover .gallery-info {

        max-height: 180px;

        opacity: 1;

        padding-top: 1rem;
    }

    .gallery-info-title {

        font-size: 1.25rem;
        font-weight: 800;

        margin-bottom: 0.4rem;
    }

    .gallery-info-line {

        font-size: 0.98rem;
        font-weight: 600;

        margin-top: 0.25rem;
    }


    /* =====================================================
       FUNDSTÜCK GEFUNDEN
       ===================================================== */

    .found-button-area {

        margin-top: 1rem;

        opacity: 0.15;

        transition:
            opacity 0.3s ease;
    }

    [class*="gallery_card_"]:hover
    .found-button-area {

        opacity: 1;
    }


    /* =====================================================
       STREAMLIT ELEMENTE
       ===================================================== */

    [data-testid="stFileUploader"],
    [data-testid="stCameraInput"] {

        margin-top: 1rem;
    }

    [data-testid="stFileUploaderDropzone"],
    [data-testid="stCameraInput"] {

        border-radius: 16px !important;
    }

    .stAlert {

        border-radius: 14px !important;
    }

    hr {

        border-color:
            rgba(255, 255, 255, 0.12) !important;
    }

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# LABELS LADEN
# =========================================================

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

                parts = line.split(
                    maxsplit=1
                )

                if (
                    len(parts) == 2
                    and parts[0].rstrip(".").isdigit()
                ):

                    labels.append(
                        parts[1].strip()
                    )

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


# =========================================================
# DATENBANK
# =========================================================

def get_connection():

    return sqlite3.connect(
        str(DB_PATH),
        timeout=10
    )


def init_database():

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
                created_at TEXT DEFAULT '',
                found INTEGER DEFAULT 0
            )
            """
        )

        conn.commit()

        cursor = conn.execute(
            "PRAGMA table_info(items)"
        )

        columns = {
            row[1]
            for row in cursor.fetchall()
        }

        required_columns = {
            "filename": "TEXT DEFAULT ''",
            "label": "TEXT DEFAULT ''",
            "location": "TEXT DEFAULT ''",
            "found_date": "TEXT DEFAULT ''",
            "created_at": "TEXT DEFAULT ''",
            "found": "INTEGER DEFAULT 0"
        }

        for column, definition in required_columns.items():

            if column not in columns:

                conn.execute(
                    f"ALTER TABLE items ADD COLUMN "
                    f"{column} {definition}"
                )

        conn.commit()

    finally:

        conn.close()


init_database()


# =========================================================
# FUNDSTÜCK SPEICHERN
# =========================================================

def save_item(
    image,
    label,
    location,
    found_date
):

    filename = (
        uuid.uuid4().hex
        +
        ".jpg"
    )

    image_path = (
        UPLOAD_DIR
        /
        filename
    )

    image.convert(
        "RGB"
    ).save(
        str(image_path),
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
                created_at,
                found
            )
            VALUES (?, ?, ?, ?, ?, 0)
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


# =========================================================
# FUNDSTÜCK ALS GEFUNDEN MARKIEREN
# =========================================================

def mark_item_as_found(
    item_id
):

    conn = get_connection()

    filename = None

    try:

        cursor = conn.execute(
            """
            SELECT filename
            FROM items
            WHERE id = ?
            """,
            (item_id,)
        )

        result = cursor.fetchone()

        if result:
            filename = result[0]

        conn.execute(
            """
            UPDATE items
            SET found = 1
            WHERE id = ?
            """,
            (item_id,)
        )

        conn.commit()

    finally:

        conn.close()

    if filename:

        image_path = (
            UPLOAD_DIR
            /
            str(filename)
        )

        try:

            if image_path.exists():
                image_path.unlink()

        except Exception:
            pass


# =========================================================
# FUNDSTÜCKE ABFRAGEN
# =========================================================

def get_latest_items(
    limit=12
):

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
            WHERE found = 0
            ORDER BY
                created_at DESC,
                id DESC
            LIMIT ?
            """,
            (limit,)
        )

        return cursor.fetchall()

    finally:

        conn.close()


def get_oldest_items(
    limit=12
):

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
            WHERE found = 0
            ORDER BY
                CASE
                    WHEN found_date = '' THEN 1
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


# =========================================================
# DATUM PARSEN
# =========================================================

def parse_search_date(
    date_text
):

    if not date_text:
        return None

    value = date_text.strip()

    if not value:
        return None

    formats = [
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%d/%m/%Y"
    ]

    for date_format in formats:

        try:

            return datetime.strptime(
                value,
                date_format
            ).date()

        except ValueError:
            continue

    return None


# =========================================================
# SUCHE
# =========================================================

def search_items(
    query,
    loss_date=None
):

    conn = get_connection()

    try:

        search_value = (
            "%"
            +
            query.lower()
            +
            "%"
        )

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
            WHERE found = 0
              AND (
                    LOWER(label) LIKE ?
                    OR LOWER(location) LIKE ?
                  )
            """,
            (
                search_value,
                search_value
            )
        )

        results = cursor.fetchall()

    finally:

        conn.close()

    if loss_date is None:

        results.sort(
            key=lambda item: (
                item[4] == "",
                item[4],
                item[5]
            ),
            reverse=True
        )

        return results[:5]

    dated_results = []

    for item in results:

        found_date_text = item[4]

        try:

            found_date = datetime.strptime(
                found_date_text,
                "%Y-%m-%d"
            ).date()

        except Exception:

            continue

        difference = abs(
            (
                found_date
                -
                loss_date
            ).days
        )

        dated_results.append(
            (
                difference,
                item
            )
        )

    dated_results.sort(
        key=lambda entry: (
            entry[0],
            entry[1][4],
            entry[1][5]
        )
    )

    return [
        item
        for difference, item
        in dated_results[:5]
    ]


# =========================================================
# KI-MODELL
# =========================================================

@st.cache_resource
def load_model():

    return tf_keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


def predict_image(
    image
):

    model = load_model()

    prepared = image.convert(
        "RGB"
    )

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

    label = (
        LABELS[index]
        if index < len(LABELS)
        else str(index)
    )

    return label, confidence


# =========================================================
# SIDEBAR
# =========================================================

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
        "Fundstücke",
        key="nav_oldest",
        width="stretch"
    ):

        st.session_state.page = "Fundstücke"
        st.rerun()


# =========================================================
# SEITENTITEL
# =========================================================

def page_title(
    title
):

    with st.container(
        border=True,
        key="page_title"
    ):

        st.title(title)


# =========================================================
# SUCHERGEBNISSE
# =========================================================

def display_search_results(
    results
):

    if not results:

        with st.container(
            border=True
        ):

            st.subheader(
                "Keine passenden Fundstücke gefunden."
            )

        return

    st.subheader(
        "Suchergebnisse"
    )

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
                    border=True,
                    key=f"result_{item_id}"
                ):

                    image_path = (
                        UPLOAD_DIR
                        /
                        str(filename)
                    )

                    if image_path.exists():

                        try:

                            image = Image.open(
                                str(image_path)
                            ).convert("RGB")

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
                        label
                        or
                        "Unbekannter Gegenstand"
                    )

                    st.write(
                        "Fundort: "
                        +
                        (
                            location
                            or
                            "Nicht angegeben"
                        )
                    )

                    st.write(
                        "Gefunden am: "
                        +
                        (
                            found_date
                            or
                            "Nicht angegeben"
                        )
                    )


# =========================================================
# SUCHSEITE
# =========================================================

def show_search_page():

    page_title(
        "Fundgrube"
    )

    with st.container(
        border=True,
        key="search_hover"
    ):

        st.markdown(
            """
            <div class="search-title">
                Etwas verloren? Suche es.
            </div>
            """,
            unsafe_allow_html=True
        )

        search_text = st.text_area(
            "Suchfeld",
            placeholder=(
                "Beschreibe, was du verloren hast.\n"
                "Zum Beispiel: schwarze Hose aus der Sporthalle"
            ),
            height=230,
            key="search_text_area",
            label_visibility="collapsed"
        )

        st.markdown(
            """
            <div class="loss-date-label">
                Ungefähres Verlustdatum
            </div>
            """,
            unsafe_allow_html=True
        )

        loss_date_text = st.text_input(
            "Verlustdatum",
            placeholder="Zum Beispiel: 15.09.2026",
            key="search_loss_date",
            label_visibility="collapsed"
        )

        search_clicked = st.button(
            "Suchen",
            key="search_button",
            width="stretch"
        )

    if search_clicked:

        query = search_text.strip()

        entered_date = loss_date_text.strip()

        loss_date = None

        if entered_date:

            loss_date = parse_search_date(
                entered_date
            )

            if loss_date is None:

                st.warning(
                    "Bitte gib das Datum im Format "
                    "TT.MM.JJJJ ein, zum Beispiel 15.09.2026."
                )

                return

        if not query and loss_date is None:

            st.warning(
                "Bitte gib entweder ein verlorenes Objekt "
                "oder ein Verlustdatum ein."
            )

            return

        results = search_items(
            query,
            loss_date
        )

        if loss_date is not None:

            st.caption(
                "Es werden die 5 Fundstücke angezeigt, "
                "deren Funddatum deinem eingegebenen "
                "Verlustdatum am nächsten liegt."
            )

        display_search_results(
            results
        )


# =========================================================
# BILD AUS MEDIATHEK
# =========================================================

def process_uploaded_image(
    uploaded_file
):

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Das Bild konnte nicht geöffnet werden."
        )

        return

    with st.container(
        border=True
    ):

        st.subheader(
            "Ausgewähltes Bild"
        )

        st.image(
            image,
            width="stretch"
        )

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
                "Erkannt: "
                +
                label
                +
                " ("
                +
                f"{confidence:.0%}"
                +
                ")"
            )

        except Exception as error:

            st.error(
                "Das KI-Modell konnte nicht ausgeführt werden."
            )

            st.caption(
                str(error)
            )

            return

    with st.container(
        border=True
    ):

        st.subheader(
            "Informationen zum Fundstück"
        )

        location = st.text_input(
            "Wo wurde der Gegenstand gefunden?",
            placeholder="Zum Beispiel: Sporthalle",
            key="media_location"
        )

        found_date = st.date_input(
            "Wann wurde der Gegenstand gefunden?",
            value=date.today(),
            key="media_date"
        )

    if st.button(
        "Fundstück speichern",
        key="media_save",
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


# =========================================================
# KAMERA
# =========================================================

def process_camera_image(
    camera_image
):

    try:

        image = Image.open(
            camera_image
        ).convert("RGB")

    except Exception:

        st.error(
            "Das Kamerabild konnte nicht geöffnet werden."
        )

        return

    with st.container(
        border=True
    ):

        st.subheader(
            "Aufgenommenes Bild"
        )

        st.image(
            image,
            width="stretch"
        )

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
                "Erkannt: "
                +
                label
                +
                " ("
                +
                f"{confidence:.0%}"
                +
                ")"
            )

        except Exception as error:

            st.error(
                "Das KI-Modell konnte nicht ausgeführt werden."
            )

            st.caption(
                str(error)
            )

            return

    with st.container(
        border=True
    ):

        st.subheader(
            "Informationen zum Fundstück"
        )

        location = st.text_input(
            "Wo wurde der Gegenstand gefunden?",
            placeholder="Zum Beispiel: Sporthalle",
            key="camera_location"
        )

        found_date = st.date_input(
            "Wann wurde der Gegenstand gefunden?",
            value=date.today(),
            key="camera_date"
        )

    if st.button(
        "Fundstück speichern",
        key="camera_save",
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


# =========================================================
# UPLOAD-SEITE
# =========================================================

def show_upload_page():

    page_title(
        "Bild hochladen"
    )

    st.subheader(
        "Wie möchtest du das Bild hinzufügen?"
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        with st.container(
            border=True,
            key="media_hover"
        ):

            st.markdown(
                """
                <div class="upload-title">
                    Lade ein<br>
                    Bild aus<br>
                    deiner Mediathek<br>
                    hoch
                </div>
                """,
                unsafe_allow_html=True
            )

            uploaded_file = st.file_uploader(
                "Bild auswählen",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp"
                ],
                key="media_uploader"
            )

            if uploaded_file is not None:

                process_uploaded_image(
                    uploaded_file
                )

    with col2:

        with st.container(
            border=True,
            key="camera_hover"
        ):

            st.markdown(
                """
                <div class="camera-title">
                    Mach ein<br>
                    Foto in<br>
                    der App
                </div>
                """,
                unsafe_allow_html=True
            )

            camera_image = st.camera_input(
                "Kamera öffnen",
                key="camera_input"
            )

            if camera_image is not None:

                process_camera_image(
                    camera_image
                )


# =========================================================
# VORSCHAUBILDER
# =========================================================

def create_preview_html(
    items
):

    valid_items = []

    for item in items:

        filename = item[1]

        image_path = (
            UPLOAD_DIR
            /
            str(filename)
        )

        if image_path.exists():

            valid_items.append(
                image_path
            )

    if not valid_items:

        return """
        <div
            style="
                width:100%;
                height:210px;
                display:flex;
                align-items:center;
                justify-content:center;
                color:white;
                font-size:1.2rem;
                font-weight:800;
            "
        >
            Noch keine Bilder vorhanden.
        </div>
        """

    all_images = valid_items

    image_tags = ""

    for image_path in all_images:

        try:

            image = Image.open(
                str(image_path)
            ).convert("RGB")

            image = ImageOps.fit(
                image,
                (650, 500)
            )

            buffer = BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=82
            )

            encoded = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

            image_tags += (
                '<img class="preview-image" '
                'src="data:image/jpeg;base64,'
                +
                encoded
                +
                '" />'
            )

        except Exception:
            continue

    return (
        '<div class="preview-window">'
        '<div class="preview-track">'
        +
        image_tags
        +
        '</div>'
        '</div>'
    )


# =========================================================
# FUNDSTÜCKE STARTSEITE
# =========================================================

def show_oldest_page():

    page_title(
        "Fundstücke"
    )

    latest_items = get_latest_items(
        8
    )

    oldest_items = get_oldest_items(
        8
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        with st.container(
            border=True,
            key="latest_selector"
        ):

            st.markdown(
                """
                <div class="selector-title">
                    <span>Neueste</span>
                    <span>Bilder</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                create_preview_html(
                    latest_items
                ),
                unsafe_allow_html=True
            )

            if st.button(
                "Neueste Bilder öffnen",
                key="open_latest",
                width="stretch"
            ):

                st.session_state.page = (
                    "Neueste Bilder"
                )

                st.rerun()

    with col2:

        with st.container(
            border=True,
            key="oldest_selector"
        ):

            st.markdown(
                """
                <div class="selector-title">
                    <span>Älteste</span>
                    <span>Bilder</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                create_preview_html(
                    oldest_items
                ),
                unsafe_allow_html=True
            )

            if st.button(
                "Älteste Bilder öffnen",
                key="open_oldest",
                width="stretch"
            ):

                st.session_state.page = (
                    "Älteste Bilder"
                )

                st.rerun()


# =========================================================
# GALERIE
# =========================================================

def show_gallery_page(
    title,
    items,
    page_name
):

    page_title(
        title
    )

    if st.button(
        "Zurück zu Fundstücken",
        key=f"back_{page_name}",
        width="stretch"
    ):

        st.session_state.page = (
            "Fundstücke"
        )

        st.rerun()

    st.write("")

    if not items:

        with st.container(
            border=True
        ):

            st.subheader(
                "Noch keine Fundstücke vorhanden."
            )

            st.write(
                "Sobald Fundstücke gespeichert wurden, "
                "werden sie hier angezeigt."
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
            3,
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
                    key=(
                        f"gallery_card_"
                        f"{page_name}_"
                        f"{item_id}"
                    )
                ):

                    image_path = (
                        UPLOAD_DIR
                        /
                        str(filename)
                    )

                    if image_path.exists():

                        try:

                            image = Image.open(
                                str(image_path)
                            ).convert("RGB")

                            thumbnail = ImageOps.fit(
                                image,
                                (650, 500)
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

                    st.markdown(
                        """
                        <div class="gallery-info">

                            <div class="gallery-info-title">
                                %s
                            </div>

                            <div class="gallery-info-line">
                                Fundort: %s
                            </div>

                            <div class="gallery-info-line">
                                Gefunden am: %s
                            </div>

                        </div>
                        """
                        %
                        (
                            label
                            or
                            "Unbekannter Gegenstand",

                            location
                            or
                            "Nicht angegeben",

                            found_date
                            or
                            "Nicht angegeben"
                        ),
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '<div class="found-button-area">',
                        unsafe_allow_html=True
                    )

                    if st.button(
                        "Fundstück gefunden",
                        key=(
                            f"found_"
                            f"{page_name}_"
                            f"{item_id}"
                        ),
                        width="stretch"
                    ):

                        mark_item_as_found(
                            item_id
                        )

                        st.success(
                            "Das Fundstück wurde entfernt."
                        )

                        st.rerun()

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )


# =========================================================
# NAVIGATION
# =========================================================

current_page = st.session_state.get(
    "page",
    "Suche"
)


if current_page == "Suche":

    show_search_page()


elif current_page == "Bild hochladen":

    show_upload_page()


elif current_page == "Fundstücke":

    show_oldest_page()


elif current_page == "Neueste Bilder":

    show_gallery_page(
        "Neueste Bilder",
        get_latest_items(30),
        "latest"
    )


elif current_page == "Älteste Bilder":

    show_gallery_page(
        "Älteste Bilder",
        get_oldest_items(30),
        "oldest"
    )


else:

    st.session_state.page = "Suche"

    show_search_page()
