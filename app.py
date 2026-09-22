import sqlite3
import uuid
from pathlib import Path
from datetime import date, datetime

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tf_keras


# ============================================================
# STREAMLIT
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

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Suche"

if "search_open" not in st.session_state:
    st.session_state.search_open = False


# ============================================================
# DESIGN / CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GESAMTER HINTERGRUND
    ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(120, 145, 160, 0.22),
                transparent 32%
            ),
            radial-gradient(
                circle at 85% 90%,
                rgba(60, 100, 125, 0.20),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #071d2b 0%,
                #0c2d41 45%,
                #092334 100%
            );

        color: white;
    }


    /* ========================================================
       HAUPTBEREICH
    ======================================================== */

    .block-container {
        max-width: 1500px !important;

        padding-top: 2rem !important;
        padding-bottom: 4rem !important;

        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #59646b 0%,
                #454f56 50%,
                #354047 100%
            ) !important;

        border-right: 1px solid rgba(255,255,255,0.15);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;

        min-height: 64px !important;

        margin-bottom: 0.5rem !important;

        border-radius: 13px !important;

        border: 2px solid #aeb8bd !important;

        background:
            linear-gradient(
                145deg,
                #737e84,
                #515c63
            ) !important;

        color: white !important;

        font-size: 1.1rem !important;
        font-weight: 800 !important;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.20),
            0 6px 14px rgba(0,0,0,0.22) !important;

        transition:
            transform 0.15s ease,
            filter 0.15s ease !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;

        filter: brightness(1.10) !important;
    }


    /* ========================================================
       GRAUE CONTAINER
    ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(
                145deg,
                #929b9f 0%,
                #778187 50%,
                #687278 100%
            ) !important;

        border: 2px solid #b6bec2 !important;

        border-radius: 20px !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.24),
            inset 0 -3px 8px rgba(0,0,0,0.12),
            0 12px 30px rgba(0,0,0,0.30) !important;
    }


    /* ========================================================
       SEITENTITEL
    ======================================================== */

    .st-key-page_title {
        width: 100% !important;

        min-height: 145px !important;

        margin-bottom: 2rem !important;

        display: flex !important;
        align-items: center !important;

        background:
            linear-gradient(
                145deg,
                #949da1,
                #737d83
            ) !important;

        border: 2px solid #b9c0c4 !important;

        border-radius: 22px !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.25),
            0 15px 35px rgba(0,0,0,0.30) !important;
    }

    .st-key-page_title h1 {
        color: white !important;

        font-size: 3.6rem !important;

        line-height: 1.1 !important;

        font-weight: 900 !important;

        margin: 0 !important;
    }


    /* ========================================================
       ÜBERSCHRIFTEN
    ======================================================== */

    h1,
    h2,
    h3,
    p,
    label {
        color: white !important;
    }

    h2 {
        font-size: 2.2rem !important;

        font-weight: 900 !important;
    }

    h3 {
        font-size: 1.7rem !important;

        font-weight: 850 !important;
    }

    .stMarkdown p {
        font-size: 1.15rem !important;
    }


    /* ========================================================
       SUCHKARTE - GESCHLOSSEN
       
       Absichtlich nur so groß, dass der Text umgeben wird.
    ======================================================== */

    .st-key-search_closed {
        width: 100% !important;

        min-height: 150px !important;

        padding: 0 !important;

        margin-bottom: 1.5rem !important;

        background:
            linear-gradient(
                145deg,
                #949da1 0%,
                #778187 55%,
                #687278 100%
            ) !important;

        border: 2px solid #b8c0c4 !important;

        border-radius: 22px !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.25),
            inset 0 -3px 8px rgba(0,0,0,0.12),
            0 12px 28px rgba(0,0,0,0.28) !important;

        transition:
            transform 0.20s ease,
            box-shadow 0.20s ease !important;
    }

    .st-key-search_closed:hover {
        transform: scale(1.012) !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.28),
            0 16px 34px rgba(0,0,0,0.35) !important;
    }


    /* Der komplette Button ist die klickbare Fläche */

    .st-key-search_closed .stButton {
        margin: 0 !important;

        padding: 0 !important;
    }

    .st-key-search_closed .stButton > button {
        width: 100% !important;

        min-height: 150px !important;

        padding: 1.5rem 2.5rem !important;

        background: transparent !important;

        border: none !important;

        border-radius: 20px !important;

        box-shadow: none !important;

        color: white !important;

        font-size: 2.65rem !important;

        line-height: 1.15 !important;

        font-weight: 900 !important;

        text-align: left !important;

        transition: none !important;
    }

    .st-key-search_closed .stButton > button:hover {
        background: transparent !important;

        border: none !important;

        box-shadow: none !important;

        transform: none !important;

        filter: none !important;
    }


    /* ========================================================
       SUCHKARTE - GEÖFFNET
       
       Wird nach dem Anklicken deutlich größer.
    ======================================================== */

    .st-key-search_open {
        width: 100% !important;

        min-height: 440px !important;

        padding: 2.5rem !important;

        margin-bottom: 2rem !important;

        background:
            linear-gradient(
                145deg,
                #949da1 0%,
                #778187 50%,
                #687278 100%
            ) !important;

        border: 2px solid #b8c0c4 !important;

        border-radius: 24px !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.25),
            inset 0 -3px 8px rgba(0,0,0,0.12),
            0 20px 45px rgba(0,0,0,0.36) !important;
    }

    .st-key-search_open h2 {
        color: white !important;

        font-size: 2.8rem !important;

        line-height: 1.15 !important;

        font-weight: 900 !important;

        margin-top: 0 !important;

        margin-bottom: 1.5rem !important;
    }


    /* ========================================================
       MEHRZEILIGES SUCHFELD
    ======================================================== */

    .st-key-search_open textarea {
        min-height: 220px !important;

        width: 100% !important;

        background: #f7f8f8 !important;

        color: #111111 !important;

        border: 3px solid #c5cbce !important;

        border-radius: 17px !important;

        font-size: 1.65rem !important;

        font-weight: 800 !important;

        line-height: 1.45 !important;

        padding: 1.3rem !important;

        resize: vertical !important;

        box-shadow:
            inset 0 3px 7px rgba(0,0,0,0.13) !important;
    }

    .st-key-search_open textarea:focus {
        border: 3px solid white !important;

        outline: none !important;

        box-shadow:
            0 0 0 3px rgba(255,255,255,0.18),
            inset 0 3px 7px rgba(0,0,0,0.13) !important;
    }

    .st-key-search_open textarea::placeholder {
        color: #555d61 !important;

        font-size: 1.4rem !important;

        font-weight: 700 !important;
    }

    .st-key-search_open label {
        color: white !important;

        font-size: 1.2rem !important;

        font-weight: 800 !important;
    }


    /* ========================================================
       NORMALE TEXT INPUTS
    ======================================================== */

    div[data-testid="stTextInput"] input {
        min-height: 62px !important;

        background: #f5f6f7 !important;

        color: #111111 !important;

        border: 2px solid #c4c9cc !important;

        border-radius: 13px !important;

        font-size: 1.25rem !important;

        font-weight: 700 !important;

        padding-left: 1rem !important;
    }


    /* ========================================================
       DATE INPUT
    ======================================================== */

    div[data-testid="stDateInput"] input {
        min-height: 60px !important;

        background: #f5f6f7 !important;

        color: #111111 !important;

        border: 2px solid #c4c9cc !important;

        border-radius: 13px !important;

        font-size: 1.2rem !important;

        font-weight: 700 !important;
    }

    div[data-testid="stDateInput"] label {
        color: white !important;

        font-size: 1.15rem !important;

        font-weight: 800 !important;
    }


    /* ========================================================
       FILE UPLOADER
    ======================================================== */

    [data-testid="stFileUploader"] {
        background:
            linear-gradient(
                145deg,
                #858f94,
                #697379
            ) !important;

        border: 2px solid #adb5b9 !important;

        border-radius: 18px !important;

        padding: 1.4rem !important;

        min-height: 160px !important;

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
        min-height: 50px !important;

        background: #f0f1f2 !important;

        color: #222222 !important;

        border: 1px solid #c4c7c9 !important;

        border-radius: 10px !important;

        font-size: 1rem !important;

        font-weight: 800 !important;
    }


    /* ========================================================
       NORMALE BUTTONS
    ======================================================== */

    div.stButton > button {
        min-height: 60px !important;

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
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;

        filter: brightness(1.10) !important;
    }


    /* ========================================================
       SUCH-BUTTONS INNERHALB DER GEÖFFNETEN KARTE
    ======================================================== */

    .st-key-search_open .stButton > button {
        min-height: 62px !important;

        font-size: 1.15rem !important;
    }


    /* ========================================================
       SPEICHERN
    ======================================================== */

    .st-key-save_button button {
        width: 100% !important;

        min-height: 72px !important;

        font-size: 1.35rem !important;
    }


    /* ========================================================
       FUNDSTÜCK-KARTEN
    ======================================================== */

    .st-key-item_card {
        min-height: 470px !important;

        padding: 1.5rem !important;

        background:
            linear-gradient(
                145deg,
                #8d969b,
                #697379
            ) !important;

        border: 2px solid #adb5b9 !important;

        border-radius: 20px !important;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.20),
            0 10px 25px rgba(0,0,0,0.25) !important;
    }

    .st-key-item_card h3 {
        color: white !important;

        font-size: 1.6rem !important;

        font-weight: 900 !important;
    }

    .st-key-item_card p {
        color: white !important;

        font-size: 1.1rem !important;
    }


    /* ========================================================
       BILDER
    ======================================================== */

    [data-testid="stImage"] img {
        border-radius: 14px !important;
    }


    /* ========================================================
       ALERTS
    ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 14px !important;

        font-size: 1.1rem !important;

        padding: 1rem !important;
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


# ============================================================
# DATENBANK
# ============================================================

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
                created_at TEXT DEFAULT ''
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
            "created_at": "TEXT DEFAULT ''"
        }

        for column, definition in required_columns.items():

            if column not in columns:

                conn.execute(
                    f"""
                    ALTER TABLE items
                    ADD COLUMN {column} {definition}
                    """
                )

        conn.commit()

    finally:

        conn.close()


init_database()


# ============================================================
# FUNDSTÜCK SPEICHERN
# ============================================================

def save_item(
    image,
    label,
    location,
    found_date
):

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        uuid.uuid4().hex
        + ".jpg"
    )

    image_path = (
        Path(str(UPLOAD_DIR))
        / filename
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


# ============================================================
# ÄLTESTE FUNDSTÜCKE
# ============================================================

def get_oldest_items(
    limit=9
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


# ============================================================
# SUCHE
# ============================================================

def search_items(
    query
):

    conn = get_connection()

    try:

        search_value = (
            "%"
            + query.lower()
            + "%"
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
# KI LADEN
# ============================================================

@st.cache_resource
def load_model():

    return tf_keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


# ============================================================
# KI VORHERSAGE
# ============================================================

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

    if index < len(LABELS):

        label = LABELS[index]

    else:

        label = str(index)

    return (
        label,
        confidence
    )


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

        st.session_state.search_open = False

        st.rerun()

    if st.button(
        "Bild hochladen",
        key="nav_upload",
        width="stretch"
    ):

        st.session_state.page = "Bild hochladen"

        st.session_state.search_open = False

        st.rerun()

    if st.button(
        "Älteste Fundstücke",
        key="nav_oldest",
        width="stretch"
    ):

        st.session_state.page = "Älteste Fundstücke"

        st.session_state.search_open = False

        st.rerun()


# ============================================================
# SEITENTITEL
# ============================================================

def page_title(
    title
):

    with st.container(
        border=True,
        key="page_title"
    ):

        st.title(title)


# ============================================================
# SUCHERGEBNISSE ANZEIGEN
# ============================================================

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

    st.write("")

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
                        Path(str(UPLOAD_DIR))
                        / str(filename)
                    )

                    if image_path.exists():

                        try:

                            image = Image.open(
                                str(image_path)
                            ).convert(
                                "RGB"
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
                        "Fundort: "
                        + (
                            location
                            or "Nicht angegeben"
                        )
                    )

                    st.write(
                        "Gefunden am: "
                        + (
                            found_date
                            or "Nicht angegeben"
                        )
                    )


# ============================================================
# ERSTE SEITE: SUCHE
# ============================================================

def show_search_page():

    page_title(
        "Fundgrube"
    )

    # --------------------------------------------------------
    # GESCHLOSSENE SUCHKARTE
    # --------------------------------------------------------

    if not st.session_state.search_open:

        with st.container(
            border=True,
            key="search_closed"
        ):

            if st.button(
                "Etwas verloren? Suche es.",
                key="open_search",
                width="stretch"
            ):

                st.session_state.search_open = True

                st.rerun()

        return


    # --------------------------------------------------------
    # GEÖFFNETE SUCHKARTE
    # --------------------------------------------------------

    with st.container(
        border=True,
        key="search_open"
    ):

        st.subheader(
            "Etwas verloren? Suche es."
        )

        search_text = st.text_area(
            "Suchfeld",
            placeholder=(
                "Beschreibe, was du verloren hast.\n"
                "Zum Beispiel: schwarze Hose aus der Sporthalle"
            ),
            height=220,
            key="search_text_area"
        )

        col1, col2 = st.columns(
            2,
            gap="medium"
        )

        with col1:

            search_clicked = st.button(
                "Suchen",
                key="search_button",
                width="stretch"
            )

        with col2:

            close_clicked = st.button(
                "Suche schließen",
                key="close_search",
                width="stretch"
            )

    # --------------------------------------------------------
    # SUCHE SCHLIESSEN
    # --------------------------------------------------------

    if close_clicked:

        st.session_state.search_open = False

        st.session_state.search_text_area = ""

        st.rerun()

    # --------------------------------------------------------
    # SUCHEN
    # --------------------------------------------------------

    if (
        search_clicked
        and search_text.strip()
    ):

        results = search_items(
            search_text.strip()
        )

        display_search_results(
            results
        )


# ============================================================
# BILD HOCHLADEN
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
            "Lade ein Foto hoch. "
            "Die KI erkennt automatisch den Gegenstand."
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
        ).convert(
            "RGB"
        )

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
                "Erkannt: "
                + label
                + " ("
                + f"{confidence:.0%}"
                + ")"
            )

        except Exception as error:

            st.error(
                "Das KI-Modell konnte nicht ausgeführt werden."
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

            st.subheader(
                "Noch keine Fundstücke vorhanden."
            )

            st.write(
                "Sobald ein Fundstück gespeichert wurde, "
                "wird es hier angezeigt."
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
                    key=f"item_card_{item_id}"
                ):

                    image_path = (
                        Path(str(UPLOAD_DIR))
                        / str(filename)
                    )

                    if image_path.exists():

                        try:

                            image = Image.open(
                                str(image_path)
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
                        "Fundort: "
                        + (
                            location
                            or "Nicht angegeben"
                        )
                    )

                    st.write(
                        "Gefunden am: "
                        + (
                            found_date
                            or "Nicht angegeben"
                        )
                    )


# ============================================================
# SEITE STARTEN
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
