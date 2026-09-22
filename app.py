import streamlit as st
import sqlite3
import os
import uuid
from pathlib import Path
from datetime import date, datetime

import numpy as np
from PIL import Image, ImageOps, ImageDraw

# ============================================================
# GRUNDEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"

DB_PATH = BASE_DIR / "fundgrube.db"
UPLOAD_DIR = BASE_DIR / "fundgrube_uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

# ============================================================
# FARBEN
# ============================================================

DARK_BLUE = "#12384D"
DARK_BLUE_2 = "#0D2E40"

METALLIC = "#727B82"
METALLIC_DARK = "#5E686F"
METALLIC_LIGHT = "#8B949A"

WHITE = "#FFFFFF"
BLACK = "#111111"

# ============================================================
# CSS
#
# Wichtig:
# Hier wird ausschließlich CSS geladen.
# Es wird KEIN HTML als sichtbarer Inhalt der App verwendet.
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GESAMTER HINTERGRUND
       -------------------------------------------------------- */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #102F42 0%,
                #12394F 35%,
                #17465D 65%,
                #0D2E40 100%
            ) !important;
        color: white !important;
    }

    [data-testid="stAppViewContainer"] {
        background:
            linear-gradient(
                135deg,
                #102F42 0%,
                #12394F 35%,
                #17465D 65%,
                #0D2E40 100%
            ) !important;
    }

    [data-testid="stMain"] {
        background: transparent !important;
    }

    /* --------------------------------------------------------
       NORMALE SCHRIFT
       -------------------------------------------------------- */

    p,
    label,
    span,
    div {
        font-family: Arial, Helvetica, sans-serif;
    }

    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #58646C 0%,
                #4D5961 50%,
                #465159 100%
            ) !important;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] [data-testid="stButton"] button {
        background: linear-gradient(
            145deg,
            #7D878E,
            #616B72
        ) !important;

        color: white !important;
        border: 1px solid #8F989E !important;
        border-radius: 12px !important;
        font-weight: 600 !important;

        box-shadow:
            0 4px 10px rgba(0,0,0,0.30) !important;

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease !important;
    }

    section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        transform: scale(1.04) !important;

        box-shadow:
            0 7px 18px rgba(0,0,0,0.42) !important;
    }

    /* --------------------------------------------------------
       HAUPT-INHALT
       -------------------------------------------------------- */

    .block-container {
        max-width: 1250px !important;
        padding-top: 3rem !important;
        padding-bottom: 4rem !important;
    }

    /* --------------------------------------------------------
       GRAUE KARTEN
       -------------------------------------------------------- */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(
                145deg,
                #858E94,
                #69737A
            ) !important;

        border: 1px solid #A2A9AD !important;

        border-radius: 18px !important;

        box-shadow:
            0 7px 18px rgba(0,0,0,0.28) !important;
    }

    /* --------------------------------------------------------
       TITEL IN GRAUER KARTE
       -------------------------------------------------------- */

    .page-title {
        color: white !important;
        font-size: 3.3rem !important;
        font-weight: 900 !important;
        letter-spacing: -1px !important;
        margin-bottom: 0.25rem !important;
    }

    .page-subtitle {
        color: white !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }

    /* --------------------------------------------------------
       STARTSEITEN-TITEL
       -------------------------------------------------------- */

    .main-title {
        color: white !important;
        font-size: 4.4rem !important;
        font-weight: 900 !important;
        letter-spacing: -2px !important;
        text-align: center !important;
        margin: 0 !important;
    }

    .main-subtitle {
        color: white !important;
        font-size: 1.15rem !important;
        text-align: center !important;
        margin-top: 0.2rem !important;
    }

    /* --------------------------------------------------------
       LUPEN-BEREICH
       -------------------------------------------------------- */

    .magnifier {
        text-align: center !important;
        font-size: 6.5rem !important;
        line-height: 1 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.7rem !important;
        filter: grayscale(1) brightness(2) !important;
    }

    /* --------------------------------------------------------
       SUCHKARTE
       -------------------------------------------------------- */

    .st-key-search_card {
        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease !important;
    }

    .st-key-search_card:hover {
        transform: scale(1.045) !important;

        box-shadow:
            0 12px 30px rgba(0,0,0,0.40) !important;
    }

    /* Suchfeld zunächst zusammenklappen */
    .st-key-search_card [data-testid="stTextInput"] {
        max-height: 0px !important;
        opacity: 0 !important;
        overflow: hidden !important;
        margin-top: 0px !important;
        transition:
            max-height 0.25s ease,
            opacity 0.2s ease,
            margin-top 0.25s ease !important;
    }

    /* Beim Darüberfahren erscheint es */
    .st-key-search_card:hover [data-testid="stTextInput"] {
        max-height: 100px !important;
        opacity: 1 !important;
        margin-top: 1rem !important;
        overflow: visible !important;
    }

    /* --------------------------------------------------------
       TEXT IM SUCHFELD
       -------------------------------------------------------- */

    .st-key-search_card input {
        background: white !important;
        color: black !important;

        border: 2px solid #D9DDE0 !important;
        border-radius: 10px !important;

        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }

    .st-key-search_card input::placeholder {
        color: #737373 !important;
        opacity: 1 !important;
    }

    /* --------------------------------------------------------
       ALLE INPUTS
       -------------------------------------------------------- */

    input,
    textarea {
        background: white !important;
        color: black !important;
        border-radius: 10px !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #777777 !important;
    }

    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    .stButton button {
        background:
            linear-gradient(
                145deg,
                #858E94,
                #626B72
            ) !important;

        color: white !important;

        border: 1px solid #A5ACB0 !important;
        border-radius: 11px !important;

        font-weight: 700 !important;

        box-shadow:
            0 4px 10px rgba(0,0,0,0.25) !important;

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease !important;
    }

    .stButton button:hover {
        transform: scale(1.035) !important;

        box-shadow:
            0 8px 18px rgba(0,0,0,0.35) !important;
    }

    /* --------------------------------------------------------
       DATE INPUT
       -------------------------------------------------------- */

    [data-testid="stDateInput"] input {
        background: white !important;
        color: black !important;
    }

    /* --------------------------------------------------------
       SELECTBOX
       -------------------------------------------------------- */

    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background: white !important;
        color: black !important;
    }

    [data-testid="stSelectbox"] * {
        color: black !important;
    }

    /* --------------------------------------------------------
       INFO / HINWEISE
       -------------------------------------------------------- */

    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* --------------------------------------------------------
       FILE UPLOADER
       -------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background:
            linear-gradient(
                145deg,
                #858E94,
                #69737A
            ) !important;

        border-radius: 15px !important;
        padding: 0.5rem !important;
    }

    [data-testid="stFileUploader"] section {
        background: #F7F8F9 !important;
        border-radius: 11px !important;
        border: 1px solid #B9BEC2 !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #222222 !important;
    }

    [data-testid="stFileUploader"] button {
        color: #222222 !important;
        background: white !important;
    }

    /* --------------------------------------------------------
       BILDER
       -------------------------------------------------------- */

    [data-testid="stImage"] img {
        border-radius: 12px !important;
    }

    /* --------------------------------------------------------
       KARTEN FÜR FUNDSTÜCKE
       -------------------------------------------------------- */

    .item-title {
        color: white !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
    }

    .item-info {
        color: white !important;
        font-size: 0.95rem !important;
    }

    /* --------------------------------------------------------
       HORIZONTALE TRENNLINIEN
       -------------------------------------------------------- */

    hr {
        border-color: rgba(255,255,255,0.25) !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATENBANK
# ============================================================

def get_connection():
    """
    Öffnet die SQLite-Datenbank.
    """
    connection = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False,
    )

    return connection


def initialize_database():
    """
    Erstellt die Datenbank und repariert ältere Versionen.

    Dadurch sollte der Fehler auf der Seite
    'Älteste Fundstücke' nicht mehr auftreten, wenn bereits
    eine ältere fundgrube.db existiert.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            filename TEXT,
            label TEXT,
            location TEXT,
            found_date TEXT,
            created_at TEXT
        )
        """
    )

    # Vorhandene Spalten auslesen
    cursor.execute("PRAGMA table_info(items)")
    existing_columns = {
        row[1]
        for row in cursor.fetchall()
    }

    # Falls eine ältere Datenbank verwendet wird,
    # fehlende Spalten automatisch ergänzen.
    required_columns = {
        "filename": "TEXT",
        "label": "TEXT",
        "location": "TEXT",
        "found_date": "TEXT",
        "created_at": "TEXT",
    }

    for column_name, column_type in required_columns.items():
        if column_name not in existing_columns:
            try:
                cursor.execute(
                    f"ALTER TABLE items ADD COLUMN {column_name} {column_type}"
                )
            except sqlite3.OperationalError:
                pass

    connection.commit()
    connection.close()


initialize_database()

# ============================================================
# LABELS LADEN
# ============================================================

def load_labels():
    """
    Lädt labels.txt.

    Unterstützt z.B.

    0 Hose
    1 Schuh
    2 T-Shirt
    3 Hoodie
    """

    if not LABELS_PATH.exists():
        return ["Hose", "Schuh", "T-Shirt", "Hoodie"]

    labels = []

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                parts = line.split(maxsplit=1)

                if len(parts) == 2:
                    labels.append(parts[1].strip())
                else:
                    labels.append(line)

    except Exception:
        return ["Hose", "Schuh", "T-Shirt", "Hoodie"]

    return labels


LABELS = load_labels()

# ============================================================
# KI-MODELL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None, (
            f"Die Datei {MODEL_PATH.name} wurde nicht gefunden."
        )

    # Erst tf_keras versuchen.
    try:
        import tf_keras

        model = tf_keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

        return model, None

    except Exception as first_error:

        # Falls tf_keras nicht funktioniert,
        # als zweite Möglichkeit TensorFlow/Keras versuchen.
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
            )

            return model, None

        except Exception as second_error:

            return None, (
                "Das KI-Modell konnte nicht geladen werden.\n\n"
                f"Erster Versuch: {first_error}\n\n"
                f"Zweiter Versuch: {second_error}"
            )


# ============================================================
# KI-VORBEREITUNG
# ============================================================

def prepare_image(image):
    """
    Bereitet das Bild für das Teachable-Machine-Modell vor.
    """

    image = image.convert("RGB")
    image = image.resize((224, 224))

    array = np.asarray(image).astype(np.float32)

    # Teachable Machine Image Models verwenden normalerweise
    # eine Skalierung von 0-255 auf -1 bis +1.
    array = (array / 127.5) - 1.0

    array = np.expand_dims(array, axis=0)

    return array


def classify_image(image):
    """
    Klassifiziert ein Bild mit dem trainierten Modell.
    """

    model, error = load_model()

    if model is None:
        return None, 0.0, error

    try:
        input_array = prepare_image(image)

        prediction = model.predict(
            input_array,
            verbose=0,
        )

        prediction = np.asarray(prediction)

        # Falls das Modell eine verschachtelte Ausgabe liefert
        if prediction.ndim > 1:
            prediction = prediction[0]

        prediction = prediction.astype(float)

        # Manche Modelle liefern Logits statt Wahrscheinlichkeiten.
        if (
            np.any(prediction < 0)
            or not np.isclose(np.sum(prediction), 1.0, atol=0.05)
        ):
            exp_values = np.exp(
                prediction - np.max(prediction)
            )

            prediction = (
                exp_values / np.sum(exp_values)
            )

        index = int(np.argmax(prediction))

        confidence = float(prediction[index])

        if index < len(LABELS):
            label = LABELS[index]
        else:
            label = f"Klasse {index}"

        return label, confidence, None

    except Exception as error:
        return None, 0.0, str(error)


# ============================================================
# DATENBANK-FUNKTIONEN
# ============================================================

def save_item(
    image,
    label,
    location,
    found_date,
):
    """
    Speichert Bild + Informationen.
    """

    item_id = str(uuid.uuid4())

    extension = ".jpg"

    filename = f"{item_id}{extension}"

    image_path = UPLOAD_DIR / filename

    # RGB sicherstellen
    image = image.convert("RGB")

    image.save(
        image_path,
        format="JPEG",
        quality=92,
    )

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO items
        (
            id,
            filename,
            label,
            location,
            found_date,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            item_id,
            filename,
            label,
            location,
            str(found_date),
            datetime.now().isoformat(),
        ),
    )

    connection.commit()
    connection.close()


def get_oldest_items(limit=9):
    """
    Liefert die ältesten gespeicherten Fundstücke.

    Wichtig:
    Die Abfrage verwendet nur Spalten, die initialize_database()
    garantiert angelegt hat.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                filename,
                label,
                location,
                found_date,
                created_at
            FROM items
            WHERE filename IS NOT NULL
              AND filename != ''
            ORDER BY
                CASE
                    WHEN found_date IS NULL
                    OR found_date = ''
                    THEN created_at
                    ELSE found_date
                END ASC,
                created_at ASC
            LIMIT ?
            """,
            (int(limit),),
        )

        rows = cursor.fetchall()

        return rows

    finally:
        connection.close()


def search_items(search_text):
    """
    Sucht nach Fundstücken.
    """

    search_text = search_text.strip()

    if not search_text:
        return []

    connection = get_connection()

    try:
        cursor = connection.cursor()

        pattern = f"%{search_text}%"

        cursor.execute(
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
                LOWER(label) LIKE LOWER(?)
                OR LOWER(location) LIKE LOWER(?)
            ORDER BY
                found_date ASC,
                created_at ASC
            """,
            (
                pattern,
                pattern,
            ),
        )

        return cursor.fetchall()

    finally:
        connection.close()


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
        """
        <div style="
            color:white;
            font-size:1.35rem;
            font-weight:900;
            margin-bottom:1.2rem;
        ">
        Fundgrube
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button(
        "🔎 Suche",
        width="stretch",
        key="nav_search",
    ):
        st.session_state.page = "Suche"
        st.rerun()

    if st.button(
        "📷 Bild hochladen",
        width="stretch",
        key="nav_upload",
    ):
        st.session_state.page = "Upload"
        st.rerun()

    if st.button(
        "◷ Älteste Fundstücke",
        width="stretch",
        key="nav_oldest",
    ):
        st.session_state.page = "Älteste"
        st.rerun()

    st.divider()

    st.caption("Fundgrube")


# ============================================================
# TITEL-KARTE
# ============================================================

def page_header(title, subtitle):
    """
    Einheitliche graue Titelkarte für alle Seiten.
    """

    with st.container(
        border=True,
        key=f"header_{title}",
    ):

        st.markdown(
            f"""
            <div class="page-title">
                {title}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="page-subtitle">
                {subtitle}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SEITE 1: SUCHE
# ============================================================

def show_search_page():

    # Großer grauer Titel
    with st.container(
        border=True,
        key="main_header",
    ):

        st.markdown(
            """
            <div class="main-title">
                Fundgrube
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="main-subtitle">
                Schul-Fundstücke einfach wiederfinden
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Lupe mittig
    st.markdown(
        """
        <div class="magnifier">
            🔎
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # --------------------------------------------------------
    # GRAUES INTERAKTIVES SUCHFELD
    # --------------------------------------------------------

    with st.container(
        border=True,
        key="search_card",
    ):

        st.markdown(
            """
            <div style="
                color:white;
                font-size:2rem;
                font-weight:900;
                line-height:1.2;
            ">
                Etwas verloren? Suche es.
            </div>
            """,
            unsafe_allow_html=True,
        )

        search_text = st.text_input(
            "Suchbegriff",
            placeholder="Suchbegriff eingeben, z. B. Hoodie, Hose oder Schuh ...",
            label_visibility="collapsed",
            key="search_input",
        )

    st.write("")

    # --------------------------------------------------------
    # SUCHERGEBNISSE
    # --------------------------------------------------------

    if search_text.strip():

        results = search_items(search_text)

        if results:

            st.markdown(
                f"""
                <div style="
                    color:white;
                    font-size:1.4rem;
                    font-weight:800;
                    margin:1rem 0;
                ">
                    Gefundene Fundstücke
                </div>
                """,
                unsafe_allow_html=True,
            )

            columns = st.columns(3)

            for index, item in enumerate(results):

                (
                    item_id,
                    filename,
                    label,
                    location,
                    found_date,
                    created_at,
                ) = item

                image_path = UPLOAD_DIR / filename

                with columns[index % 3]:

                    with st.container(
                        border=True,
                        key=f"search_item_{item_id}",
                    ):

                        if image_path.exists():

                            try:
                                image = Image.open(
                                    image_path
                                )

                                st.image(
                                    image,
                                    width="stretch",
                                )

                            except Exception:
                                st.warning(
                                    "Bild konnte nicht geladen werden."
                                )

                        st.markdown(
                            f"""
                            <div class="item-title">
                                {label}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"""
                            <div class="item-info">
                                📍 {location}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"""
                            <div class="item-info">
                                📅 {found_date}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        else:

            with st.container(
                border=True,
                key="no_results",
            ):
                st.write(
                    "Es wurde kein passendes Fundstück gefunden."
                )

    else:

        st.info(
            "Fahre mit der Maus über das graue Feld. "
            "Dann erscheint die Suchleiste."
        )


# ============================================================
# SEITE 2: UPLOAD
# ============================================================

def show_upload_page():

    page_header(
        "📷 Bild hochladen",
        "Lade ein Foto hoch und lasse es von der KI erkennen.",
    )

    st.write("")

    # --------------------------------------------------------
    # UPLOADER
    # --------------------------------------------------------

    with st.container(
        border=True,
        key="upload_container",
    ):

        st.subheader(
            "Foto aus deiner Mediathek auswählen"
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
            key="uploaded_image",
        )

    # --------------------------------------------------------
    # NOCH KEIN BILD
    # --------------------------------------------------------

    if uploaded_file is None:

        st.info(
            "Wähle ein Bild aus, um die KI-Klassifizierung zu starten."
        )

        return

    # --------------------------------------------------------
    # BILD ÖFFNEN
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as error:

        st.error(
            f"Das Bild konnte nicht geöffnet werden: {error}"
        )

        return

    # --------------------------------------------------------
    # BILD ANZEIGEN
    # --------------------------------------------------------

    with st.container(
        border=True,
        key="preview_container",
    ):

        st.subheader("Ausgewähltes Bild")

        st.image(
            image,
            width="stretch",
        )

    # --------------------------------------------------------
    # KI-KLASSIFIZIERUNG
    # --------------------------------------------------------

    with st.container(
        border=True,
        key="classification_container",
    ):

        st.subheader("KI-Erkennung")

        progress = st.progress(
            0,
            text="KI wird gestartet ...",
        )

        progress.progress(
            30,
            text="Bild wird vorbereitet ...",
        )

        label, confidence, error = classify_image(
            image
        )

        progress.progress(
            100,
            text="Erkennung abgeschlossen.",
        )

        if error:

            st.error(
                "Das Bild konnte nicht klassifiziert werden."
            )

            st.code(
                error
            )

            return

        st.success(
            f"Erkannt: {label}"
        )

        st.write(
            f"Übereinstimmung: {confidence * 100:.1f} %"
        )

    # --------------------------------------------------------
    # INFORMATIONEN ZUM FUNDSTÜCK
    # --------------------------------------------------------

    with st.container(
        border=True,
        key="details_container",
    ):

        st.subheader(
            "Wo und wann wurde das Fundstück gefunden?"
        )

        location = st.text_input(
            "Fundort",
            placeholder="z. B. Sporthalle, Klassenraum 204 ...",
            key="found_location",
        )

        found_date = st.date_input(
            "Funddatum",
            value=date.today(),
            key="found_date",
        )

        save_button = st.button(
            "Fundstück speichern",
            type="primary",
            width="stretch",
            key="save_item",
        )

        if save_button:

            if not location.strip():

                st.warning(
                    "Bitte gib zuerst den Fundort ein."
                )

            else:

                try:

                    save_item(
                        image=image,
                        label=label,
                        location=location.strip(),
                        found_date=found_date,
                    )

                    st.success(
                        "Das Fundstück wurde erfolgreich gespeichert."
                    )

                    st.balloons()

                except Exception as error:

                    st.error(
                        "Das Fundstück konnte nicht gespeichert werden."
                    )

                    st.code(
                        str(error)
                    )


# ============================================================
# SEITE 3: ÄLTESTE FUNDSTÜCKE
# ============================================================

def show_oldest_page():

    page_header(
        "◷ Älteste Fundstücke",
        "Die neun ältesten gespeicherten Fundstücke.",
    )

    st.write("")

    # --------------------------------------------------------
    # DATENBANK ABFRAGEN
    # --------------------------------------------------------

    try:

        items = get_oldest_items(9)

    except Exception as error:

        st.error(
            "Die Fundstücke konnten nicht geladen werden."
        )

        st.code(
            str(error)
        )

        return

    # --------------------------------------------------------
    # KEINE FUNDSTÜCKE
    # --------------------------------------------------------

    if not items:

        with st.container(
            border=True,
            key="empty_items",
        ):

            st.info(
                "Es wurden bisher noch keine Fundstücke gespeichert."
            )

        return

    # --------------------------------------------------------
    # 9 FUNDSTÜCKE
    # --------------------------------------------------------

    columns = st.columns(3)

    for index, item in enumerate(items):

        (
            item_id,
            filename,
            label,
            location,
            found_date,
            created_at,
        ) = item

        image_path = UPLOAD_DIR / filename

        with columns[index % 3]:

            with st.container(
                border=True,
                key=f"oldest_item_{item_id}",
            ):

                # Bild
                if image_path.exists():

                    try:

                        image = Image.open(
                            image_path
                        ).convert("RGB")

                        # Quadratisches Vorschaubild
                        image = ImageOps.fit(
                            image,
                            (500, 500),
                            method=Image.Resampling.LANCZOS,
                        )

                        st.image(
                            image,
                            width="stretch",
                        )

                    except Exception:

                        st.warning(
                            "Bild konnte nicht geladen werden."
                        )

                else:

                    st.info(
                        "Kein Bild vorhanden."
                    )

                # Bezeichnung
                st.markdown(
                    f"""
                    <div class="item-title">
                        {label}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Ort
                st.markdown(
                    f"""
                    <div class="item-info">
                        📍 {location}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Datum
                st.markdown(
                    f"""
                    <div class="item-info">
                        📅 {found_date}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# SEITE AUSWÄHLEN
# ============================================================

if st.session_state.page == "Suche":

    show_search_page()

elif st.session_state.page == "Upload":

    show_upload_page()

elif st.session_state.page == "Älteste":

    show_oldest_page()

else:

    st.session_state.page = "Suche"

    show_search_page()
