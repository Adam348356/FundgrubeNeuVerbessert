import streamlit as st
import tensorflow as tf
import tf_keras
import numpy as np

from PIL import Image
from pathlib import Path
import sqlite3
import uuid
from datetime import date, datetime


# ============================================================
# EINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# DATEIPFADE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"

UPLOAD_DIR = BASE_DIR / "fundgrube_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DATABASE_PATH = BASE_DIR / "fundgrube.db"


# ============================================================
# FARBEN
# ============================================================

ROSA = "#F1878D"
BLAU = "#9BDCF5"
WEISS = "#FFFFFF"
SCHWARZ = "#000000"


# ============================================================
# ALLGEMEINES DESIGN
# ============================================================

st.markdown(
    f"""
    <style>

    /* --------------------------------------------------------
       GRUNDLAYOUT
    -------------------------------------------------------- */

    .stApp {{
        background: white;
    }}

    .block-container {{
        padding-top: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
    }}

    header {{
        background: transparent !important;
    }}

    /* --------------------------------------------------------
       STREAMLIT MENU / FOOTER AUSBLENDEN
    -------------------------------------------------------- */

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    /* --------------------------------------------------------
       HAMBURGER
    -------------------------------------------------------- */

    .hamburger-button {{
        position: absolute;
        top: 18px;
        left: 18px;
        z-index: 20;
        font-size: 42px;
        font-weight: 400;
        line-height: 1;
        color: black;
    }}

    /* --------------------------------------------------------
       TITEL
    -------------------------------------------------------- */

    .page-title {{
        color: black;
        font-size: clamp(48px, 7vw, 92px);
        font-weight: 800;
        line-height: 1;
        text-align: center;
        margin: 0;
        padding-top: 32px;
        letter-spacing: -3px;
    }}

    .upload-title {{
        color: black;
        font-size: clamp(42px, 6vw, 82px);
        font-weight: 800;
        line-height: 1;
        text-align: center;
        margin: 0;
        padding-top: 30px;
        letter-spacing: -3px;
    }}

    /* --------------------------------------------------------
       HEADER
    -------------------------------------------------------- */

    .header-area {{
        width: 100%;
        position: relative;
        overflow: hidden;
    }}

    .header-pink {{
        background: {ROSA};
    }}

    .header-blue {{
        background: {BLAU};
    }}

    .school-svg {{
        width: 100%;
        display: block;
        margin-top: 15px;
    }}

    /* --------------------------------------------------------
       SUCHBEREICH
    -------------------------------------------------------- */

    .search-area {{
        background: white;
        text-align: center;
        padding-top: 25px;
    }}

    .magnifying-glass {{
        width: 180px;
        height: 180px;
        margin: 0 auto 15px auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }}

    .search-description {{
        display: inline-block;
        border: 3px solid black;
        border-radius: 40px;
        padding: 8px 30px;
        font-size: clamp(20px, 2.2vw, 31px);
        color: black;
        background: white;
        margin-bottom: 20px;
    }}

    /* --------------------------------------------------------
       UPLOADBEREICH
    -------------------------------------------------------- */

    .upload-area {{
        background: white;
        padding-top: 35px;
        text-align: center;
    }}

    .upload-box {{
        width: min(800px, 80%);
        min-height: 340px;
        margin: 0 auto;
        border: 4px dashed black;
        border-radius: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 40px;
        background: white;
    }}

    .upload-text {{
        display: inline-block;
        border: 3px solid black;
        border-radius: 28px;
        padding: 14px 35px;
        font-size: clamp(18px, 2vw, 28px);
        color: black;
        background: white;
    }}

    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton > button {{
        border: 3px solid black !important;
        border-radius: 30px !important;
        background: white !important;
        color: black !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        min-height: 50px !important;
    }}

    .stButton > button:hover {{
        background: #eeeeee !important;
        color: black !important;
        border-color: black !important;
    }}

    /* --------------------------------------------------------
       TEXT INPUT
    -------------------------------------------------------- */

    div[data-baseweb="input"] {{
        border: 3px solid black !important;
        border-radius: 35px !important;
        background: white !important;
    }}

    div[data-baseweb="input"] input {{
        color: black !important;
        font-size: 20px !important;
        text-align: center !important;
    }}

    /* --------------------------------------------------------
       FILE UPLOADER
    -------------------------------------------------------- */

    section[data-testid="stFileUploader"] {{
        width: min(800px, 85%);
        margin: 0 auto;
    }}

    section[data-testid="stFileUploader"] > div {{
        border: 4px dashed black !important;
        border-radius: 30px !important;
        background: white !important;
        padding: 20px !important;
    }}

    /* --------------------------------------------------------
       KARTEN
    -------------------------------------------------------- */

    .item-card {{
        background: white;
        border: 3px solid black;
        border-radius: 22px;
        padding: 12px;
        margin-bottom: 25px;
        overflow: hidden;
    }}

    .item-title {{
        color: black;
        font-size: 22px;
        font-weight: 700;
        margin-top: 10px;
    }}

    .item-info {{
        color: black;
        font-size: 16px;
        line-height: 1.5;
        margin-top: 5px;
    }}

    /* --------------------------------------------------------
       KLASSIFIZIERUNG
    -------------------------------------------------------- */

    .classification-box {{
        width: min(800px, 85%);
        margin: 25px auto;
        border: 3px solid black;
        border-radius: 25px;
        padding: 25px;
        text-align: center;
        background: white;
    }}

    .classification-title {{
        font-size: 25px;
        font-weight: 700;
        color: black;
    }}

    .classification-result {{
        font-size: 35px;
        font-weight: 800;
        color: black;
        margin-top: 10px;
    }}

    /* --------------------------------------------------------
       SEITENTITEL
    -------------------------------------------------------- */

    .section-title {{
        color: black;
        text-align: center;
        font-size: clamp(35px, 5vw, 60px);
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 10px;
    }}

    .section-subtitle {{
        color: black;
        text-align: center;
        font-size: 20px;
        margin-bottom: 30px;
    }}

    /* --------------------------------------------------------
       MOBILE
    -------------------------------------------------------- */

    @media (max-width: 700px) {{

        .page-title {{
            padding-top: 65px;
            font-size: 48px;
        }}

        .upload-title {{
            padding-top: 65px;
            font-size: 40px;
        }}

        .magnifying-glass {{
            width: 140px;
            height: 140px;
        }}

        .upload-box {{
            min-height: 260px;
            width: 85%;
        }}

        .search-description {{
            margin-left: 15px;
            margin-right: 15px;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SCHUL-SILHOUETTE
# ============================================================

def school_silhouette(background_color):
    """
    Zeichnet die Schulsilhouette aus den Skizzen direkt als SVG.
    Dadurch wird keine zusätzliche Bilddatei benötigt.
    """

    st.markdown(
        f"""
        <svg
            class="school-svg"
            viewBox="0 0 1500 320"
            preserveAspectRatio="none"
            xmlns="http://www.w3.org/2000/svg"
        >

            <!-- Hintergrund -->
            <rect
                x="0"
                y="0"
                width="1500"
                height="320"
                fill="{background_color}"
            />

            <!-- weiße Grundfläche -->
            <path
                d="
                M 0 285
                L 250 285

                L 250 145
                L 285 145
                L 320 105
                L 355 145
                L 430 145
                L 430 285

                L 620 285

                L 620 110
                L 690 110
                L 760 30
                L 830 110
                L 900 110
                L 900 285

                L 1080 285

                L 1080 145
                L 1150 145
                L 1185 105
                L 1220 145
                L 1295 145
                L 1295 285

                L 1500 285
                L 1500 320
                L 0 320
                Z
                "
                fill="white"
            />

            <!-- Außenlinien der Gebäude -->

            <!-- linkes Gebäude -->
            <path
                d="
                M 250 285
                L 250 145
                L 285 145
                L 320 105
                L 355 145
                L 430 145
                L 430 285
                "
                fill="none"
                stroke="black"
                stroke-width="5"
            />

            <!-- mittleres Gebäude -->
            <path
                d="
                M 620 285
                L 620 110
                L 690 110
                L 760 30
                L 830 110
                L 900 110
                L 900 285
                "
                fill="none"
                stroke="black"
                stroke-width="5"
            />

            <!-- rechtes Gebäude -->
            <path
                d="
                M 1080 285
                L 1080 145
                L 1150 145
                L 1185 105
                L 1220 145
                L 1295 145
                L 1295 285
                "
                fill="none"
                stroke="black"
                stroke-width="5"
            />

            <!-- Dächer -->

            <path
                d="M 270 220 L 320 155 L 410 220"
                fill="none"
                stroke="black"
                stroke-width="3"
            />

            <path
                d="M 650 220 L 760 90 L 870 220"
                fill="none"
                stroke="black"
                stroke-width="3"
            />

            <path
                d="M 1100 220 L 1185 155 L 1275 220"
                fill="none"
                stroke="black"
                stroke-width="3"
            />

            <!-- Fenster linkes Gebäude -->

            <line
                x1="320"
                y1="155"
                x2="320"
                y2="275"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="360"
                y1="185"
                x2="360"
                y2="275"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="395"
                y1="185"
                x2="395"
                y2="275"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="285"
                y1="220"
                x2="410"
                y2="220"
                stroke="black"
                stroke-width="3"
            />

            <!-- Fenster mittleres Gebäude -->

            <line
                x1="760"
                y1="90"
                x2="760"
                y2="275"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="705"
                y1="155"
                x2="705"
                y2="275"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="815"
                y1="155"
                x2="815"
                y2="275"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="670"
                y1="205"
                x2="850"
                y2="205"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="700"
                y1="155"
                x2="820"
                y2="155"
                stroke="black"
                stroke-width="3"
            />

            <!-- Fenster rechtes Gebäude -->

            <line
                x1="1185"
                y1="155"
                x2="1185"
                y2="275"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="1145"
                y1="190"
                x2="1145"
                y2="275"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="1230"
                y1="190"
                x2="1230"
                y2="275"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="1110"
                y1="220"
                x2="1270"
                y2="220"
                stroke="black"
                stroke-width="3"
            />

            <!-- Bodenlinie -->
            <line
                x1="0"
                y1="285"
                x2="1500"
                y2="285"
                stroke="black"
                stroke-width="5"
            />

        </svg>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LABELS
# ============================================================

def load_labels():

    labels = []

    if not LABELS_PATH.exists():

        st.error(
            "Die Datei labels.txt wurde nicht gefunden."
        )

        return labels

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(" ", 1)

            if (
                len(parts) == 2
                and parts[0].isdigit()
            ):
                labels.append(parts[1])

            else:
                labels.append(line)

    return labels


LABELS = load_labels()


# ============================================================
# MODELL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "keras_model.h5 wurde nicht gefunden."
        )

    model = tf_keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# DATENBANK
# ============================================================

def get_connection():

    return sqlite3.connect(
        DATABASE_PATH
    )


def init_database():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            item_type TEXT NOT NULL,
            location TEXT NOT NULL,
            found_date TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


init_database()


# ============================================================
# BILD VORBEREITEN
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.asarray(
        image
    ).astype(
        np.float32
    )

    image_array = (
        image_array / 127.5
    ) - 1.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# KLASSIFIZIERUNG
# ============================================================

def classify_image(image):

    model = load_model()

    prepared_image = prepare_image(
        image
    )

    prediction = model.predict(
        prepared_image,
        verbose=0
    )

    prediction = np.asarray(
        prediction
    ).reshape(-1)

    best_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[best_index]
    )

    if best_index < len(LABELS):

        label = LABELS[best_index]

    else:

        label = f"Klasse {best_index}"

    return label, confidence


# ============================================================
# FUNDSTÜCK SPEICHERN
# ============================================================

def save_item(
    image,
    item_type,
    location,
    found_date
):

    unique_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}.jpg"
    )

    image_path = (
        UPLOAD_DIR / unique_name
    )

    image = image.convert("RGB")

    image.save(
        image_path,
        format="JPEG",
        quality=90
    )

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO items
        (
            filename,
            item_type,
            location,
            found_date,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            unique_name,
            item_type,
            location,
            str(found_date),
            datetime.now().isoformat()
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# FUNDSTÜCKE LADEN
# ============================================================

def get_items(
    search_term=None,
    limit=None
):

    connection = get_connection()
    cursor = connection.cursor()

    if search_term:

        cursor.execute(
            """
            SELECT
                id,
                filename,
                item_type,
                location,
                found_date,
                created_at
            FROM items
            WHERE item_type LIKE ?
            ORDER BY found_date DESC
            """,
            (
                f"%{search_term}%",
            )
        )

    else:

        query = """
            SELECT
                id,
                filename,
                item_type,
                location,
                found_date,
                created_at
            FROM items
            ORDER BY found_date ASC
        """

        if limit:

            query += (
                f" LIMIT {int(limit)}"
            )

        cursor.execute(query)

    items = cursor.fetchall()

    connection.close()

    return items


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "Suche"


# ============================================================
# HAMBURGER-MENÜ
# ============================================================

def show_menu():

    menu_left, menu_right = st.columns(
        [8, 1]
    )

    with menu_right:

        with st.popover(
            "☰",
            use_container_width=False
        ):

            st.markdown(
                "### Menü"
            )

            if st.button(
                "🔎 Suche",
                use_container_width=True
            ):

                st.session_state.page = "Suche"
                st.rerun()

            if st.button(
                "📤 Bild hochladen",
                use_container_width=True
            ):

                st.session_state.page = "Hochladen"
                st.rerun()

            if st.button(
                "🕐 Älteste Fundstücke",
                use_container_width=True
            ):

                st.session_state.page = (
                    "Älteste Fundstücke"
                )

                st.rerun()


# ============================================================
# SEITE 1 – SUCHE
# ============================================================

if st.session_state.page == "Suche":

    # --------------------------------------------------------
    # ROSA HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="header-area header-pink">

            <div class="page-title">
                Fundgrube
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    show_menu()

    school_silhouette(
        ROSA
    )

    # --------------------------------------------------------
    # LUPE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="search-area">

            <div class="magnifying-glass">

                <svg
                    width="180"
                    height="180"
                    viewBox="0 0 180 180"
                    xmlns="http://www.w3.org/2000/svg"
                >

                    <circle
                        cx="75"
                        cy="75"
                        r="48"
                        fill="white"
                        stroke="black"
                        stroke-width="7"
                    />

                    <circle
                        cx="75"
                        cy="75"
                        r="38"
                        fill="white"
                        stroke="black"
                        stroke-width="3"
                    />

                    <line
                        x1="108"
                        y1="108"
                        x2="145"
                        y2="145"
                        stroke="black"
                        stroke-width="12"
                        stroke-linecap="round"
                    />

                    <line
                        x1="113"
                        y1="113"
                        x2="145"
                        y2="145"
                        stroke="white"
                        stroke-width="3"
                        stroke-linecap="round"
                    />

                </svg>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SUCHTEXT
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            text-align:center;
            color:black;
            font-size:20px;
            margin-bottom:8px;
        ">
            Beschreibe dein verlorenes Kleidungsstück
        </div>
        """,
        unsafe_allow_html=True
    )

    search = st.text_input(
        "",
        placeholder="z. B. schwarzer Hoodie, blauer Schuh ...",
        label_visibility="collapsed"
    )

    # --------------------------------------------------------
    # ERGEBNISSE
    # --------------------------------------------------------

    if search.strip():

        items = get_items(
            search_term=search.strip()
        )

        if items:

            st.markdown(
                """
                <div class="section-title">
                    Gefundene Sachen
                </div>
                """,
                unsafe_allow_html=True
            )

            columns = st.columns(3)

            for index, item in enumerate(items):

                (
                    item_id,
                    filename,
                    item_type,
                    location,
                    found_date,
                    created_at
                ) = item

                image_path = (
                    UPLOAD_DIR / filename
                )

                with columns[
                    index % 3
                ]:

                    st.markdown(
                        '<div class="item-card">',
                        unsafe_allow_html=True
                    )

                    if image_path.exists():

                        st.image(
                            str(image_path),
                            use_container_width=True
                        )

                    st.markdown(
                        f"""
                        <div class="item-title">
                            {item_type}
                        </div>

                        <div class="item-info">
                            📍 {location}<br>
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
                "Leider wurde kein passendes Fundstück gefunden."
            )


# ============================================================
# SEITE 2 – HOCHLADEN
# ============================================================

elif st.session_state.page == "Hochladen":

    # --------------------------------------------------------
    # BLAUER HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="header-area header-blue">

            <div class="upload-title">
                Lade ein Bild hoch
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    show_menu()

    school_silhouette(
        BLAU
    )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="upload-area">

            <div class="upload-text">
                Lade ein Foto aus deiner Mediathek hoch
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<br>",
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
        label_visibility="collapsed"
    )

    # --------------------------------------------------------
    # BILD WURDE AUSGEWÄHLT
    # --------------------------------------------------------

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        st.image(
            image,
            caption="Dein hochgeladenes Bild",
            use_container_width=True
        )

        # ----------------------------------------------------
        # KI
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-title">
                KI erkennt dein Kleidungsstück
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.status(
            "Die KI analysiert dein Bild ...",
            expanded=True
        ) as status:

            st.write(
                "📷 Bild wird vorbereitet ..."
            )

            prepared_image = prepare_image(
                image
            )

            st.write(
                "🧠 KI-Modell wird geladen ..."
            )

            model = load_model()

            st.write(
                "🔍 Kleidungsstück wird erkannt ..."
            )

            prediction = model.predict(
                prepared_image,
                verbose=0
            )

            prediction = np.asarray(
                prediction
            ).reshape(-1)

            best_index = int(
                np.argmax(prediction)
            )

            confidence = float(
                prediction[best_index]
            )

            if best_index < len(LABELS):

                detected_label = (
                    LABELS[best_index]
                )

            else:

                detected_label = (
                    f"Klasse {best_index}"
                )

            status.update(
                label="Erkennung abgeschlossen",
                state="complete",
                expanded=False
            )

        # ----------------------------------------------------
        # ERGEBNIS
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="classification-box">

                <div class="classification-title">
                    Die KI erkennt:
                </div>

                <div class="classification-result">
                    {detected_label}
                </div>

                <div style="
                    margin-top:10px;
                    font-size:18px;
                ">
                    Sicherheit:
                    {confidence * 100:.1f} %
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # FUNDORT
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-title">
                Wo wurde es gefunden?
            </div>
            """,
            unsafe_allow_html=True
        )

        location = st.text_input(
            "Fundort",
            placeholder="z. B. Sporthalle, Schulhof, Raum 204 ..."
        )

        found_date = st.date_input(
            "Wann wurde es gefunden?",
            value=date.today()
        )

        if st.button(
            "💾 Fundstück speichern",
            type="primary",
            use_container_width=True
        ):

            if not location.strip():

                st.warning(
                    "Bitte gib noch den Fundort ein."
                )

            else:

                save_item(
                    image=image,
                    item_type=detected_label,
                    location=location.strip(),
                    found_date=found_date
                )

                st.success(
                    "Das Fundstück wurde erfolgreich gespeichert! 🎉"
                )

                st.balloons()


# ============================================================
# SEITE 3 – ÄLTESTE FUNDSTÜCKE
# ============================================================

elif st.session_state.page == "Älteste Fundstücke":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="header-area header-pink">

            <div class="page-title">
                Älteste Fundstücke
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    show_menu()

    school_silhouette(
        ROSA
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Hier findest du die neun ältesten Fundstücke.
        </div>
        """,
        unsafe_allow_html=True
    )

    items = get_items(
        limit=9
    )

    if not items:

        st.info(
            "Es wurden noch keine Fundstücke gespeichert."
        )

    else:

        columns = st.columns(3)

        for index, item in enumerate(items):

            (
                item_id,
                filename,
                item_type,
                location,
                found_date,
                created_at
            ) = item

            image_path = (
                UPLOAD_DIR / filename
            )

            with columns[
                index % 3
            ]:

                st.markdown(
                    '<div class="item-card">',
                    unsafe_allow_html=True
                )

                if image_path.exists():

                    st.image(
                        str(image_path),
                        use_container_width=True
                    )

                st.markdown(
                    f"""
                    <div class="item-title">
                        {item_type}
                    </div>

                    <div class="item-info">
                        📍 {location}<br>
                        📅 {found_date}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )
