import streamlit as st
import tensorflow as tf
import tf_keras
import numpy as np

from PIL import Image
from pathlib import Path
import sqlite3
import uuid
from datetime import date, datetime
import html


# ============================================================
# 1. STREAMLIT EINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. DATEIEN UND ORDNER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"

DATABASE_PATH = BASE_DIR / "fundgrube.db"

UPLOAD_DIR = BASE_DIR / "fundgrube_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. FARBEN
# ============================================================

ROSA = "#F1878D"
BLAU = "#A9E0F7"
WEISS = "#FFFFFF"
SCHWARZ = "#000000"


# ============================================================
# 4. CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       GRUNDLAYOUT
    ======================================================== */

    html, body, [data-testid="stAppViewContainer"] {{
        margin: 0;
        padding: 0;
        background: white;
    }}

    [data-testid="stAppViewContainer"] {{
        background: white;
    }}

    .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    header {{
        background: transparent !important;
    }}


    /* ========================================================
       HAMBURGER
    ======================================================== */

    .menu-button {{
        position: absolute;
        top: 14px;
        left: 18px;
        width: 175px;
        height: 100px;
        z-index: 100;
    }}

    .menu-line {{
        width: 175px;
        height: 6px;
        background: black;
        border-radius: 5px;
        margin-bottom: 28px;
    }}


    /* ========================================================
       HEADER
    ======================================================== */

    .main-header {{
        width: 100%;
        position: relative;
        overflow: hidden;
        padding-top: 20px;
    }}

    .pink-header {{
        background: {ROSA};
    }}

    .blue-header {{
        background: {BLAU};
    }}


    /* ========================================================
       TITEL
    ======================================================== */

    .main-title {{
        color: black;
        text-align: center;
        font-family: Arial, Helvetica, sans-serif;
        font-size: clamp(52px, 7vw, 100px);
        font-weight: 800;
        line-height: 1;
        margin: 0;
        padding: 25px 80px 25px 80px;
        letter-spacing: -4px;
    }}


    /* ========================================================
       SILHOUETTE
    ======================================================== */

    .school-scene {{
        position: relative;
        width: 100%;
        height: 315px;
        overflow: hidden;
        background: white;
    }}

    .school-background {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 175px;
    }}

    .school-ground {{
        position: absolute;
        left: 0;
        bottom: 0;
        width: 100%;
        height: 140px;
        background: white;
    }}

    .ground-line {{
        position: absolute;
        left: 0;
        bottom: 137px;
        width: 100%;
        height: 5px;
        background: black;
    }}


    /* ========================================================
       GEBÄUDE ALLGEMEIN
    ======================================================== */

    .building {{
        position: absolute;
        bottom: 138px;
        background: white;
        border: 5px solid black;
        border-bottom: none;
        box-sizing: border-box;
    }}


    /* ========================================================
       LINKES GEBÄUDE
    ======================================================== */

    .building-left {{
        left: 17%;
        width: 13%;
        height: 145px;
    }}

    .roof-left {{
        position: absolute;
        width: 78px;
        height: 78px;
        left: 50%;
        top: -61px;
        transform: translateX(-50%) rotate(45deg);
        background: white;
        border-left: 5px solid black;
        border-top: 5px solid black;
    }}

    .roof-left-cover {{
        position: absolute;
        width: 105px;
        height: 35px;
        left: 50%;
        top: -3px;
        transform: translateX(-50%);
        background: white;
        z-index: 3;
    }}

    .roof-left-line {{
        position: absolute;
        left: 50%;
        top: -50px;
        width: 5px;
        height: 115px;
        background: black;
        transform: translateX(-50%) rotate(0deg);
        z-index: 5;
    }}


    /* ========================================================
       MITTLERES GEBÄUDE
    ======================================================== */

    .building-middle {{
        left: 41%;
        width: 20%;
        height: 190px;
    }}

    .roof-middle {{
        position: absolute;
        width: 105px;
        height: 105px;
        left: 50%;
        top: -80px;
        transform: translateX(-50%) rotate(45deg);
        background: white;
        border-left: 5px solid black;
        border-top: 5px solid black;
    }}

    .roof-middle-cover {{
        position: absolute;
        width: 145px;
        height: 50px;
        left: 50%;
        top: -3px;
        transform: translateX(-50%);
        background: white;
        z-index: 3;
    }}

    .roof-middle-line {{
        position: absolute;
        left: 50%;
        top: -69px;
        width: 5px;
        height: 145px;
        background: black;
        transform: translateX(-50%);
        z-index: 5;
    }}


    /* ========================================================
       RECHTES GEBÄUDE
    ======================================================== */

    .building-right {{
        left: 74%;
        width: 13%;
        height: 145px;
    }}

    .roof-right {{
        position: absolute;
        width: 78px;
        height: 78px;
        left: 50%;
        top: -61px;
        transform: translateX(-50%) rotate(45deg);
        background: white;
        border-left: 5px solid black;
        border-top: 5px solid black;
    }}

    .roof-right-cover {{
        position: absolute;
        width: 105px;
        height: 35px;
        left: 50%;
        top: -3px;
        transform: translateX(-50%);
        background: white;
        z-index: 3;
    }}

    .roof-right-line {{
        position: absolute;
        left: 50%;
        top: -50px;
        width: 5px;
        height: 115px;
        background: black;
        transform: translateX(-50%);
        z-index: 5;
    }}


    /* ========================================================
       FENSTER
    ======================================================== */

    .window-vertical {{
        position: absolute;
        top: 28px;
        bottom: 0;
        width: 3px;
        background: black;
    }}

    .window-horizontal {{
        position: absolute;
        left: 10%;
        right: 10%;
        top: 70px;
        height: 3px;
        background: black;
    }}

    .left-window-1 {{
        left: 35%;
    }}

    .left-window-2 {{
        left: 65%;
    }}

    .middle-window-1 {{
        left: 30%;
    }}

    .middle-window-2 {{
        left: 50%;
    }}

    .middle-window-3 {{
        left: 70%;
    }}

    .right-window-1 {{
        left: 35%;
    }}

    .right-window-2 {{
        left: 65%;
    }}


    /* ========================================================
       LUPE
    ======================================================== */

    .magnifying-glass-container {{
        width: 100%;
        height: 230px;
        display: flex;
        justify-content: center;
        align-items: center;
        background: white;
    }}

    .magnifying-glass {{
        position: relative;
        width: 135px;
        height: 135px;
        border: 8px solid black;
        border-radius: 50%;
        background: white;
        box-sizing: border-box;
    }}

    .magnifying-glass::before {{
        content: "";
        position: absolute;
        width: 105px;
        height: 105px;
        border: 3px solid black;
        border-radius: 50%;
        top: 7px;
        left: 7px;
        box-sizing: border-box;
    }}

    .magnifying-glass::after {{
        content: "";
        position: absolute;
        width: 22px;
        height: 82px;
        background: white;
        border: 8px solid black;
        border-radius: 15px;
        right: -48px;
        bottom: -49px;
        transform: rotate(-42deg);
        transform-origin: center;
        box-sizing: border-box;
    }}


    /* ========================================================
       BESCHREIBUNG
    ======================================================== */

    .description-wrapper {{
        width: 100%;
        text-align: center;
        background: white;
        padding: 0 20px 35px 20px;
        box-sizing: border-box;
    }}

    .description-box {{
        display: inline-block;
        max-width: 90%;
        border: 3px solid black;
        border-radius: 45px;
        padding: 8px 25px;
        font-family: Arial, Helvetica, sans-serif;
        font-size: clamp(17px, 2.1vw, 28px);
        color: black;
        background: white;
    }}


    /* ========================================================
       UPLOAD-BEREICH
    ======================================================== */

    .upload-space {{
        background: white;
        width: 100%;
        padding: 35px 20px 20px 20px;
        box-sizing: border-box;
        text-align: center;
    }}

    .upload-box {{
        width: min(800px, 85%);
        min-height: 280px;
        margin: 0 auto;
        border: 4px dashed black;
        border-radius: 35px;
        display: flex;
        justify-content: center;
        align-items: center;
        background: white;
        box-sizing: border-box;
    }}

    .upload-text {{
        color: black;
        font-size: clamp(20px, 2.5vw, 30px);
        padding: 30px;
    }}


    /* ========================================================
       FILE UPLOADER
    ======================================================== */

    section[data-testid="stFileUploader"] {{
        width: min(800px, 85%);
        margin: 0 auto;
    }}

    section[data-testid="stFileUploader"] > div {{
        border: 4px dashed black !important;
        border-radius: 35px !important;
        background: white !important;
        padding: 30px !important;
    }}


    /* ========================================================
       TEXT INPUT
    ======================================================== */

    div[data-baseweb="input"] {{
        border: 3px solid black !important;
        border-radius: 40px !important;
        background: white !important;
    }}

    div[data-baseweb="input"] input {{
        color: black !important;
        font-size: 18px !important;
        text-align: center !important;
    }}


    /* ========================================================
       BUTTONS
    ======================================================== */

    .stButton > button {{
        border: 3px solid black !important;
        border-radius: 35px !important;
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


    /* ========================================================
       STREAMLIT POPOVER
    ======================================================== */

    [data-testid="stPopover"] > button {{
        position: absolute;
        top: 15px;
        left: 15px;
        z-index: 200;
        width: 175px;
        height: 105px;
        opacity: 0;
        cursor: pointer;
    }}


    /* ========================================================
       MENÜ-LINIEN
    ======================================================== */

    .hamburger-visual {{
        position: absolute;
        top: 15px;
        left: 18px;
        z-index: 150;
        pointer-events: none;
    }}

    .hamburger-line {{
        width: 175px;
        height: 6px;
        background: black;
        border-radius: 5px;
        margin-bottom: 28px;
    }}


    /* ========================================================
       SEITENABSCHNITTE
    ======================================================== */

    .section-title {{
        text-align: center;
        font-family: Arial, Helvetica, sans-serif;
        color: black;
        font-size: clamp(32px, 5vw, 55px);
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 25px;
    }}

    .section-subtitle {{
        text-align: center;
        color: black;
        font-size: 20px;
        margin: 25px;
    }}


    /* ========================================================
       KI-ERGEBNIS
    ======================================================== */

    .classification-box {{
        width: min(750px, 90%);
        margin: 30px auto;
        padding: 25px;
        border: 3px solid black;
        border-radius: 25px;
        background: white;
        text-align: center;
    }}

    .classification-title {{
        color: black;
        font-size: 23px;
        font-weight: 700;
    }}

    .classification-result {{
        color: black;
        font-size: 36px;
        font-weight: 800;
        margin-top: 8px;
    }}


    /* ========================================================
       FUNDSTÜCK-KARTEN
    ======================================================== */

    .item-card {{
        border: 3px solid black;
        border-radius: 22px;
        padding: 12px;
        background: white;
        margin-bottom: 25px;
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
        line-height: 1.6;
        margin-top: 5px;
    }}


    /* ========================================================
       MOBILE
    ======================================================== */

    @media (max-width: 700px) {{

        .main-title {{
            font-size: 48px;
            padding-top: 65px;
        }}

        .school-scene {{
            height: 230px;
        }}

        .building {{
            bottom: 90px;
            border-width: 3px;
        }}

        .ground-line {{
            bottom: 87px;
        }}

        .school-ground {{
            height: 90px;
        }}

        .building-middle {{
            height: 130px;
        }}

        .building-left,
        .building-right {{
            height: 100px;
        }}

        .roof-middle {{
            width: 75px;
            height: 75px;
            top: -58px;
        }}

        .roof-left,
        .roof-right {{
            width: 55px;
            height: 55px;
            top: -43px;
        }}

        .magnifying-glass-container {{
            height: 190px;
        }}

    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. HAMBURGER-VISUAL
# ============================================================

def hamburger_visual():

    st.markdown(
        """
        <div class="hamburger-visual">

            <div class="hamburger-line"></div>
            <div class="hamburger-line"></div>
            <div class="hamburger-line"></div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 6. SCHUL-SILHOUETTE
# ============================================================

def school_silhouette(background_color):

    st.markdown(
        f"""
        <div class="school-scene">

            <div
                class="school-background"
                style="background:{background_color};"
            ></div>

            <div class="school-ground"></div>


            <!-- ================================================
                 LINKES GEBÄUDE
            ================================================= -->

            <div class="building building-left">

                <div class="roof-left"></div>
                <div class="roof-left-cover"></div>
                <div class="roof-left-line"></div>

                <div
                    class="window-vertical left-window-1"
                ></div>

                <div
                    class="window-vertical left-window-2"
                ></div>

                <div class="window-horizontal"></div>

            </div>


            <!-- ================================================
                 MITTLERES GEBÄUDE
            ================================================= -->

            <div class="building building-middle">

                <div class="roof-middle"></div>
                <div class="roof-middle-cover"></div>
                <div class="roof-middle-line"></div>

                <div
                    class="window-vertical middle-window-1"
                ></div>

                <div
                    class="window-vertical middle-window-2"
                ></div>

                <div
                    class="window-vertical middle-window-3"
                ></div>

                <div class="window-horizontal"></div>

            </div>


            <!-- ================================================
                 RECHTES GEBÄUDE
            ================================================= -->

            <div class="building building-right">

                <div class="roof-right"></div>
                <div class="roof-right-cover"></div>
                <div class="roof-right-line"></div>

                <div
                    class="window-vertical right-window-1"
                ></div>

                <div
                    class="window-vertical right-window-2"
                ></div>

                <div class="window-horizontal"></div>

            </div>


            <!-- Bodenlinie -->

            <div class="ground-line"></div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 7. LABELS LADEN
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
# 8. KI-MODELL LADEN
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
# 9. DATENBANK INITIALISIEREN
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
# 10. BILD VORBEREITEN
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.asarray(
        image
    ).astype(np.float32)

    image_array = (
        image_array / 127.5
    ) - 1.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# 11. BILD KLASSIFIZIEREN
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

        label = LABELS[
            best_index
        ]

    else:

        label = (
            f"Klasse {best_index}"
        )

    return label, confidence


# ============================================================
# 12. FUNDSTÜCK SPEICHERN
# ============================================================

def save_item(
    image,
    item_type,
    location,
    found_date
):

    filename = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_"
        + uuid.uuid4().hex[:8]
        + ".jpg"
    )

    image_path = (
        UPLOAD_DIR / filename
    )

    image.convert("RGB").save(
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
            filename,
            item_type,
            location,
            str(found_date),
            datetime.now().isoformat()
        )
    )

    connection.commit()

    connection.close()


# ============================================================
# 13. FUNDSTÜCKE LADEN
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

            WHERE
                item_type LIKE ?
                OR location LIKE ?

            ORDER BY found_date DESC
            """,
            (
                f"%{search_term}%",
                f"%{search_term}%"
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

        if limit is not None:

            query += (
                f" LIMIT {int(limit)}"
            )

        cursor.execute(query)

    items = cursor.fetchall()

    connection.close()

    return items


# ============================================================
# 14. NAVIGATION
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "Suche"


def show_menu():

    hamburger_visual()

    with st.popover(
        "Menü"
    ):

        st.markdown(
            "### Fundgrube"
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
# 15. SUCHSEITE
# ============================================================

if st.session_state.page == "Suche":

    # --------------------------------------------------------
    # ROSA HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="main-header pink-header">

            <div class="main-title">
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
        <div class="magnifying-glass-container">

            <div class="magnifying-glass"></div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # BESCHREIBUNG
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="description-wrapper">

            <div class="description-box">
                Beschreibe dein verlorenes Kleidungsstück
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SUCHFELD
    # --------------------------------------------------------

    search = st.text_input(
        "Suche",
        placeholder="z. B. schwarzer Hoodie ...",
        label_visibility="collapsed"
    )


    # --------------------------------------------------------
    # SUCHERGEBNISSE
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

                    safe_type = html.escape(
                        str(item_type)
                    )

                    safe_location = html.escape(
                        str(location)
                    )

                    safe_date = html.escape(
                        str(found_date)
                    )

                    st.markdown(
                        f"""
                        <div class="item-title">
                            {safe_type}
                        </div>

                        <div class="item-info">

                            📍 {safe_location}

                            <br>

                            📅 {safe_date}

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
# 16. UPLOADSEITE
# ============================================================

elif st.session_state.page == "Hochladen":

    # --------------------------------------------------------
    # BLAUER HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="main-header blue-header">

            <div class="main-title">
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
    # UPLOAD-TEXT
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="description-wrapper">

            <div class="description-box">
                Lade ein Foto aus deiner Mediathek hoch
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # BILD HOCHLADEN
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # BILD ANZEIGEN
        # ----------------------------------------------------

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
        # KI-ERGEBNIS
        # ----------------------------------------------------

        safe_label = html.escape(
            str(detected_label)
        )

        st.markdown(
            f"""
            <div class="classification-box">

                <div class="classification-title">
                    Die KI erkennt:
                </div>

                <div class="classification-result">
                    {safe_label}
                </div>

                <div style="
                    margin-top:10px;
                    font-size:18px;
                    color:black;
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


        # ----------------------------------------------------
        # DATUM
        # ----------------------------------------------------

        found_date = st.date_input(
            "Wann wurde es gefunden?",
            value=date.today()
        )


        # ----------------------------------------------------
        # SPEICHERN
        # ----------------------------------------------------

        if st.button(
            "💾 Fundstück speichern",
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
# 17. ÄLTESTE FUNDSTÜCKE
# ============================================================

elif st.session_state.page == "Älteste Fundstücke":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="main-header pink-header">

            <div class="main-title">
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


    # --------------------------------------------------------
    # BESCHREIBUNG
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-subtitle">
            Hier findest du die neun ältesten Fundstücke.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # FUNDSTÜCKE
    # --------------------------------------------------------

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

                safe_type = html.escape(
                    str(item_type)
                )

                safe_location = html.escape(
                    str(location)
                )

                safe_date = html.escape(
                    str(found_date)
                )

                st.markdown(
                    f"""
                    <div class="item-title">
                        {safe_type}
                    </div>

                    <div class="item-info">

                        📍 {safe_location}

                        <br>

                        📅 {safe_date}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )
