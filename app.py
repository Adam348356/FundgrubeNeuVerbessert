import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import tf_keras
import numpy as np

from PIL import Image
from pathlib import Path
import sqlite3
import uuid
from datetime import date, datetime


# ============================================================
# 1. GRUNDEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. DATEIPFADE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"

UPLOAD_DIR = BASE_DIR / "fundgrube_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DATABASE_PATH = BASE_DIR / "fundgrube.db"


# ============================================================
# 3. FARBEN
# ============================================================

ROSA = "#F1878D"
BLAU = "#9BDCF5"
WEISS = "#FFFFFF"
SCHWARZ = "#000000"


# ============================================================
# 4. SEITEN-DESIGN
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       GRUNDLAYOUT
    ======================================================== */

    .stApp {{
        background-color: white;
    }}

    .block-container {{
        padding-top: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        padding-bottom: 40px !important;
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
       TITEL
    ======================================================== */

    .page-title {{
        color: black;
        text-align: center;
        font-size: clamp(48px, 7vw, 90px);
        font-weight: 800;
        line-height: 1;
        padding-top: 32px;
        padding-bottom: 20px;
        letter-spacing: -3px;
    }}

    .upload-title {{
        color: black;
        text-align: center;
        font-size: clamp(42px, 6vw, 82px);
        font-weight: 800;
        line-height: 1;
        padding-top: 32px;
        padding-bottom: 20px;
        letter-spacing: -3px;
    }}


    /* ========================================================
       HEADER
    ======================================================== */

    .header {{
        width: 100%;
        position: relative;
        overflow: hidden;
    }}

    .pink-header {{
        background: {ROSA};
    }}

    .blue-header {{
        background: {BLAU};
    }}


    /* ========================================================
       SUCHBEREICH
    ======================================================== */

    .search-area {{
        width: 100%;
        background: white;
        text-align: center;
        padding-top: 30px;
    }}

    .search-description {{
        display: inline-block;
        border: 3px solid black;
        border-radius: 40px;
        padding: 8px 28px;
        color: black;
        background: white;
        font-size: clamp(18px, 2vw, 28px);
        margin-bottom: 18px;
    }}


    /* ========================================================
       LUPE
    ======================================================== */

    .magnifying-glass {{
        width: 190px;
        height: 190px;
        margin: 0 auto;
    }}


    /* ========================================================
       UPLOADBEREICH
    ======================================================== */

    .upload-area {{
        width: 100%;
        background: white;
        text-align: center;
        padding-top: 30px;
    }}

    .upload-description {{
        display: inline-block;
        border: 3px solid black;
        border-radius: 28px;
        padding: 12px 28px;
        color: black;
        background: white;
        font-size: clamp(18px, 2vw, 28px);
    }}


    /* ========================================================
       STREAMLIT FILE UPLOADER
    ======================================================== */

    section[data-testid="stFileUploader"] {{
        width: min(850px, 88%);
        margin-left: auto;
        margin-right: auto;
    }}

    section[data-testid="stFileUploader"] > div {{
        border: 4px dashed black !important;
        border-radius: 32px !important;
        background: white !important;
        padding: 25px !important;
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
        font-size: 19px !important;
        text-align: center !important;
    }}


    /* ========================================================
       BUTTONS
    ======================================================== */

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
        border-color: black !important;
        color: black !important;
    }}


    /* ========================================================
       POPover / MENÜ
    ======================================================== */

    [data-testid="stPopover"] {{
        position: absolute;
        top: 18px;
        right: 20px;
        z-index: 1000;
    }}

    [data-testid="stPopover"] > button {{
        border: none !important;
        background: transparent !important;
        color: black !important;
        font-size: 38px !important;
        padding: 0 !important;
        min-height: 45px !important;
    }}


    /* ========================================================
       SEITENTITEL
    ======================================================== */

    .section-title {{
        text-align: center;
        color: black;
        font-size: clamp(32px, 5vw, 55px);
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 15px;
    }}

    .section-subtitle {{
        text-align: center;
        color: black;
        font-size: 20px;
        margin: 20px;
    }}


    /* ========================================================
       KLASSIFIZIERUNG
    ======================================================== */

    .classification-box {{
        width: min(800px, 85%);
        margin: 30px auto;
        padding: 25px;
        background: white;
        border: 3px solid black;
        border-radius: 25px;
        text-align: center;
    }}

    .classification-title {{
        font-size: 24px;
        font-weight: 700;
        color: black;
    }}

    .classification-result {{
        font-size: 36px;
        font-weight: 800;
        color: black;
        margin-top: 10px;
    }}


    /* ========================================================
       FUNDSTÜCK-KARTEN
    ======================================================== */

    .item-card {{
        background: white;
        border: 3px solid black;
        border-radius: 22px;
        padding: 12px;
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
        line-height: 1.5;
        margin-top: 5px;
    }}


    /* ========================================================
       MOBILE
    ======================================================== */

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
            width: 145px;
            height: 145px;
        }}

    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. SCHUL-SILHOUETTE
# ============================================================

def school_silhouette(background_color):

    svg_code = f"""
    <!DOCTYPE html>

    <html>

    <head>

        <style>

            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                background: {background_color};
            }}

            svg {{
                display: block;
                width: 100%;
                height: 100%;
            }}

        </style>

    </head>

    <body>

        <svg
            viewBox="0 0 1500 330"
            preserveAspectRatio="none"
            xmlns="http://www.w3.org/2000/svg"
        >

            <!-- =================================================
                 HINTERGRUND
            ================================================== -->

            <rect
                x="0"
                y="0"
                width="1500"
                height="330"
                fill="{background_color}"
            />


            <!-- =================================================
                 WEISSE SILHOUETTE
            ================================================== -->

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
                    L 1500 330
                    L 0 330

                    Z
                "
                fill="white"
            />


            <!-- =================================================
                 LINKES GEBÄUDE
            ================================================== -->

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

            <!-- Dach links -->

            <path
                d="
                    M 275 220
                    L 320 155
                    L 410 220
                "
                fill="none"
                stroke="black"
                stroke-width="4"
            />

            <!-- Fenster links -->

            <line
                x1="320"
                y1="155"
                x2="320"
                y2="275"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="355"
                y1="185"
                x2="355"
                y2="275"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="390"
                y1="185"
                x2="390"
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


            <!-- =================================================
                 MITTLERES GEBÄUDE
            ================================================== -->

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

            <!-- Großes mittleres Dach -->

            <path
                d="
                    M 650 220
                    L 760 90
                    L 870 220
                "
                fill="none"
                stroke="black"
                stroke-width="4"
            />

            <!-- Mittlere Fenster -->

            <line
                x1="760"
                y1="90"
                x2="760"
                y2="285"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="705"
                y1="155"
                x2="705"
                y2="285"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="815"
                y1="155"
                x2="815"
                y2="285"
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

            <line
                x1="720"
                y1="110"
                x2="720"
                y2="155"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="800"
                y1="110"
                x2="800"
                y2="155"
                stroke="black"
                stroke-width="3"
            />


            <!-- =================================================
                 RECHTES GEBÄUDE
            ================================================== -->

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

            <!-- Dach rechts -->

            <path
                d="
                    M 1100 220
                    L 1185 155
                    L 1275 220
                "
                fill="none"
                stroke="black"
                stroke-width="4"
            />

            <!-- Fenster rechts -->

            <line
                x1="1185"
                y1="155"
                x2="1185"
                y2="285"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="1145"
                y1="190"
                x2="1145"
                y2="285"
                stroke="black"
                stroke-width="3"
            />

            <line
                x1="1230"
                y1="190"
                x2="1230"
                y2="285"
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


            <!-- =================================================
                 BODENLINIE
            ================================================== -->

            <line
                x1="0"
                y1="285"
                x2="1500"
                y2="285"
                stroke="black"
                stroke-width="5"
            />

        </svg>

    </body>

    </html>
    """

    components.html(
        svg_code,
        height=330,
        scrolling=False
    )


# ============================================================
# 6. LABELS LADEN
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

                labels.append(
                    parts[1]
                )

            else:

                labels.append(
                    line
                )

    return labels


LABELS = load_labels()


# ============================================================
# 7. KI-MODELL LADEN
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Die Datei keras_model.h5 wurde nicht gefunden."
        )

    model = tf_keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# 8. DATENBANK
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
# 9. BILD FÜR KI VORBEREITEN
# ============================================================

def prepare_image(image):

    image = image.convert(
        "RGB"
    )

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
# 10. KLASSIFIZIERUNG
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
# 11. FUNDSTÜCK SPEICHERN
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

    image = image.convert(
        "RGB"
    )

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
# 12. FUNDSTÜCKE LADEN
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

        cursor.execute(
            query
        )

    items = cursor.fetchall()

    connection.close()

    return items


# ============================================================
# 13. NAVIGATION
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "Suche"


def show_menu():

    left, right = st.columns(
        [12, 1]
    )

    with right:

        with st.popover("☰"):

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
# 14. SEITE – SUCHE
# ============================================================

if st.session_state.page == "Suche":

    # --------------------------------------------------------
    # ROSA HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="header pink-header">

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
                    width="190"
                    height="190"
                    viewBox="0 0 190 190"
                    xmlns="http://www.w3.org/2000/svg"
                >

                    <circle
                        cx="78"
                        cy="78"
                        r="50"
                        fill="white"
                        stroke="black"
                        stroke-width="7"
                    />

                    <circle
                        cx="78"
                        cy="78"
                        r="39"
                        fill="white"
                        stroke="black"
                        stroke-width="3"
                    />

                    <line
                        x1="113"
                        y1="113"
                        x2="153"
                        y2="153"
                        stroke="black"
                        stroke-width="12"
                        stroke-linecap="round"
                    />

                </svg>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SUCHBESCHREIBUNG
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            text-align:center;
            margin-bottom:10px;
        ">

            <div class="search-description">
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

                    st.markdown(
                        f"""
                        <div class="item-title">
                            {item_type}
                        </div>

                        <div class="item-info">

                            📍 {location}

                            <br>

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
# 15. SEITE – BILD HOCHLADEN
# ============================================================

elif st.session_state.page == "Hochladen":

    # --------------------------------------------------------
    # BLAUER HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="header blue-header">

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
    # UPLOAD-TEXT
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="upload-area">

            <div class="upload-description">
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


    # --------------------------------------------------------
    # BILD-UPLOAD
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
    # BILD WURDE HOCHGELADEN
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
        # KI-KLASSIFIZIERUNG
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
        # ERKENNUNGSERGEBNIS
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
# 16. SEITE – ÄLTESTE FUNDSTÜCKE
# ============================================================

elif st.session_state.page == "Älteste Fundstücke":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="header pink-header">

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

                st.markdown(
                    f"""
                    <div class="item-title">
                        {item_type}
                    </div>

                    <div class="item-info">

                        📍 {location}

                        <br>

                        📅 {found_date}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )
