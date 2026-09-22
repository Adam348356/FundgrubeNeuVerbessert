import streamlit as st
import tensorflow as tf
import tf_keras
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path
from datetime import date
import sqlite3
import uuid


# ============================================================
# SEITENEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FARBEN
# ============================================================

# Dunkles, leicht metallisches Blau
BACKGROUND = "#173746"
BACKGROUND_2 = "#244B5C"

# Metallisches Grau
METAL = "#747C82"
METAL_DARK = "#596167"
METAL_LIGHT = "#8B9398"

WHITE = "#FFFFFF"
BLACK = "#1C2226"


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    f"""
    <style>

    /* =====================================================
       GESAMTE APP
       ===================================================== */

    .stApp {{
        background:
            linear-gradient(
                135deg,
                {BACKGROUND} 0%,
                #1D4353 35%,
                {BACKGROUND_2} 65%,
                #163441 100%
            );

        background-attachment: fixed;
    }}


    /* =====================================================
       HAUPTBEREICH
       ===================================================== */

    .main .block-container {{
        max-width: 1150px;
        padding-top: 35px;
        padding-bottom: 80px;
    }}


    /* =====================================================
       SEITENLEISTE
       ===================================================== */

    section[data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                #515A60 0%,
                #3E474D 50%,
                #333C42 100%
            );
    }}

    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}


    /* =====================================================
       TITEL-PANEL
       ===================================================== */

    .page-title-box {{
        width: 100%;
        box-sizing: border-box;

        background:
            linear-gradient(
                145deg,
                #8E969B 0%,
                #747C82 35%,
                #636B71 65%,
                #858D92 100%
            );

        border: 2px solid #4F575C;

        border-radius: 18px;

        padding: 28px 35px 30px 35px;

        margin-bottom: 28px;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.30),
            inset 0 -2px 5px rgba(0,0,0,0.20),
            0 5px 14px rgba(0,0,0,0.30);

        text-align: center;
    }}


    .page-title-box h1 {{
        color: white !important;

        font-size: 52px !important;

        font-weight: 800 !important;

        margin: 0 !important;

        padding: 0 !important;

        line-height: 1.1 !important;

        text-shadow:
            0 2px 3px rgba(0,0,0,0.45);
    }}


    .page-title-box p {{
        color: #F4F4F4 !important;

        font-size: 19px !important;

        margin-top: 10px !important;

        margin-bottom: 0 !important;

        text-shadow:
            0 1px 2px rgba(0,0,0,0.35);
    }}


    /* =====================================================
       SUCHFELD
       ===================================================== */

    div[data-baseweb="input"] {{
        background:
            linear-gradient(
                145deg,
                #858D92,
                #687177
            ) !important;

        border: 2px solid #4F575C !important;

        border-radius: 12px !important;

        box-shadow:
            inset 0 1px 2px rgba(255,255,255,0.20),
            inset 0 -2px 4px rgba(0,0,0,0.20) !important;
    }}

    div[data-baseweb="input"] input {{
        color: white !important;

        background: transparent !important;

        font-size: 16px !important;
    }}

    div[data-baseweb="input"] input::placeholder {{
        color: #E7E7E7 !important;

        opacity: 1 !important;
    }}


    /* =====================================================
       DATE / SELECT
       ===================================================== */

    div[data-baseweb="select"] > div {{
        background:
            linear-gradient(
                145deg,
                #858D92,
                #687177
            ) !important;

        color: white !important;

        border: 2px solid #4F575C !important;
    }}


    /* =====================================================
       UPLOADER
       ===================================================== */

    section[data-testid="stFileUploaderDropzone"] {{
        background:
            linear-gradient(
                145deg,
                #858D92,
                #687177
            ) !important;

        border: 2px solid #4F575C !important;

        border-radius: 14px !important;

        box-shadow:
            inset 0 1px 2px rgba(255,255,255,0.20),
            inset 0 -2px 4px rgba(0,0,0,0.20);
    }}

    section[data-testid="stFileUploaderDropzone"] * {{
        color: white !important;
    }}


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {{
        width: 100%;

        background:
            linear-gradient(
                145deg,
                #8C9499,
                #687177
            ) !important;

        color: white !important;

        border: 2px solid #4E565B !important;

        border-radius: 12px !important;

        font-weight: 700 !important;

        min-height: 48px !important;

        box-shadow:
            inset 0 1px 2px rgba(255,255,255,0.25),
            0 3px 7px rgba(0,0,0,0.25);

        transition:
            transform 0.15s ease,
            filter 0.15s ease;
    }}

    .stButton > button:hover {{
        transform: scale(1.025);

        filter: brightness(1.10);

        color: white !important;
    }}


    /* =====================================================
       TEXT
       ===================================================== */

    label {{
        color: white !important;

        font-weight: 600 !important;
    }}

    .stMarkdown {{
        color: white;
    }}

    .stCaption {{
        color: #DCE3E6 !important;
    }}


    /* =====================================================
       INFO-BOX
       ===================================================== */

    div[data-testid="stAlert"] {{
        background:
            linear-gradient(
                145deg,
                #5D9FBC,
                #4788A5
            ) !important;

        color: white !important;

        border: 1px solid #376F88 !important;

        border-radius: 12px !important;

        box-shadow:
            0 3px 8px rgba(0,0,0,0.20);
    }}


    /* =====================================================
       KARTEN
       ===================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background:
            linear-gradient(
                145deg,
                #81898E,
                #656D72
            ) !important;

        border: 2px solid #4E565B !important;

        border-radius: 16px !important;

        box-shadow:
            inset 0 1px 2px rgba(255,255,255,0.20),
            0 4px 10px rgba(0,0,0,0.25);
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATEIEN
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"

UPLOAD_DIR = BASE_DIR / "fundgrube_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DB_PATH = BASE_DIR / "fundgrube.db"


# ============================================================
# DATENBANK
# ============================================================

def init_database():

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS found_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
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
# LABELS
# ============================================================

def load_labels():

    if not LABELS_PATH.exists():
        return [
            "Hose",
            "Schuh",
            "T-Shirt",
            "Hoodie"
        ]

    labels = []

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
                    parts[1].strip()
                )
            else:
                labels.append(line)

    return labels


LABELS = load_labels()


# ============================================================
# KI-MODELL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    try:

        return tf_keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

    except Exception as error:

        st.error(
            "Das KI-Modell konnte nicht geladen werden."
        )

        st.exception(error)

        return None


model = load_model()


# ============================================================
# BILDERKENNUNG
# ============================================================

def classify_image(image):

    if model is None:
        return None, 0.0

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    array = np.asarray(
        image
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
    )

    probabilities = prediction[0]

    index = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        probabilities[index]
    )

    if index < len(LABELS):
        label = LABELS[index]
    else:
        label = f"Klasse {index}"

    return label, confidence


# ============================================================
# TITELBLOCK
# ============================================================

def page_header(
    title,
    subtitle=""
):

    if subtitle:

        st.markdown(
            f"""
            <div class="page-title-box">
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="page-title-box">
                <h1>{title}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Suche"


with st.sidebar:

    st.markdown(
        "## Fundgrube"
    )

    st.divider()

    if st.button(
        "🔎 Suche",
        width="stretch"
    ):

        st.session_state.page = "Suche"

        st.rerun()

    if st.button(
        "📷 Bild hochladen",
        width="stretch"
    ):

        st.session_state.page = "Bild hochladen"

        st.rerun()

    if st.button(
        "🕘 Älteste Fundstücke",
        width="stretch"
    ):

        st.session_state.page = "Älteste Fundstücke"

        st.rerun()

    st.divider()

    st.caption(
        "Fundgrube"
    )


# ============================================================
# SEITE: SUCHE
# ============================================================

if st.session_state.page == "Suche":

    page_header(
        "Fundgrube",
        "Schul-Fundstücke einfach wiederfinden"
    )

    # --------------------------------------------------------
    # LUPEN-EMOJI / GRAFIK
    # --------------------------------------------------------

    left, center, right = st.columns(
        [1, 2, 1]
    )

    with center:

        st.markdown(
            """
            <div
                style="
                    text-align:center;
                    font-size:125px;
                    line-height:1;
                    margin:10px 0 20px 0;
                "
            >
                🔎
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # SUCHFELD
    # --------------------------------------------------------

    search_text = st.text_input(
        "Was hast du verloren?",
        placeholder=(
            "z. B. Hoodie, schwarze Hose, Schuh ..."
        ),
        key="search_input"
    )

    if not search_text:

        st.info(
            "Gib oben ein Kleidungsstück ein, "
            "um passende Fundstücke zu finden."
        )

    else:

        connection = sqlite3.connect(
            DB_PATH
        )

        rows = connection.execute(
            """
            SELECT
                id,
                image_path,
                item_type,
                location,
                found_date
            FROM found_items
            ORDER BY found_date ASC
            """
        ).fetchall()

        connection.close()

        search_lower = search_text.lower()

        matches = []

        for row in rows:

            item_type = str(
                row[2]
            ).lower()

            location = str(
                row[3]
            ).lower()

            if (
                search_lower in item_type
                or search_lower in location
            ):

                matches.append(row)

        if not matches:

            st.warning(
                "Leider wurde kein passendes "
                "Fundstück gefunden."
            )

        else:

            st.subheader(
                f"{len(matches)} passende Fundstücke"
            )

            columns = st.columns(3)

            for index, row in enumerate(matches):

                with columns[
                    index % 3
                ]:

                    with st.container(
                        border=True
                    ):

                        image_path = Path(
                            row[1]
                        )

                        if image_path.exists():

                            image = Image.open(
                                image_path
                            ).convert("RGB")

                            image = ImageOps.fit(
                                image,
                                (500, 500)
                            )

                            st.image(
                                image,
                                width="stretch"
                            )

                        st.markdown(
                            f"### {row[2]}"
                        )

                        st.write(
                            f"📍 {row[3]}"
                        )

                        st.write(
                            f"📅 {row[4]}"
                        )


# ============================================================
# SEITE: BILD HOCHLADEN
# ============================================================

elif st.session_state.page == "Bild hochladen":

    page_header(
        "Bild hochladen",
        "Lade ein Foto hoch und lass es von der KI erkennen"
    )

    st.subheader(
        "Foto aus deiner Mediathek auswählen"
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

    if uploaded_file is None:

        st.info(
            "Wähle ein Bild aus, um die "
            "automatische Erkennung zu starten."
        )

    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        left, center, right = st.columns(
            [1, 2, 1]
        )

        with center:

            st.image(
                image,
                width="stretch"
            )

        st.subheader(
            "🤖 KI-Erkennung"
        )

        with st.spinner(
            "Die KI untersucht das Bild ..."
        ):

            predicted_label, confidence = (
                classify_image(image)
            )

        if predicted_label:

            st.success(
                f"Erkannt: {predicted_label}"
            )

            st.progress(
                confidence
            )

            st.write(
                f"Sicherheit: "
                f"{confidence * 100:.1f} %"
            )

        else:

            predicted_label = "Unbekannt"

            st.warning(
                "Das KI-Modell konnte nicht "
                "geladen werden."
            )

        st.divider()

        st.subheader(
            "📍 Wo wurde der Gegenstand gefunden?"
        )

        location = st.text_input(
            "Fundort",
            placeholder=(
                "z. B. Sporthalle, Mensa, Raum 203 ..."
            )
        )

        st.subheader(
            "📅 Wann wurde der Gegenstand gefunden?"
        )

        found_date = st.date_input(
            "Funddatum",
            value=date.today()
        )

        if st.button(
            "Fundstück speichern",
            width="stretch"
        ):

            if not location.strip():

                st.warning(
                    "Bitte gib zuerst den Fundort ein."
                )

            else:

                filename = (
                    str(uuid.uuid4())
                    + ".jpg"
                )

                image_path = (
                    UPLOAD_DIR / filename
                )

                image.save(
                    image_path,
                    format="JPEG",
                    quality=92
                )

                connection = sqlite3.connect(
                    DB_PATH
                )

                connection.execute(
                    """
                    INSERT INTO found_items
                    (
                        image_path,
                        item_type,
                        location,
                        found_date,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        str(image_path),
                        predicted_label,
                        location.strip(),
                        found_date.isoformat()
                    )
                )

                connection.commit()
                connection.close()

                st.success(
                    "Das Fundstück wurde "
                    "erfolgreich gespeichert."
                )


# ============================================================
# SEITE: ÄLTESTE FUNDSTÜCKE
# ============================================================

elif st.session_state.page == "Älteste Fundstücke":

    page_header(
        "Älteste Fundstücke",
        "Die neun ältesten gespeicherten Fundstücke"
    )

    connection = sqlite3.connect(
        DB_PATH
    )

    rows = connection.execute(
        """
        SELECT
            id,
            image_path,
            item_type,
            location,
            found_date
        FROM found_items
        ORDER BY found_date ASC, id ASC
        LIMIT 9
        """
    ).fetchall()

    connection.close()

    if not rows:

        st.info(
            "Es wurden bisher noch keine "
            "Fundstücke gespeichert."
        )

    else:

        columns = st.columns(3)

        for index, row in enumerate(rows):

            with columns[
                index % 3
            ]:

                with st.container(
                    border=True
                ):

                    image_path = Path(
                        row[1]
                    )

                    if image_path.exists():

                        image = Image.open(
                            image_path
                        ).convert("RGB")

                        image = ImageOps.fit(
                            image,
                            (500, 500)
                        )

                        st.image(
                            image,
                            width="stretch"
                        )

                    st.markdown(
                        f"### {row[2]}"
                    )

                    st.write(
                        f"📍 **Fundort:** {row[3]}"
                    )

                    st.write(
                        f"📅 **Gefunden:** {row[4]}"
                    )


# ============================================================
# FUSSZEILE
# ============================================================

st.write("")

st.caption(
    "Fundgrube – Schul-Fundstücke wiederfinden"
)
