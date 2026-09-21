import os
import re
import sqlite3
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import tf_keras
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# GRUNDEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

DB_PATH = "fundgrube.db"
UPLOAD_DIR = Path("fundgrube_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Suche"


# ============================================================
# LABELS LADEN
# ============================================================

def load_labels():
    labels = []

    if not os.path.exists(LABELS_PATH):
        return ["Hose", "Schuh", "T-Shirt", "Hoodie"]

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # z.B.:
            # 0 Hose
            # 1 Schuh
            # 2 T-Shirt
            # 3 Hoodie
            match = re.match(r"^\s*\d+\s+(.*)$", line)

            if match:
                labels.append(match.group(1).strip())
            else:
                labels.append(line)

    return labels


LABELS = load_labels()


# ============================================================
# DATENBANK
# ============================================================

def init_database():
    connection = sqlite3.connect(DB_PATH)
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
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Das Teachable-Machine-Modell laden.

    tf_keras wird hier bewusst verwendet, weil das H5-Modell
    aus Teachable Machine stammt und damit kompatibel geladen wird.
    """
    return tf_keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


def classify_image(image):
    """
    Bild für das Teachable-Machine-Modell vorbereiten
    und Klasse + Wahrscheinlichkeit zurückgeben.
    """

    model = load_model()

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image).astype(np.float32)

    # Standard-Preprocessing von Teachable Machine
    image_array = (image_array / 127.5) - 1.0

    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)

    probabilities = prediction[0]

    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])

    if index < len(LABELS):
        label = LABELS[index]
    else:
        label = f"Klasse {index}"

    return label, confidence


# ============================================================
# SCHRIFT
# ============================================================

def get_font(size, bold=False):
    """
    Versucht eine normale Linux-Schrift zu laden.
    Falls sie nicht vorhanden ist, wird die PIL-Standardschrift
    verwendet.
    """

    possible_fonts = []

    if bold:
        possible_fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    else:
        possible_fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]

    for font_path in possible_fonts:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass

    return ImageFont.load_default()


# ============================================================
# SCHUL-GRAFIK
# ============================================================

def draw_school_silhouette(draw, width, base_y, scale=1.0):
    """
    Zeichnet die Schule aus der Vorlage nach:
    kleines Gebäude – großes Mittelgebäude – kleines Gebäude.
    """

    black = (15, 15, 15)
    white = (255, 255, 255)

    # --------------------------------------------------------
    # Linkes Gebäude
    # --------------------------------------------------------

    left_x = int(width * 0.17)
    left_w = int(250 * scale)

    left_top = base_y - int(160 * scale)
    left_roof_top = left_top - int(55 * scale)

    left_points = [
        (left_x, base_y),
        (left_x, left_top),
        (left_x + int(65 * scale), left_top),
        (left_x + int(125 * scale), left_roof_top),
        (left_x + int(185 * scale), left_top),
        (left_x + left_w, left_top),
        (left_x + left_w, base_y),
    ]

    draw.polygon(
        left_points,
        fill=white,
        outline=black,
    )

    # Dachlinie
    draw.line(
        [
            (
                left_x + int(25 * scale),
                base_y - int(20 * scale),
            ),
            (
                left_x + int(125 * scale),
                left_roof_top + int(10 * scale),
            ),
            (
                left_x + int(225 * scale),
                base_y - int(20 * scale),
            ),
        ],
        fill=black,
        width=max(2, int(3 * scale)),
    )

    # Fenster
    for x in [85, 125, 165]:
        draw.line(
            [
                (
                    left_x + int(x * scale),
                    left_top + int(45 * scale),
                ),
                (
                    left_x + int(x * scale),
                    base_y - int(10 * scale),
                ),
            ],
            fill=black,
            width=max(1, int(2 * scale)),
        )

    draw.line(
        [
            (left_x + int(70 * scale), left_top + int(85 * scale)),
            (left_x + int(205 * scale), left_top + int(85 * scale)),
        ],
        fill=black,
        width=max(1, int(2 * scale)),
    )

    # --------------------------------------------------------
    # Mittleres großes Gebäude
    # --------------------------------------------------------

    center_w = int(390 * scale)
    center_x = (width - center_w) // 2

    center_top = base_y - int(205 * scale)
    center_roof_top = center_top - int(105 * scale)

    center_points = [
        (center_x, base_y),
        (center_x, center_top),
        (center_x + int(80 * scale), center_top),
        (center_x + center_w // 2, center_roof_top),
        (
            center_x + center_w - int(80 * scale),
            center_top,
        ),
        (center_x + center_w, center_top),
        (center_x + center_w, base_y),
    ]

    draw.polygon(
        center_points,
        fill=white,
        outline=black,
    )

    # großes Dach
    draw.line(
        [
            (
                center_x + int(35 * scale),
                base_y - int(10 * scale),
            ),
            (
                center_x + center_w // 2,
                center_roof_top + int(20 * scale),
            ),
            (
                center_x + center_w - int(35 * scale),
                base_y - int(10 * scale),
            ),
        ],
        fill=black,
        width=max(2, int(3 * scale)),
    )

    # Fenster des Mittelgebäudes
    window_top = center_top + int(50 * scale)
    window_bottom = base_y - int(10 * scale)

    for x in [70, 125, 195, 265, 320]:
        draw.line(
            [
                (
                    center_x + int(x * scale),
                    window_top,
                ),
                (
                    center_x + int(x * scale),
                    window_bottom,
                ),
            ],
            fill=black,
            width=max(1, int(2 * scale)),
        )

    for y in [90, 135]:
        draw.line(
            [
                (
                    center_x + int(55 * scale),
                    center_top + int(y * scale),
                ),
                (
                    center_x + center_w - int(55 * scale),
                    center_top + int(y * scale),
                ),
            ],
            fill=black,
            width=max(1, int(2 * scale)),
        )

    # --------------------------------------------------------
    # Rechtes Gebäude
    # --------------------------------------------------------

    right_x = int(width * 0.70)
    right_w = int(250 * scale)

    right_top = base_y - int(160 * scale)
    right_roof_top = right_top - int(55 * scale)

    right_points = [
        (right_x, base_y),
        (right_x, right_top),
        (right_x + int(65 * scale), right_top),
        (right_x + int(125 * scale), right_roof_top),
        (right_x + int(185 * scale), right_top),
        (right_x + right_w, right_top),
        (right_x + right_w, base_y),
    ]

    draw.polygon(
        right_points,
        fill=white,
        outline=black,
    )

    draw.line(
        [
            (
                right_x + int(25 * scale),
                base_y - int(20 * scale),
            ),
            (
                right_x + int(125 * scale),
                right_roof_top + int(10 * scale),
            ),
            (
                right_x + int(225 * scale),
                base_y - int(20 * scale),
            ),
        ],
        fill=black,
        width=max(2, int(3 * scale)),
    )

    for x in [85, 125, 165]:
        draw.line(
            [
                (
                    right_x + int(x * scale),
                    right_top + int(45 * scale),
                ),
                (
                    right_x + int(x * scale),
                    base_y - int(10 * scale),
                ),
            ],
            fill=black,
            width=max(1, int(2 * scale)),
        )

    draw.line(
        [
            (right_x + int(70 * scale), right_top + int(85 * scale)),
            (right_x + int(205 * scale), right_top + int(85 * scale)),
        ],
        fill=black,
        width=max(1, int(2 * scale)),
    )


# ============================================================
# HEADER-GRAFIK
# ============================================================

def create_header(title, background_color):
    """
    Erstellt den oberen Bereich der Skizzen als echtes Bild.
    Dadurch braucht Streamlit keinerlei HTML/CSS/SVG zu interpretieren.
    """

    width = 1600
    height = 440

    image = Image.new(
        "RGB",
        (width, height),
        background_color,
    )

    draw = ImageDraw.Draw(image)

    # --------------------------------------------------------
    # Titel
    # --------------------------------------------------------

    title_font = get_font(88, bold=True)

    bbox = draw.textbbox(
        (0, 0),
        title,
        font=title_font,
    )

    title_width = bbox[2] - bbox[0]

    title_x = (width - title_width) // 2
    title_y = 35

    draw.text(
        (title_x, title_y),
        title,
        fill=(0, 0, 0),
        font=title_font,
    )

    # --------------------------------------------------------
    # Schul-Silhouette
    # --------------------------------------------------------

    draw_school_silhouette(
        draw,
        width,
        base_y=425,
        scale=1.15,
    )

    # --------------------------------------------------------
    # Schwarze Grundlinie
    # --------------------------------------------------------

    draw.line(
        [(0, 425), (width, 425)],
        fill=(10, 10, 10),
        width=4,
    )

    # --------------------------------------------------------
    # Hamburger-Linien links wie in der Vorlage
    # --------------------------------------------------------

    for y in [20, 90, 160]:
        draw.line(
            [(25, y), (190, y)],
            fill=(10, 10, 10),
            width=6,
        )

    return image


# ============================================================
# LUPEN-GRAFIK
# ============================================================

def create_magnifying_glass():
    """
    Zeichnet die Lupe aus der ersten Skizze.
    """

    width = 500
    height = 430

    image = Image.new(
        "RGBA",
        (width, height),
        (255, 255, 255, 0),
    )

    draw = ImageDraw.Draw(image)

    black = (0, 0, 0, 255)
    white = (255, 255, 255, 255)

    # großer Kreis
    draw.ellipse(
        (100, 40, 380, 320),
        fill=white,
        outline=black,
        width=10,
    )

    # innerer Kreis
    draw.ellipse(
        (125, 65, 355, 295),
        fill=white,
        outline=black,
        width=8,
    )

    # Griff
    draw.line(
        [(145, 285), (65, 390)],
        fill=black,
        width=35,
    )

    draw.line(
        [(170, 305), (90, 405)],
        fill=white,
        width=20,
    )

    return image


# ============================================================
# QUADRATISCHE VORSCHAU
# ============================================================

def square_image(image, size=500):
    """
    Macht aus jedem Foto eine quadratische Vorschau.
    """

    image = image.convert("RGB")

    return ImageOps.fit(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


# ============================================================
# NAVIGATION
# ============================================================

def navigation():
    """
    Echtes Streamlit-Menü.
    Kein HTML und kein CSS.
    """

    left, right = st.columns([8, 1])

    with right:
        with st.popover("☰"):
            st.write("**Menü**")

            if st.button(
                "🔎 Suche",
                use_container_width=True,
            ):
                st.session_state.page = "Suche"
                st.rerun()

            if st.button(
                "📷 Bild hochladen",
                use_container_width=True,
            ):
                st.session_state.page = "Hochladen"
                st.rerun()

            if st.button(
                "🕐 Älteste Fundstücke",
                use_container_width=True,
            ):
                st.session_state.page = "Älteste"
                st.rerun()


# ============================================================
# SEITE 1 – SUCHE
# ============================================================

def search_page():

    navigation()

    # Kopfbereich aus der Vorlage
    header = create_header(
        "Fundgrube",
        (242, 133, 143),
    )

    st.image(
        header,
        use_container_width=True,
    )

    # Lupe
    left, center, right = st.columns([1, 2, 1])

    with center:
        st.image(
            create_magnifying_glass(),
            width=240,
        )

    # Suchfeld
    search_text = st.text_input(
        "Suche",
        placeholder="Beschreibe dein verlorenes Kleidungsstück",
        label_visibility="collapsed",
    )

    st.write("")

    # --------------------------------------------------------
    # Ergebnisse
    # --------------------------------------------------------

    if not search_text.strip():

        st.info(
            "Beschreibe oben dein verlorenes Kleidungsstück, "
            "z. B. „schwarzer Hoodie“ oder „blaue Hose“."
        )

        return

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    search = f"%{search_text.strip()}%"

    cursor.execute(
        """
        SELECT id, item_type, location, found_date, image_path
        FROM items
        WHERE
            item_type LIKE ?
            OR location LIKE ?
        ORDER BY found_date ASC
        """,
        (search, search),
    )

    results = cursor.fetchall()

    connection.close()

    if not results:
        st.warning(
            "Leider wurde kein passender Fund gefunden."
        )
        return

    st.subheader("Gefundene Gegenstände")

    columns = st.columns(3)

    for index, item in enumerate(results):

        item_id, item_type, location, found_date, image_path = item

        with columns[index % 3]:

            with st.container(border=True):

                if os.path.exists(image_path):

                    try:
                        image = Image.open(image_path)

                        st.image(
                            square_image(image),
                            use_container_width=True,
                        )

                    except Exception:
                        st.warning("Bild konnte nicht geladen werden.")

                st.write(f"**{item_type}**")
                st.write(f"📍 {location}")
                st.write(f"📅 {found_date}")


# ============================================================
# SEITE 2 – HOCHLADEN
# ============================================================

def upload_page():

    navigation()

    # Kopfbereich passend zur zweiten Vorlage
    header = create_header(
        "Lade ein Bild hoch",
        (154, 216, 244),
    )

    st.image(
        header,
        use_container_width=True,
    )

    st.write("")

    st.subheader("📷 Foto auswählen")

    uploaded_file = st.file_uploader(
        "Lade ein Foto aus deiner Mediathek hoch",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        label_visibility="visible",
    )

    if uploaded_file is None:

        st.info(
            "Wähle ein Foto aus deiner Mediathek aus. "
            "Die KI erkennt anschließend automatisch das Kleidungsstück."
        )

        return

    # --------------------------------------------------------
    # Bild anzeigen
    # --------------------------------------------------------

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error(
            "Das Bild konnte nicht geöffnet werden."
        )
        return

    st.subheader("Dein Bild")

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.image(
            image,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # KI-KLASSIFIZIERUNG
    # --------------------------------------------------------

    st.subheader("🤖 KI-Erkennung")

    progress = st.progress(0)

    status = st.empty()

    status.write("Bild wird vorbereitet …")
    progress.progress(25)

    try:

        status.write("KI analysiert das Bild …")

        label, confidence = classify_image(image)

        progress.progress(100)

        status.empty()

        confidence_percent = confidence * 100

        st.success(
            f"Erkannt: **{label}**"
        )

        st.write(
            f"Erkennungswahrscheinlichkeit: "
            f"**{confidence_percent:.1f} %**"
        )

        # Hinweis bei geringer Sicherheit
        if confidence < 0.50:
            st.warning(
                "Die KI ist sich bei der Erkennung nicht ganz sicher. "
                "Bitte überprüfe die erkannte Kategorie."
            )

    except Exception as error:

        progress.empty()

        st.error(
            "Die KI konnte das Bild nicht analysieren."
        )

        st.exception(error)

        return

    # --------------------------------------------------------
    # FUNDORT UND DATUM
    # --------------------------------------------------------

    st.subheader("📍 Fundstück speichern")

    location = st.text_input(
        "Wo wurde der Gegenstand gefunden?",
        placeholder="z. B. Sporthalle, Aula, Raum 203",
    )

    found_date = st.date_input(
        "Wann wurde er gefunden?",
        value=date.today(),
    )

    # --------------------------------------------------------
    # SPEICHERN
    # --------------------------------------------------------

    if st.button(
        "Fundstück speichern",
        type="primary",
        use_container_width=True,
    ):

        if not location.strip():
            st.warning(
                "Bitte gib zuerst den Fundort ein."
            )
            return

        # eindeutiger Dateiname
        filename = (
            f"{uuid.uuid4().hex}.jpg"
        )

        image_path = UPLOAD_DIR / filename

        try:

            image.save(
                image_path,
                format="JPEG",
                quality=90,
            )

            connection = sqlite3.connect(DB_PATH)
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
                "Das Fundstück wurde erfolgreich gespeichert!"
            )

        except Exception as error:

            st.error(
                "Das Fundstück konnte nicht gespeichert werden."
            )

            st.exception(error)


# ============================================================
# SEITE 3 – ÄLTESTE FUNDSTÜCKE
# ============================================================

def oldest_page():

    navigation()

    header = create_header(
        "Älteste Fundstücke",
        (242, 133, 143),
    )

    st.image(
        header,
        use_container_width=True,
    )

    st.write("")

    st.subheader(
        "Die 9 ältesten gefundenen Gegenstände"
    )

    connection = sqlite3.connect(DB_PATH)
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
        ORDER BY found_date ASC, id ASC
        LIMIT 9
        """
    )

    items = cursor.fetchall()

    connection.close()

    if not items:

        st.info(
            "Es wurden noch keine Fundstücke gespeichert."
        )

        return

    # Immer 3 Spalten → 3 x 3 Karten
    columns = st.columns(3)

    for index, item in enumerate(items):

        item_id, item_type, location, found_date, image_path = item

        with columns[index % 3]:

            with st.container(border=True):

                if os.path.exists(image_path):

                    try:

                        image = Image.open(image_path)

                        st.image(
                            square_image(image),
                            use_container_width=True,
                        )

                    except Exception:

                        st.warning(
                            "Bild konnte nicht geladen werden."
                        )

                st.write(
                    f"**{item_type}**"
                )

                st.write(
                    f"📍 {location}"
                )

                st.write(
                    f"📅 {found_date}"
                )


# ============================================================
# HAUPTPROGRAMM
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
