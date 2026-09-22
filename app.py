import os
import re
import sqlite3
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import streamlit as st
import tf_keras
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# FUNDGRUBE
# Streamlit App
# ============================================================


# ============================================================
# SEITENEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DATEIEN
# ============================================================

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

DB_PATH = "fundgrube.db"

UPLOAD_DIR = Path("fundgrube_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# FARBEN
# ============================================================

# Hellblau
LIGHT_BLUE = (175, 225, 245)

# Etwas dunkleres Hellblau
LIGHT_BLUE_DARK = (145, 210, 238)

# Metallic-Grau
METALLIC = (105, 110, 115)

# Helleres Metallic-Grau
METALLIC_LIGHT = (135, 140, 145)

# Sehr dunkles Grau
METALLIC_DARK = (75, 80, 85)

# Weiß
WHITE = (255, 255, 255)

# Schwarzer Akzent
BLACK = (15, 15, 15)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Suche"


# ============================================================
# LABELS AUS labels.txt
# ============================================================

def load_labels():

    labels = []

    if not os.path.exists(LABELS_PATH):

        return [
            "Hose",
            "Schuh",
            "T-Shirt",
            "Hoodie",
        ]

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Beispiel:
            # 0 Hose
            # 1 Schuh
            # 2 T-Shirt
            # 3 Hoodie

            match = re.match(
                r"^\s*\d+\s+(.*)$",
                line,
            )

            if match:

                labels.append(
                    match.group(1).strip()
                )

            else:

                labels.append(line)

    return labels


LABELS = load_labels()


# ============================================================
# DATENBANK INITIALISIEREN
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
# KI-MODELL LADEN
# ============================================================

@st.cache_resource
def load_model():

    model = tf_keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    return model


# ============================================================
# BILD KLASSIFIZIEREN
# ============================================================

def classify_image(image):

    model = load_model()

    # RGB
    image = image.convert("RGB")

    # Teachable Machine verwendet 224 x 224
    image = image.resize(
        (224, 224)
    )

    # Numpy
    image_array = np.asarray(
        image
    ).astype(np.float32)

    # Teachable-Machine-Preprocessing
    image_array = (
        image_array / 127.5
    ) - 1.0

    # Batch-Dimension
    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    # Vorhersage
    prediction = model.predict(
        image_array,
        verbose=0,
    )

    probabilities = prediction[0]

    # Höchste Wahrscheinlichkeit
    index = int(
        np.argmax(probabilities)
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
# SCHRIFT
# ============================================================

def get_font(
    size,
    bold=False,
):

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

                return ImageFont.truetype(
                    font_path,
                    size,
                )

            except Exception:

                pass

    return ImageFont.load_default()


# ============================================================
# SCHULGEBÄUDE
# ============================================================

def draw_school_silhouette(
    draw,
    width,
    base_y,
    scale=1.0,
):

    metallic = METALLIC
    white = WHITE

    # ========================================================
    # LINKES GEBÄUDE
    # ========================================================

    left_x = int(
        width * 0.17
    )

    left_w = int(
        245 * scale
    )

    left_top = (
        base_y
        - int(150 * scale)
    )

    left_roof_top = (
        left_top
        - int(55 * scale)
    )

    left_points = [

        (
            left_x,
            base_y,
        ),

        (
            left_x,
            left_top,
        ),

        (
            left_x
            + int(65 * scale),
            left_top,
        ),

        (
            left_x
            + int(122 * scale),
            left_roof_top,
        ),

        (
            left_x
            + int(180 * scale),
            left_top,
        ),

        (
            left_x
            + left_w,
            left_top,
        ),

        (
            left_x
            + left_w,
            base_y,
        ),
    ]

    draw.polygon(
        left_points,
        fill=metallic,
        outline=metallic,
    )

    # Dach
    draw.line(
        [
            (
                left_x
                + int(25 * scale),
                base_y
                - int(20 * scale),
            ),
            (
                left_x
                + int(122 * scale),
                left_roof_top
                + int(10 * scale),
            ),
            (
                left_x
                + int(220 * scale),
                base_y
                - int(20 * scale),
            ),
        ],
        fill=white,
        width=max(
            2,
            int(4 * scale),
        ),
    )

    # Fenster
    for x in [
        85,
        122,
        160,
    ]:

        draw.line(
            [
                (
                    left_x
                    + int(x * scale),
                    left_top
                    + int(42 * scale),
                ),
                (
                    left_x
                    + int(x * scale),
                    base_y
                    - int(10 * scale),
                ),
            ],
            fill=white,
            width=max(
                1,
                int(3 * scale),
            ),
        )

    draw.line(
        [
            (
                left_x
                + int(65 * scale),
                left_top
                + int(82 * scale),
            ),
            (
                left_x
                + int(205 * scale),
                left_top
                + int(82 * scale),
            ),
        ],
        fill=white,
        width=max(
            1,
            int(3 * scale),
        ),
    )

    # ========================================================
    # MITTLERES GEBÄUDE
    # ========================================================

    center_w = int(
        370 * scale
    )

    center_x = (
        width - center_w
    ) // 2

    center_top = (
        base_y
        - int(190 * scale)
    )

    center_roof_top = (
        center_top
        - int(95 * scale)
    )

    center_points = [

        (
            center_x,
            base_y,
        ),

        (
            center_x,
            center_top,
        ),

        (
            center_x
            + int(75 * scale),
            center_top,
        ),

        (
            center_x
            + center_w // 2,
            center_roof_top,
        ),

        (
            center_x
            + center_w
            - int(75 * scale),
            center_top,
        ),

        (
            center_x
            + center_w,
            center_top,
        ),

        (
            center_x
            + center_w,
            base_y,
        ),
    ]

    draw.polygon(
        center_points,
        fill=metallic,
        outline=metallic,
    )

    # Dach
    draw.line(
        [
            (
                center_x
                + int(35 * scale),
                base_y
                - int(10 * scale),
            ),
            (
                center_x
                + center_w // 2,
                center_roof_top
                + int(18 * scale),
            ),
            (
                center_x
                + center_w
                - int(35 * scale),
                base_y
                - int(10 * scale),
            ),
        ],
        fill=white,
        width=max(
            2,
            int(4 * scale),
        ),
    )

    # Fenster
    window_top = (
        center_top
        + int(45 * scale)
    )

    window_bottom = (
        base_y
        - int(10 * scale)
    )

    for x in [
        65,
        115,
        185,
        255,
        305,
    ]:

        draw.line(
            [
                (
                    center_x
                    + int(x * scale),
                    window_top,
                ),
                (
                    center_x
                    + int(x * scale),
                    window_bottom,
                ),
            ],
            fill=white,
            width=max(
                1,
                int(3 * scale),
            ),
        )

    for y in [
        85,
        125,
    ]:

        draw.line(
            [
                (
                    center_x
                    + int(50 * scale),
                    center_top
                    + int(y * scale),
                ),
                (
                    center_x
                    + center_w
                    - int(50 * scale),
                    center_top
                    + int(y * scale),
                ),
            ],
            fill=white,
            width=max(
                1,
                int(3 * scale),
            ),
        )

    # ========================================================
    # RECHTES GEBÄUDE
    # ========================================================

    right_x = int(
        width * 0.70
    )

    right_w = int(
        245 * scale
    )

    right_top = (
        base_y
        - int(150 * scale)
    )

    right_roof_top = (
        right_top
        - int(55 * scale)
    )

    right_points = [

        (
            right_x,
            base_y,
        ),

        (
            right_x,
            right_top,
        ),

        (
            right_x
            + int(65 * scale),
            right_top,
        ),

        (
            right_x
            + int(122 * scale),
            right_roof_top,
        ),

        (
            right_x
            + int(180 * scale),
            right_top,
        ),

        (
            right_x
            + right_w,
            right_top,
        ),

        (
            right_x
            + right_w,
            base_y,
        ),
    ]

    draw.polygon(
        right_points,
        fill=metallic,
        outline=metallic,
    )

    # Dach
    draw.line(
        [
            (
                right_x
                + int(25 * scale),
                base_y
                - int(20 * scale),
            ),
            (
                right_x
                + int(122 * scale),
                right_roof_top
                + int(10 * scale),
            ),
            (
                right_x
                + int(220 * scale),
                base_y
                - int(20 * scale),
            ),
        ],
        fill=white,
        width=max(
            2,
            int(4 * scale),
        ),
    )

    # Fenster
    for x in [
        85,
        122,
        160,
    ]:

        draw.line(
            [
                (
                    right_x
                    + int(x * scale),
                    right_top
                    + int(42 * scale),
                ),
                (
                    right_x
                    + int(x * scale),
                    base_y
                    - int(10 * scale),
                ),
            ],
            fill=white,
            width=max(
                1,
                int(3 * scale),
            ),
        )

    draw.line(
        [
            (
                right_x
                + int(65 * scale),
                right_top
                + int(82 * scale),
            ),
            (
                right_x
                + int(205 * scale),
                right_top
                + int(82 * scale),
            ),
        ],
        fill=white,
        width=max(
            1,
            int(3 * scale),
        ),
    )


# ============================================================
# HEADER
# ============================================================

def create_header(
    title,
    background_color=LIGHT_BLUE,
):

    width = 1600
    height = 470

    image = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        background_color,
    )

    draw = ImageDraw.Draw(
        image
    )

    # ========================================================
    # TITEL
    # ========================================================

    title_font = get_font(
        76,
        bold=True,
    )

    bbox = draw.textbbox(
        (0, 0),
        title,
        font=title_font,
    )

    title_width = (
        bbox[2] - bbox[0]
    )

    title_x = (
        width - title_width
    ) // 2

    title_y = 30

    # Metallic-Fläche hinter dem Titel
    draw.rounded_rectangle(
        (
            title_x - 35,
            title_y - 8,
            title_x
            + title_width
            + 35,
            title_y + 105,
        ),
        radius=25,
        fill=METALLIC,
    )

    # Weiße Schrift
    draw.text(
        (
            title_x,
            title_y,
        ),
        title,
        fill=WHITE,
        font=title_font,
    )

    # ========================================================
    # SCHULE
    # ========================================================

    draw_school_silhouette(
        draw,
        width,
        base_y=455,
        scale=0.72,
    )

    # ========================================================
    # GRUNDLINIE
    # ========================================================

    draw.line(
        [
            (0, 455),
            (width, 455),
        ],
        fill=METALLIC_DARK,
        width=8,
    )

    return image


# ============================================================
# LUPE
# ============================================================

def create_magnifying_glass():

    width = 500
    height = 430

    image = Image.new(
        "RGBA",
        (
            width,
            height,
        ),
        (
            0,
            0,
            0,
            0,
        ),
    )

    draw = ImageDraw.Draw(
        image
    )

    # ========================================================
    # ÄUSSERER RING
    # ========================================================

    draw.ellipse(
        (
            100,
            40,
            380,
            320,
        ),
        fill=METALLIC,
        outline=METALLIC,
        width=10,
    )

    # ========================================================
    # INNERER BEREICH
    # ========================================================

    draw.ellipse(
        (
            125,
            65,
            355,
            295,
        ),
        fill=WHITE,
        outline=METALLIC_DARK,
        width=7,
    )

    # ========================================================
    # GRIFF
    # ========================================================

    draw.line(
        [
            (145, 285),
            (65, 390),
        ],
        fill=METALLIC,
        width=42,
    )

    draw.line(
        [
            (150, 300),
            (82, 395),
        ],
        fill=WHITE,
        width=18,
    )

    return image


# ============================================================
# QUADRATISCHES BILD
# ============================================================

def square_image(
    image,
    size=500,
):

    image = image.convert(
        "RGB"
    )

    return ImageOps.fit(
        image,
        (
            size,
            size,
        ),
        method=Image.Resampling.LANCZOS,
        centering=(
            0.5,
            0.5,
        ),
    )


# ============================================================
# NAVIGATION
# ============================================================

def navigation():

    left, right = st.columns(
        [8, 1]
    )

    with right:

        with st.popover("☰"):

            st.write(
                "**Fundgrube**"
            )

            st.write(
                "Navigation"
            )

            if st.button(
                "🔎 Suche",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "Suche"
                )

                st.rerun()

            if st.button(
                "📷 Bild hochladen",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "Hochladen"
                )

                st.rerun()

            if st.button(
                "🕐 Älteste Fundstücke",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "Älteste"
                )

                st.rerun()


# ============================================================
# SEITE 1 – SUCHE
# ============================================================

def search_page():

    navigation()

    # ========================================================
    # HEADER
    # ========================================================

    header = create_header(
        "Fundgrube",
        LIGHT_BLUE,
    )

    st.image(
        header,
        use_container_width=True,
    )

    # ========================================================
    # LUPE
    # ========================================================

    left, center, right = st.columns(
        [1, 2, 1]
    )

    with center:

        st.image(
            create_magnifying_glass(),
            width=230,
        )

    # ========================================================
    # SUCHFELD
    # ========================================================

    search_text = st.text_input(
        "Suche",
        placeholder=(
            "Beschreibe dein verlorenes Kleidungsstück"
        ),
        label_visibility="collapsed",
    )

    # ========================================================
    # KEINE SUCHE
    # ========================================================

    if not search_text.strip():

        return

    # ========================================================
    # DATENBANK
    # ========================================================

    connection = sqlite3.connect(
        DB_PATH
    )

    cursor = connection.cursor()

    search = (
        "%"
        + search_text.strip()
        + "%"
    )

    cursor.execute(
        """
        SELECT
            id,
            item_type,
            location,
            found_date,
            image_path
        FROM items
        WHERE
            item_type LIKE ?
            OR location LIKE ?
        ORDER BY found_date ASC
        """,
        (
            search,
            search,
        ),
    )

    results = cursor.fetchall()

    connection.close()

    # ========================================================
    # KEIN TREFFER
    # ========================================================

    if not results:

        st.warning(
            "Kein passendes Fundstück gefunden."
        )

        return

    # ========================================================
    # TREFFER
    # ========================================================

    st.subheader(
        "Gefundene Gegenstände"
    )

    columns = st.columns(3)

    for index, item in enumerate(
        results
    ):

        (
            item_id,
            item_type,
            location,
            found_date,
            image_path,
        ) = item

        with columns[
            index % 3
        ]:

            with st.container(
                border=True
            ):

                # Bild
                if os.path.exists(
                    image_path
                ):

                    try:

                        found_image = (
                            Image.open(
                                image_path
                            )
                        )

                        st.image(
                            square_image(
                                found_image
                            ),
                            use_container_width=True,
                        )

                    except Exception:

                        st.warning(
                            "Bild konnte nicht geladen werden."
                        )

                # Informationen
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
# SEITE 2 – HOCHLADEN
# ============================================================

def upload_page():

    navigation()

    # ========================================================
    # HEADER
    # ========================================================

    header = create_header(
        "Lade ein Bild hoch",
        LIGHT_BLUE,
    )

    st.image(
        header,
        use_container_width=True,
    )

    st.write("")

    st.subheader(
        "📷 Foto auswählen"
    )

    # ========================================================
    # UPLOAD
    # ========================================================

    uploaded_file = st.file_uploader(
        "Lade ein Foto aus deiner Mediathek hoch",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    # Kein Bild
    if uploaded_file is None:

        return

    # ========================================================
    # BILD ÖFFNEN
    # ========================================================

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Das Bild konnte nicht geöffnet werden."
        )

        return

    # ========================================================
    # BILD ANZEIGEN
    # ========================================================

    st.subheader(
        "Dein Bild"
    )

    left, center, right = st.columns(
        [1, 2, 1]
    )

    with center:

        st.image(
            image,
            use_container_width=True,
        )

    # ========================================================
    # KI
    # ========================================================

    st.subheader(
        "🤖 KI-Erkennung"
    )

    progress = st.progress(0)

    status = st.empty()

    status.write(
        "Bild wird vorbereitet …"
    )

    progress.progress(20)

    try:

        status.write(
            "KI analysiert das Bild …"
        )

        progress.progress(50)

        label, confidence = (
            classify_image(image)
        )

        progress.progress(100)

        status.empty()

        confidence_percent = (
            confidence * 100
        )

        st.success(
            f"Erkannt: **{label}**"
        )

        st.write(
            "Erkennungswahrscheinlichkeit: "
            f"**{confidence_percent:.1f} %**"
        )

        if confidence < 0.50:

            st.warning(
                "Die KI ist sich bei der "
                "Erkennung nicht ganz sicher."
            )

    except Exception as error:

        progress.empty()

        status.empty()

        st.error(
            "Die KI konnte das Bild nicht analysieren."
        )

        st.exception(
            error
        )

        return

    # ========================================================
    # FUNDORT
    # ========================================================

    st.subheader(
        "📍 Fundstück speichern"
    )

    location = st.text_input(
        "Wo wurde der Gegenstand gefunden?",
        placeholder=(
            "z. B. Sporthalle, Aula, Raum 203"
        ),
    )

    # ========================================================
    # DATUM
    # ========================================================

    found_date = st.date_input(
        "Wann wurde er gefunden?",
        value=date.today(),
    )

    # ========================================================
    # SPEICHERN
    # ========================================================

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

        # ====================================================
        # BILD SPEICHERN
        # ====================================================

        filename = (
            uuid.uuid4().hex
            + ".jpg"
        )

        image_path = (
            UPLOAD_DIR
            / filename
        )

        try:

            image.save(
                image_path,
                format="JPEG",
                quality=90,
            )

            # =================================================
            # DATENBANK
            # =================================================

            connection = sqlite3.connect(
                DB_PATH
            )

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

            st.exception(
                error
            )


# ============================================================
# SEITE 3 – ÄLTESTE FUNDSTÜCKE
# ============================================================

def oldest_page():

    navigation()

    # ========================================================
    # HEADER
    # ========================================================

    header = create_header(
        "Älteste Fundstücke",
        LIGHT_BLUE,
    )

    st.image(
        header,
        use_container_width=True,
    )

    st.write("")

    st.subheader(
        "Die 9 ältesten gefundenen Gegenstände"
    )

    # ========================================================
    # DATENBANK
    # ========================================================

    connection = sqlite3.connect(
        DB_PATH
    )

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
        ORDER BY
            found_date ASC,
            id ASC
        LIMIT 9
        """
    )

    items = cursor.fetchall()

    connection.close()

    # ========================================================
    # NOCH KEINE FUNDSTÜCKE
    # ========================================================

    if not items:

        st.info(
            "Es wurden noch keine Fundstücke gespeichert."
        )

        return

    # ========================================================
    # 3 x 3 KARTEN
    # ========================================================

    columns = st.columns(3)

    for index, item in enumerate(
        items
    ):

        (
            item_id,
            item_type,
            location,
            found_date,
            image_path,
        ) = item

        with columns[
            index % 3
        ]:

            with st.container(
                border=True
            ):

                # Bild
                if os.path.exists(
                    image_path
                ):

                    try:

                        found_image = (
                            Image.open(
                                image_path
                            )
                        )

                        st.image(
                            square_image(
                                found_image
                            ),
                            use_container_width=True,
                        )

                    except Exception:

                        st.warning(
                            "Bild konnte nicht geladen werden."
                        )

                # Text
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
