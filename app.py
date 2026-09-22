import os
import re
import sqlite3
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import streamlit as st
import tf_keras
from PIL import Image, ImageOps


# ============================================================
# KONFIGURATION
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
    """
    Liest labels.txt ein.

    Unterstützt zum Beispiel:
        0 Hose
        1 Schuh
        2 T-Shirt
        3 Hoodie
    """

    fallback = [
        "Hose",
        "Schuh",
        "T-Shirt",
        "Hoodie",
    ]

    if not os.path.exists(LABELS_PATH):
        return fallback

    labels = []

    try:
        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                # Nummer am Anfang entfernen
                match = re.match(
                    r"^\s*\d+\s+(.*)$",
                    line,
                )

                if match:
                    label = match.group(1).strip()
                else:
                    label = line

                labels.append(label)

    except Exception:
        return fallback

    return labels if labels else fallback


LABELS = load_labels()


# ============================================================
# DATENBANK
# ============================================================

def get_connection():
    """
    Erstellt eine Verbindung zur lokalen SQLite-Datenbank.
    """

    return sqlite3.connect(
        DB_PATH,
        timeout=10,
    )


def init_database():

    connection = get_connection()
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
# KI-MODELL
# ============================================================

@st.cache_resource
def load_model():
    """
    Lädt das Teachable-Machine-Modell nur einmal.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Die Datei '{MODEL_PATH}' wurde nicht gefunden."
        )

    return tf_keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


# ============================================================
# KI-KLASSIFIZIERUNG
# ============================================================

def classify_image(image):
    """
    Klassifiziert ein Bild mit dem Teachable-Machine-Modell.
    """

    model = load_model()

    # RGB erzwingen
    image = image.convert("RGB")

    # Teachable Machine: 224 x 224
    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    # Bild -> numpy
    image_array = np.asarray(
        image
    ).astype(np.float32)

    # Teachable-Machine-Normalisierung
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
    class_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_index]
    )

    # Label bestimmen
    if class_index < len(LABELS):
        label = LABELS[class_index]
    else:
        label = f"Klasse {class_index}"

    return label, confidence


# ============================================================
# BILDER
# ============================================================

def make_square_image(
    image,
    size=500,
):
    """
    Schneidet ein Bild quadratisch zu,
    ohne es zu verzerren.
    """

    image = image.convert("RGB")

    return ImageOps.fit(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


# ============================================================
# SUCHBEGRIFFE
# ============================================================

def normalize_text(text):
    """
    Vereinheitlicht Suchtext.

    Beispiel:
        'Schwarzer Hoodie'
        ->
        ['schwarzer', 'hoodie']
    """

    text = text.lower()

    # Sonderzeichen entfernen
    text = re.sub(
        r"[^a-zäöüß0-9\- ]",
        " ",
        text,
    )

    words = [
        word.strip()
        for word in text.split()
        if word.strip()
    ]

    return words


# ============================================================
# NAVIGATION
# ============================================================

def show_navigation():

    with st.sidebar:

        st.title("Fundgrube")

        st.divider()

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

        st.divider()

        st.caption(
            "Fundgrube – Schul-Fundstücke"
        )


# ============================================================
# SEITENKOPF
# ============================================================

def page_header(
    title,
    description=None,
):

    st.title(title)

    if description:
        st.caption(description)

    st.divider()


# ============================================================
# SEITE: SUCHE
# ============================================================

def search_page():

    show_navigation()

    page_header(
        "🔎 Fundgrube",
        "Finde verlorene Kleidungsstücke wieder.",
    )

    # --------------------------------------------------------
    # Suchfeld
    # --------------------------------------------------------

    search_text = st.text_input(
        "Was hast du verloren?",
        placeholder=(
            "z. B. Hoodie, schwarze Hose, Schuh ..."
        ),
    )

    # Noch keine Suche
    if not search_text.strip():

        st.info(
            "Gib oben ein Kleidungsstück oder einen "
            "Begriff ein, um nach einem Fundstück zu suchen."
        )

        return

    # --------------------------------------------------------
    # Suchbegriffe
    # --------------------------------------------------------

    words = normalize_text(
        search_text
    )

    if not words:
        return

    # --------------------------------------------------------
    # Datenbank
    # --------------------------------------------------------

    connection = get_connection()
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
        """
    )

    all_items = cursor.fetchall()

    connection.close()

    # --------------------------------------------------------
    # Treffer suchen
    # --------------------------------------------------------

    results = []

    for item in all_items:

        (
            item_id,
            item_type,
            location,
            found_date,
            image_path,
        ) = item

        searchable_text = (
            f"{item_type} {location}"
        ).lower()

        # Treffer, wenn mindestens ein Suchbegriff
        # vorkommt.
        if any(
            word in searchable_text
            for word in words
        ):
            results.append(item)

    # --------------------------------------------------------
    # Ergebnis
    # --------------------------------------------------------

    if not results:

        st.warning(
            "Kein passendes Fundstück gefunden."
        )

        return

    st.subheader(
        f"{len(results)} Fundstück"
        + (" gefunden" if len(results) == 1 else "e gefunden")
    )

    # --------------------------------------------------------
    # Karten
    # --------------------------------------------------------

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

                        image = Image.open(
                            image_path
                        )

                        st.image(
                            make_square_image(
                                image
                            ),
                            use_container_width=True,
                        )

                    except Exception:

                        st.warning(
                            "Bild konnte nicht geladen werden."
                        )

                else:

                    st.info(
                        "Kein Bild verfügbar."
                    )

                st.subheader(
                    item_type
                )

                st.write(
                    f"📍 **Fundort:** {location}"
                )

                st.write(
                    f"📅 **Gefunden am:** {found_date}"
                )


# ============================================================
# SEITE: HOCHLADEN
# ============================================================

def upload_page():

    show_navigation()

    page_header(
        "📷 Fundstück hochladen",
        "Lade ein Bild hoch und lass es automatisch erkennen.",
    )

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Foto aus deiner Mediathek auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded_file is None:

        st.info(
            "Wähle ein Foto aus, um die KI-Erkennung zu starten."
        )

        return

    # --------------------------------------------------------
    # Bild öffnen
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Das Bild konnte nicht geöffnet werden."
        )

        return

    # --------------------------------------------------------
    # Bild anzeigen
    # --------------------------------------------------------

    st.subheader(
        "Hochgeladenes Bild"
    )

    image_column, info_column = st.columns(
        [2, 1]
    )

    with image_column:

        st.image(
            image,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # KI
    # --------------------------------------------------------

    with info_column:

        st.subheader(
            "🤖 KI-Erkennung"
        )

        progress = st.progress(
            0
        )

        status = st.empty()

        try:

            status.write(
                "Bild wird vorbereitet ..."
            )

            progress.progress(
                25
            )

            status.write(
                "KI analysiert das Bild ..."
            )

            progress.progress(
                60
            )

            label, confidence = (
                classify_image(image)
            )

            progress.progress(
                100
            )

            status.empty()

            confidence_percent = (
                confidence * 100
            )

            st.success(
                f"Erkannt: **{label}**"
            )

            st.metric(
                "Sicherheit",
                f"{confidence_percent:.1f} %",
            )

            if confidence < 0.50:

                st.warning(
                    "Die KI ist bei dieser Erkennung "
                    "nicht besonders sicher."
                )

        except Exception as error:

            progress.empty()
            status.empty()

            st.error(
                "Bei der KI-Erkennung ist ein Fehler aufgetreten."
            )

            st.exception(
                error
            )

            return

    st.divider()

    # --------------------------------------------------------
    # Fundinformationen
    # --------------------------------------------------------

    st.subheader(
        "📍 Fundinformationen"
    )

    location = st.text_input(
        "Wo wurde der Gegenstand gefunden?",
        placeholder=(
            "z. B. Sporthalle, Aula oder Raum 203"
        ),
    )

    found_date = st.date_input(
        "Wann wurde der Gegenstand gefunden?",
        value=date.today(),
    )

    # --------------------------------------------------------
    # Speichern
    # --------------------------------------------------------

    save = st.button(
        "💾 Fundstück speichern",
        type="primary",
        use_container_width=True,
    )

    if not save:
        return

    # Fundort überprüfen
    if not location.strip():

        st.warning(
            "Bitte gib zuerst den Fundort ein."
        )

        return

    # --------------------------------------------------------
    # Bild speichern
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Datenbank speichern
        # ----------------------------------------------------

        connection = get_connection()
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
            "✅ Das Fundstück wurde erfolgreich gespeichert!"
        )

        st.info(
            f"Gespeichert als: {label} · {location}"
        )

    except Exception as error:

        st.error(
            "Das Fundstück konnte nicht gespeichert werden."
        )

        st.exception(
            error
        )


# ============================================================
# SEITE: ÄLTESTE FUNDSTÜCKE
# ============================================================

def oldest_page():

    show_navigation()

    page_header(
        "🕐 Älteste Fundstücke",
        "Hier werden die neun ältesten Fundstücke angezeigt.",
    )

    # --------------------------------------------------------
    # Datenbank
    # --------------------------------------------------------

    connection = get_connection()
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

    # --------------------------------------------------------
    # Keine Fundstücke
    # --------------------------------------------------------

    if not items:

        st.info(
            "Es wurden bisher noch keine Fundstücke gespeichert."
        )

        return

    # --------------------------------------------------------
    # 3 x 3 Karten
    # --------------------------------------------------------

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

                        image = Image.open(
                            image_path
                        )

                        st.image(
                            make_square_image(
                                image
                            ),
                            use_container_width=True,
                        )

                    except Exception:

                        st.warning(
                            "Bild konnte nicht geladen werden."
                        )

                else:

                    st.info(
                        "Kein Bild verfügbar."
                    )

                # Daten
                st.subheader(
                    item_type
                )

                st.write(
                    f"📍 **Fundort:** {location}"
                )

                st.write(
                    f"📅 **Gefunden am:** {found_date}"
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
