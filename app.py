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


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    layout="wide"
)


# ============================================================
# DATEIEN UND ORDNER
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
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

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


    .block-container {
        max-width: 1500px !important;

        padding-top: 5.5rem !important;
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

        font-size: 1.15rem !important;
        font-weight: 900 !important;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.20),
            0 6px 14px rgba(0,0,0,0.22) !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        filter: brightness(1.10) !important;
    }


    /* ========================================================
       ALLGEMEINE GRAUE CONTAINER
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
        font-weight: 900 !important;
    }

    .stMarkdown p {
        font-size: 1.15rem !important;
    }


    /* ========================================================
       SUCHFELD
       ======================================================== */

    .st-key-search_hover {

        width: 100% !important;

        min-height: 145px !important;

        padding: 0 !important;
        margin-bottom: 2rem !important;

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

        overflow: hidden !important;

        transition:
            min-height 0.30s ease,
            transform 0.30s ease,
            box-shadow 0.30s ease !important;
    }

    .st-key-search_hover .search-title {

        min-height: 145px !important;

        width: 100% !important;

        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;

        box-sizing: border-box !important;

        padding: 1.5rem 2.5rem !important;

        color: white !important;

        font-size: 2.8rem !important;

        line-height: 1.15 !important;

        font-weight: 900 !important;

        text-align: left !important;
    }

    .st-key-search_hover:hover {

        min-height: 780px !important;

        transform: scale(1.015) !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.28),
            0 22px 45px rgba(0,0,0,0.40) !important;
    }

    .st-key-search_hover:hover .search-title {

        min-height: 100px !important;

        font-size: 2.4rem !important;

        padding: 1.5rem 2rem !important;
    }


    /* ========================================================
       SUCHBESCHREIBUNG
       ======================================================== */

    .st-key-search_hover [data-testid="stTextArea"] {

        max-height: 0 !important;
        opacity: 0 !important;

        overflow: hidden !important;

        pointer-events: none !important;

        transform: translateY(-20px) !important;

        margin: 0 !important;

        transition:
            max-height 0.35s ease,
            opacity 0.25s ease,
            transform 0.35s ease,
            margin 0.35s ease !important;
    }

    .st-key-search_hover:hover [data-testid="stTextArea"] {

        max-height: 300px !important;

        opacity: 1 !important;

        overflow: visible !important;

        pointer-events: auto !important;

        transform: translateY(0) !important;

        margin: 0 2rem 1rem 2rem !important;
    }

    .st-key-search_hover [data-testid="stTextArea"] label {
        display: none !important;
    }

    .st-key-search_hover [data-testid="stTextArea"] textarea {

        width: 100% !important;

        min-height: 230px !important;

        background: #f7f8f8 !important;

        color: #111111 !important;

        border: 3px solid #c5cbce !important;

        border-radius: 17px !important;

        font-family: inherit !important;

        font-size: 2rem !important;

        font-weight: 900 !important;

        line-height: 1.45 !important;

        padding: 1.5rem !important;

        resize: vertical !important;

        box-sizing: border-box !important;
    }

    .st-key-search_hover [data-testid="stTextArea"] textarea:focus {

        border: 3px solid white !important;

        outline: none !important;

        box-shadow:
            0 0 0 4px rgba(255,255,255,0.18),
            inset 0 4px 9px rgba(0,0,0,0.15) !important;
    }

    .st-key-search_hover [data-testid="stTextArea"] textarea::placeholder {

        color: #4f575b !important;

        font-size: 1.7rem !important;

        font-weight: 900 !important;

        opacity: 1 !important;
    }


    /* ========================================================
       VERLUSTDATUM
       ======================================================== */

    .st-key-search_hover .loss-date-label {

        color: white !important;

        font-size: 1.35rem !important;

        font-weight: 900 !important;

        margin:
            0.7rem
            2rem
            0.5rem
            2rem !important;
    }

    .st-key-search_hover [data-testid="stTextInput"] {

        max-height: 0 !important;

        opacity: 0 !important;

        overflow: hidden !important;

        pointer-events: none !important;

        transform: translateY(-20px) !important;

        margin: 0 !important;

        transition:
            max-height 0.35s ease,
            opacity 0.25s ease,
            transform 0.35s ease,
            margin 0.35s ease !important;
    }

    .st-key-search_hover:hover [data-testid="stTextInput"] {

        max-height: 120px !important;

        opacity: 1 !important;

        overflow: visible !important;

        pointer-events: auto !important;

        transform: translateY(0) !important;

        margin:
            0
            2rem
            1rem
            2rem !important;
    }

    .st-key-search_hover [data-testid="stTextInput"] label {
        display: none !important;
    }

    .st-key-search_hover [data-testid="stTextInput"] input {

        width: 100% !important;

        min-height: 70px !important;

        background: #f7f8f8 !important;

        color: #111111 !important;

        border: 3px solid #c5cbce !important;

        border-radius: 15px !important;

        font-size: 1.7rem !important;

        font-weight: 900 !important;

        padding-left: 1.2rem !important;

        box-sizing: border-box !important;
    }

    .st-key-search_hover [data-testid="stTextInput"] input::placeholder {

        color: #4f575b !important;

        font-size: 1.45rem !important;

        font-weight: 900 !important;

        opacity: 1 !important;
    }

    .st-key-search_hover [data-testid="stTextInput"] input:focus {

        border: 3px solid white !important;

        outline: none !important;

        box-shadow:
            0 0 0 4px rgba(255,255,255,0.18),
            inset 0 4px 9px rgba(0,0,0,0.15) !important;
    }


    /* ========================================================
       SUCH-BUTTON
       ======================================================== */

    .st-key-search_hover .stButton {

        max-height: 0 !important;

        opacity: 0 !important;

        overflow: hidden !important;

        pointer-events: none !important;

        transform: translateY(-20px) !important;

        margin: 0 !important;

        transition:
            max-height 0.35s ease,
            opacity 0.25s ease,
            transform 0.35s ease,
            margin 0.35s ease !important;
    }

    .st-key-search_hover:hover .stButton {

        max-height: 100px !important;

        opacity: 1 !important;

        overflow: visible !important;

        pointer-events: auto !important;

        transform: translateY(0) !important;

        margin: 0 2rem 1rem 2rem !important;
    }

    .st-key-search_hover .stButton > button {

        width: 100% !important;

        min-height: 65px !important;

        border-radius: 13px !important;

        border: 2px solid #aeb7bb !important;

        background:
            linear-gradient(
                145deg,
                #8e989d,
                #667177
            ) !important;

        color: white !important;

        font-size: 1.35rem !important;

        font-weight: 900 !important;
    }


    /* ========================================================
       UPLOAD FELDER
       ======================================================== */

    .st-key-media_hover,
    .st-key-camera_hover {

        width: 100% !important;

        min-height: 230px !important;

        padding: 0 !important;

        background:
            linear-gradient(
                145deg,
                #949da1 0%,
                #778187 50%,
                #687278 100%
            ) !important;

        border: 2px solid #b8c0c4 !important;

        border-radius: 22px !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.24),
            0 12px 28px rgba(0,0,0,0.30) !important;

        overflow: hidden !important;

        transition:
            min-height 0.30s ease,
            transform 0.30s ease,
            box-shadow 0.30s ease !important;
    }

    .st-key-media_hover .upload-title,
    .st-key-camera_hover .camera-title {

        display: flex !important;

        align-items: center !important;
        justify-content: center !important;

        min-height: 230px !important;

        padding: 2rem !important;

        color: white !important;

        font-size: 2.35rem !important;

        line-height: 1.2 !important;

        font-weight: 900 !important;

        text-align: center !important;
    }

    .st-key-media_hover:hover,
    .st-key-camera_hover:hover {

        min-height: 650px !important;

        transform: scale(1.02) !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.28),
            0 22px 45px rgba(0,0,0,0.40) !important;
    }

    .st-key-media_hover:hover .upload-title,
    .st-key-camera_hover:hover .camera-title {

        min-height: 100px !important;

        font-size: 2.1rem !important;
    }

    .st-key-media_hover [data-testid="stFileUploader"],
    .st-key-camera_hover [data-testid="stCameraInput"] {

        max-height: 0 !important;

        opacity: 0 !important;

        overflow: hidden !important;

        pointer-events: none !important;

        transform: translateY(-20px) !important;
    }

    .st-key-media_hover:hover [data-testid="stFileUploader"] {

        max-height: 400px !important;

        opacity: 1 !important;

        overflow: visible !important;

        pointer-events: auto !important;

        transform: translateY(0) !important;

        margin: 0 1.5rem 1.5rem 1.5rem !important;
    }

    .st-key-camera_hover:hover [data-testid="stCameraInput"] {

        max-height: 500px !important;

        opacity: 1 !important;

        overflow: visible !important;

        pointer-events: auto !important;

        transform: translateY(0) !important;

        margin: 0 1.5rem 1.5rem 1.5rem !important;
    }

    .st-key-media_hover [data-testid="stFileUploader"] {

        background:
            linear-gradient(
                145deg,
                #858f94,
                #697379
            ) !important;

        border: 2px solid #adb5b9 !important;

        border-radius: 16px !important;

        padding: 1.3rem !important;
    }

    .st-key-media_hover [data-testid="stFileUploader"] label {

        color: white !important;

        font-size: 1.35rem !important;

        font-weight: 900 !important;
    }

    .st-key-media_hover [data-testid="stFileUploader"] button {

        min-height: 60px !important;

        background: #f1f2f3 !important;

        color: #171717 !important;

        border-radius: 11px !important;

        font-size: 1.15rem !important;

        font-weight: 900 !important;
    }

    .st-key-camera_hover [data-testid="stCameraInput"] {

        width: calc(100% - 3rem) !important;
    }

    .st-key-camera_hover [data-testid="stCameraInput"] button {

        min-height: 65px !important;

        border-radius: 12px !important;

        font-size: 1.2rem !important;

        font-weight: 900 !important;
    }


    /* ========================================================
       NORMALE EINGABEFELDER
       ======================================================== */

    div[data-testid="stTextInput"] input {

        min-height: 62px !important;

        background: #f5f6f7 !important;

        color: #111111 !important;

        border: 2px solid #c4c9cc !important;

        border-radius: 13px !important;

        font-size: 1.25rem !important;

        font-weight: 800 !important;

        padding-left: 1rem !important;
    }

    div[data-testid="stDateInput"] input {

        min-height: 60px !important;

        background: #f5f6f7 !important;

        color: #111111 !important;

        border: 2px solid #c4c9cc !important;

        border-radius: 13px !important;

        font-size: 1.2rem !important;

        font-weight: 800 !important;
    }

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

        font-weight: 900 !important;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.22),
            0 7px 15px rgba(0,0,0,0.22) !important;
    }


    /* ========================================================
       NEUESTE / ÄLTESTE AUSWAHL
       ======================================================== */

    .st-key-latest_selector,
    .st-key-oldest_selector {

        width: 100% !important;

        min-height: 235px !important;

        padding: 0 !important;

        margin-bottom: 1rem !important;

        background:
            linear-gradient(
                145deg,
                #949da1 0%,
                #778187 50%,
                #687278 100%
            ) !important;

        border: 2px solid #b8c0c4 !important;

        border-radius: 22px !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.24),
            0 12px 28px rgba(0,0,0,0.30) !important;

        overflow: hidden !important;

        transition:
            min-height 0.45s ease,
            transform 0.35s ease,
            box-shadow 0.35s ease !important;
    }

    .st-key-latest_selector:hover,
    .st-key-oldest_selector:hover {

        min-height: 390px !important;

        transform: scale(1.015) !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.28),
            0 22px 45px rgba(0,0,0,0.40) !important;
    }

    .st-key-latest_selector .selector-title,
    .st-key-oldest_selector .selector-title {

        min-height: 120px !important;

        display: flex !important;

        align-items: center !important;
        justify-content: center !important;

        color: white !important;

        font-size: 2.7rem !important;

        line-height: 1.1 !important;

        font-weight: 900 !important;

        text-align: center !important;

        transition:
            font-size 0.35s ease,
            min-height 0.35s ease !important;
    }

    .st-key-latest_selector:hover .selector-title,
    .st-key-oldest_selector:hover .selector-title {

        min-height: 95px !important;

        font-size: 2.25rem !important;
    }


    /* ========================================================
       VORSCHAUBILDER
       ======================================================== */

    .preview-window {

        width: 100% !important;

        height: 210px !important;

        overflow: hidden !important;

        position: relative !important;

        display: flex !important;

        align-items: center !important;
    }

    .preview-track {

        display: flex !important;

        gap: 18px !important;

        width: max-content !important;

        animation:
            preview-move 24s linear infinite !important;

        padding-left: 20px !important;
    }

    .preview-window:hover .preview-track {
        animation-play-state: paused !important;
    }

    .preview-image {

        width: 145px !important;

        height: 145px !important;

        object-fit: cover !important;

        border-radius: 14px !important;

        border: 2px solid rgba(255,255,255,0.55) !important;

        box-shadow:
            0 8px 20px rgba(0,0,0,0.30) !important;

        flex-shrink: 0 !important;
    }

    @keyframes preview-move {

        0% {
            transform: translateX(0);
        }

        100% {
            transform: translateX(-50%);
        }
    }


    /* ========================================================
       AUSWAHL-BUTTON
       ======================================================== */

    .st-key-latest_selector .stButton,
    .st-key-oldest_selector .stButton {

        padding:
            0
            1.5rem
            1.2rem
            1.5rem !important;
    }

    .st-key-latest_selector .stButton > button,
    .st-key-oldest_selector .stButton > button {

        min-height: 58px !important;

        border-radius: 13px !important;

        background:
            linear-gradient(
                145deg,
                #8e989d,
                #667177
            ) !important;

        color: white !important;

        font-size: 1.15rem !important;

        font-weight: 900 !important;
    }


    /* ========================================================
       GALERIE
       ======================================================== */

    .st-key-gallery_card {

        min-height: 570px !important;

        padding: 1.3rem !important;

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

        overflow: hidden !important;

        transition:
            transform 0.35s ease,
            min-height 0.35s ease,
            box-shadow 0.35s ease !important;
    }

    .st-key-gallery_card:hover {

        transform: scale(1.045) !important;

        min-height: 700px !important;

        box-shadow:
            inset 0 2px 0 rgba(255,255,255,0.25),
            0 25px 50px rgba(0,0,0,0.45) !important;

        position: relative !important;

        z-index: 20 !important;
    }

    .st-key-gallery_card img {

        border-radius: 15px !important;

        transition:
            transform 0.35s ease !important;
    }

    .st-key-gallery_card:hover img {

        transform: scale(1.02) !important;
    }


    /* ========================================================
       INFO UNTER DEM BILD
       ======================================================== */

    .gallery-info {

        margin-top: 1rem !important;

        padding-top: 0.8rem !important;

        border-top:
            2px solid
            rgba(255,255,255,0.22) !important;

        color: white !important;

        opacity: 0 !important;

        max-height: 0 !important;

        overflow: hidden !important;

        transform: translateY(20px) !important;

        transition:
            opacity 0.35s ease,
            max-height 0.40s ease,
            transform 0.35s ease !important;
    }

    .st-key-gallery_card:hover .gallery-info {

        opacity: 1 !important;

        max-height: 260px !important;

        transform: translateY(0) !important;
    }

    .gallery-info-title {

        color: white !important;

        font-size: 1.45rem !important;

        font-weight: 900 !important;

        margin-bottom: 0.7rem !important;
    }

    .gallery-info-line {

        color: white !important;

        font-size: 1.05rem !important;

        font-weight: 700 !important;

        margin:
            0.35rem 0 !important;
    }


    /* ========================================================
       BUTTON FUNDSTÜCK GEFUNDEN
       ======================================================== */

    .found-button-area {

        margin-top: 1rem !important;

        opacity: 0 !important;

        max-height: 0 !important;

        overflow: hidden !important;

        transform: translateY(15px) !important;

        transition:
            opacity 0.35s ease,
            max-height 0.35s ease,
            transform 0.35s ease !important;
    }

    .st-key-gallery_card:hover .found-button-area {

        opacity: 1 !important;

        max-height: 100px !important;

        transform: translateY(0) !important;
    }


    /* ========================================================
       BILD-GALERIE BUTTON
       ======================================================== */

    .st-key-gallery_card .stButton > button {

        min-height: 55px !important;

        border-radius: 12px !important;

        border: 2px solid #b9c1c5 !important;

        background:
            linear-gradient(
                145deg,
                #7b878d,
                #59656b
            ) !important;

        color: white !important;

        font-size: 1.05rem !important;

        font-weight: 900 !important;
    }

    .st-key-gallery_card .stButton > button:hover {

        filter: brightness(1.12) !important;

        transform: translateY(-2px) !important;
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
# LABELS
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

            "filename":
                "TEXT DEFAULT ''",

            "label":
                "TEXT DEFAULT ''",

            "location":
                "TEXT DEFAULT ''",

            "found_date":
                "TEXT DEFAULT ''",

            "created_at":
                "TEXT DEFAULT ''",

            "found":
                "INTEGER DEFAULT 0"
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

    filename = (
        uuid.uuid4().hex
        + ".jpg"
    )

    image_path = (
        UPLOAD_DIR
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


# ============================================================
# FUNDSTÜCK ALS GEFUNDEN MARKIEREN
# ============================================================

def mark_item_as_found(item_id):

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
            / str(filename)
        )

        try:

            if image_path.exists():

                image_path.unlink()

        except Exception:

            pass


# ============================================================
# NEUESTE FUNDSTÜCKE
# ============================================================

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

            WHERE
                found = 0

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


# ============================================================
# ÄLTESTE FUNDSTÜCKE
# ============================================================

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

            WHERE
                found = 0

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
# DATUM EINGABE UMGEWANDELN
# ============================================================

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


# ============================================================
# SUCHE
# ============================================================

def search_items(
    query,
    loss_date=None
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

                found = 0

                AND

                (
                    LOWER(label) LIKE ?

                    OR

                    LOWER(location) LIKE ?
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


    # ========================================================
    # WENN KEIN DATUM ANGEGEBEN WURDE
    # ========================================================

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


    # ========================================================
    # DATUMSABSTAND BERECHNEN
    # ========================================================

    dated_results = []
    undated_results = []


    for item in results:

        found_date_text = item[4]

        try:

            found_date = datetime.strptime(
                found_date_text,
                "%Y-%m-%d"
            ).date()

        except Exception:

            undated_results.append(
                item
            )

            continue


        difference = abs(
            (found_date - loss_date).days
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


    # Nur die 5 zeitlich nächsten Fundstücke
    return [
        item
        for difference, item
        in dated_results[:5]
    ]


# ============================================================
# KI
# ============================================================

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

        st.rerun()


    if st.button(
        "Bild hochladen",
        key="nav_upload",
        width="stretch"
    ):

        st.session_state.page = "Bild hochladen"

        st.rerun()


    if st.button(
        "Älteste Fundstücke",
        key="nav_oldest",
        width="stretch"
    ):

        st.session_state.page = (
            "Älteste Fundstücke"
        )

        st.rerun()


# ============================================================
# TITEL
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
# SUCHERGEBNISSE
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
                        / str(filename)
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


# ============================================================
# SEITE SUCHE
# ============================================================

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


        # ----------------------------------------------------
        # SUCHTEXT
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # VERLUSTDATUM
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # SUCHEN
        # ----------------------------------------------------

        search_clicked = st.button(

            "Suchen",

            key="search_button",

            width="stretch"
        )


    if search_clicked:

        query = search_text.strip()

        entered_date = loss_date_text.strip()


        # ----------------------------------------------------
        # DATUM PRÜFEN
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # WEDER TEXT NOCH DATUM
        # ----------------------------------------------------

        if not query and loss_date is None:

            st.warning(
                "Bitte gib entweder ein verlorenes "
                "Objekt oder ein Verlustdatum ein."
            )

            return


        # ----------------------------------------------------
        # WENN NUR DATUM EINGEGEBEN WURDE
        # ----------------------------------------------------

        if not query:

            query = ""


        # ----------------------------------------------------
        # SUCHEN
        # ----------------------------------------------------

        results = search_items(
            query,
            loss_date
        )


        # ----------------------------------------------------
        # HINWEIS BEI DATUMSSUCHE
        # ----------------------------------------------------

        if loss_date is not None:

            st.caption(
                "Es werden die 5 Fundstücke angezeigt, "
                "deren Funddatum deinem eingegebenen "
                "Verlustdatum am nächsten liegt."
            )


        display_search_results(
            results
        )


# ============================================================
# UPLOAD
# ============================================================

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

                label, confidence = (
                    predict_image(image)
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


# ============================================================
# KAMERA
# ============================================================

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

                label, confidence = (
                    predict_image(image)
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


# ============================================================
# SEITE BILD HOCHLADEN
# ============================================================

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
                    Lade ein Bild aus deiner Mediathek hoch
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
                    Mach ein Foto in der App
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


# ============================================================
# VORSCHAUBILDER
# ============================================================

def create_preview_html(
    items
):

    valid_items = []


    for item in items:

        filename = item[1]

        image_path = (
            UPLOAD_DIR
            / str(filename)
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


    all_images = (
        valid_items
        +
        valid_items
    )


    image_tags = ""


    for image_path in all_images:

        try:

            image = Image.open(
                str(image_path)
            ).convert("RGB")


            image = ImageOps.fit(
                image,
                (145, 145)
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
                '<img '
                'class="preview-image" '
                'src="data:image/jpeg;base64,'
                + encoded
                + '" />'
            )


        except Exception:

            continue


    return (
        '<div class="preview-window">'
        '<div class="preview-track">'
        + image_tags
        + '</div>'
        '</div>'
    )


# ============================================================
# LETZTE SEITE
# ============================================================

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


    # ========================================================
    # NEUESTE
    # ========================================================

    with col1:

        with st.container(
            border=True,
            key="latest_selector"
        ):

            st.markdown(
                """
                <div class="selector-title">
                    Neueste Bilder
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


    # ========================================================
    # ÄLTESTE
    # ========================================================

    with col2:

        with st.container(
            border=True,
            key="oldest_selector"
        ):

            st.markdown(
                """
                <div class="selector-title">
                    Älteste Bilder
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


# ============================================================
# GALERIE
# ============================================================

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
            "Älteste Fundstücke"
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


    # ========================================================
    # DREI BILDER PRO REIHE
    # ========================================================

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
                        / str(filename)
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


                    # ----------------------------------------
                    # INFORMATIONEN
                    # ----------------------------------------

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
                        % (

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


                    # ----------------------------------------
                    # FUNDSTÜCK GEFUNDEN
                    # ----------------------------------------

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


# ============================================================
# SEITENSTEUERUNG
# ============================================================

current_page = st.session_state.get(
    "page",
    "Suche"
)


if current_page == "Suche":

    show_search_page()


elif current_page == "Bild hochladen":

    show_upload_page()


elif current_page == "Älteste Fundstücke":

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
