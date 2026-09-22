# ============================================================
# FARBEN
# ============================================================

HELLBLAU = "#BFE8F7"
METALL_GRAU = "#70787E"
METALL_GRAU_DUNKEL = "#555D63"
WEISS = "#FFFFFF"
SCHWARZ = "#20262A"


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    f"""
    <style>

    /* ======================================================
       GESAMTE APP
       ====================================================== */

    .stApp {{
        background-color: {HELLBLAU};
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {HELLBLAU};
    }}

    [data-testid="stHeader"] {{
        background-color: {HELLBLAU};
    }}

    .main .block-container {{
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}


    /* ======================================================
       TITEL
       ====================================================== */

    .fundgrube-title {{
        color: {SCHWARZ};
        text-align: center;

        font-size: 5.5rem;
        font-weight: 900;

        line-height: 1.05;

        margin-top: 0.5rem;
        margin-bottom: 0.2rem;

        letter-spacing: -2px;
    }}

    .fundgrube-subtitle {{
        color: {SCHWARZ};

        text-align: center;

        font-size: 1.15rem;
        font-weight: 500;

        margin-bottom: 0.5rem;
    }}


    /* ======================================================
       LUPE
       ====================================================== */

    .loupe-container {{
        display: flex;
        justify-content: center;
        align-items: center;

        width: 100%;

        margin-top: 5px;
        margin-bottom: 25px;

        transition:
            transform 0.25s ease;
    }}

    .loupe-container:hover {{
        transform: scale(1.10);
    }}


    /* ======================================================
       BESCHRIFTUNGEN
       ====================================================== */

    label {{
        color: {SCHWARZ} !important;
        font-weight: 700 !important;
    }}

    .stMarkdown p {{
        color: {SCHWARZ};
    }}


    /* ======================================================
       SUCHFELD
       ====================================================== */

    /* Äußerer Bereich des Suchfeldes */
    div[data-testid="stTextInput"] {{
        transition:
            transform 0.20s ease,
            filter 0.20s ease;
    }}

    /* Wenn Maus über Suchfeld */
    div[data-testid="stTextInput"]:hover {{
        transform: scale(1.015);
    }}

    /* Wenn Suchfeld aktiv ist */
    div[data-testid="stTextInput"]:focus-within {{
        transform: scale(1.025);
    }}

    /* Eigentliches Eingabefeld */
    div[data-testid="stTextInput"] div[data-baseweb="input"] {{
        background-color: {METALL_GRAU} !important;

        border: 2px solid {METALL_GRAU_DUNKEL} !important;

        border-radius: 13px !important;

        min-height: 52px;

        transition:
            background-color 0.2s ease,
            border-color 0.2s ease,
            box-shadow 0.2s ease;
    }}

    /* Aktives Suchfeld */
    div[data-testid="stTextInput"]:focus-within
    div[data-baseweb="input"] {{
        background-color: {METALL_GRAU_DUNKEL} !important;

        border-color: {SCHWARZ} !important;

        box-shadow:
            0 0 0 3px rgba(255,255,255,0.30) !important;
    }}

    /* Text im Suchfeld */
    div[data-testid="stTextInput"]
    input {{
        color: {WEISS} !important;

        background-color: transparent !important;

        font-size: 1.05rem !important;

        font-weight: 500 !important;
    }}

    /* Platzhalter */
    div[data-testid="stTextInput"]
    input::placeholder {{
        color: {WEISS} !important;

        opacity: 0.85 !important;
    }}


    /* ======================================================
       ALLE ANDEREN TEXTFELDER
       ====================================================== */

    div[data-testid="stTextInput"] input {{
        caret-color: white !important;
    }}


    /* ======================================================
       DATE INPUT
       ====================================================== */

    div[data-testid="stDateInput"] {{
        transition:
            transform 0.20s ease;
    }}

    div[data-testid="stDateInput"]:hover {{
        transform: scale(1.015);
    }}

    div[data-testid="stDateInput"]:focus-within {{
        transform: scale(1.02);
    }}

    div[data-testid="stDateInput"]
    div[data-baseweb="input"] {{
        background-color: {METALL_GRAU} !important;

        border: 2px solid {METALL_GRAU_DUNKEL} !important;

        border-radius: 13px !important;
    }}

    div[data-testid="stDateInput"] input {{
        color: {WEISS} !important;

        background-color: transparent !important;
    }}


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton {{
        transition:
            transform 0.20s ease;
    }}

    .stButton:hover {{
        transform: scale(1.02);
    }}

    .stButton > button {{
        background-color: {METALL_GRAU} !important;

        color: {WEISS} !important;

        border: 2px solid {METALL_GRAU_DUNKEL} !important;

        border-radius: 13px !important;

        min-height: 50px;

        font-weight: 700 !important;

        transition:
            background-color 0.2s ease,
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }}

    .stButton > button:hover {{
        background-color: {METALL_GRAU_DUNKEL} !important;

        color: {WEISS} !important;

        border-color: {SCHWARZ} !important;

        box-shadow:
            0 5px 12px rgba(0,0,0,0.20);
    }}


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {{
        background-color: {METALL_GRAU} !important;

        border-radius: 15px !important;

        padding: 1rem !important;

        border: 2px solid {METALL_GRAU_DUNKEL} !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }}

    [data-testid="stFileUploader"]:hover {{
        transform: scale(1.015);

        box-shadow:
            0 5px 12px rgba(0,0,0,0.15);
    }}

    [data-testid="stFileUploader"] * {{
        color: {WEISS} !important;
    }}


    /* ======================================================
       KARTEN
       ====================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {METALL_GRAU} !important;

        border-radius: 16px !important;

        border: 2px solid {METALL_GRAU_DUNKEL} !important;

        padding: 0.5rem !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        transform: scale(1.015);

        box-shadow:
            0 7px 15px rgba(0,0,0,0.20);
    }}

    [data-testid="stVerticalBlockBorderWrapper"] * {{
        color: {WEISS};
    }}


    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {{
        background-color: {METALL_GRAU_DUNKEL};
    }}

    [data-testid="stSidebar"] * {{
        color: {WEISS} !important;
    }}


    /* ======================================================
       TRENNLINIEN
       ====================================================== */

    hr {{
        border-color: {METALL_GRAU} !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)
