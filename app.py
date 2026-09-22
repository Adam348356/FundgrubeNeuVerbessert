def create_header(title, background_color):
    """
    Erstellt den Kopfbereich passend zu den beiden Skizzen.
    Alles wird als echtes PIL-Bild erzeugt – kein HTML/CSS/SVG.
    """

    width = 1600
    height = 470

    image = Image.new(
        "RGB",
        (width, height),
        background_color,
    )

    draw = ImageDraw.Draw(image)

    # --------------------------------------------------------
    # TITEL
    # --------------------------------------------------------

    # Etwas kleinere Schrift, damit beide Titel sicher passen
    title_font = get_font(76, bold=True)

    bbox = draw.textbbox(
        (0, 0),
        title,
        font=title_font,
    )

    title_width = bbox[2] - bbox[0]
    title_height = bbox[3] - bbox[1]

    title_x = (width - title_width) // 2
    title_y = 25

    draw.text(
        (title_x, title_y),
        title,
        fill=(0, 0, 0),
        font=title_font,
    )

    # --------------------------------------------------------
    # SCHULE
    # --------------------------------------------------------
    # Wichtig:
    # Die Schule wird deutlich kleiner und weiter nach unten
    # gesetzt, damit sie den Titel NICHT mehr überdeckt.

    draw_school_silhouette(
        draw,
        width,
        base_y=455,
        scale=0.72,
    )

    # --------------------------------------------------------
    # BODENLINIE
    # --------------------------------------------------------

    draw.line(
        [(0, 455), (width, 455)],
        fill=(10, 10, 10),
        width=4,
    )

    return image
