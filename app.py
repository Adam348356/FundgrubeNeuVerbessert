    /* ========================================================
       KRÄFTIGER ANIMIERTER WALLPAPER
       ======================================================== */

    .stApp {

        background:

            /* großes violettes Licht links oben */
            radial-gradient(
                ellipse 55% 55% at 12% 18%,
                rgba(175, 55, 255, 0.78) 0%,
                rgba(137, 48, 235, 0.55) 22%,
                rgba(83, 35, 170, 0.25) 45%,
                transparent 72%
            ),

            /* kräftiges blaues Licht rechts oben */
            radial-gradient(
                ellipse 60% 55% at 88% 12%,
                rgba(50, 105, 255, 0.72) 0%,
                rgba(42, 73, 220, 0.48) 25%,
                rgba(32, 52, 150, 0.20) 48%,
                transparent 75%
            ),

            /* violettes Licht unten rechts */
            radial-gradient(
                ellipse 60% 60% at 82% 88%,
                rgba(150, 45, 255, 0.68) 0%,
                rgba(110, 38, 210, 0.42) 25%,
                rgba(65, 30, 145, 0.20) 48%,
                transparent 76%
            ),

            /* blau-violettes Licht unten links */
            radial-gradient(
                ellipse 65% 55% at 8% 88%,
                rgba(42, 91, 255, 0.62) 0%,
                rgba(43, 58, 190, 0.38) 25%,
                rgba(30, 42, 125, 0.18) 50%,
                transparent 76%
            ),

            /* zentrale violette Lichtwolke */
            radial-gradient(
                ellipse 45% 45% at 50% 50%,
                rgba(112, 55, 255, 0.32) 0%,
                rgba(78, 42, 180, 0.16) 38%,
                transparent 72%
            ),

            /* tiefer Hintergrund */
            linear-gradient(
                135deg,
                #030513 0%,
                #080a22 25%,
                #090c2d 50%,
                #070921 75%,
                #02040f 100%
            ) !important;


        background-size:
            190% 190%,
            185% 185%,
            190% 190%,
            185% 185%,
            170% 170%,
            100% 100% !important;


        background-position:
            0% 0%,
            100% 0%,
            100% 100%,
            0% 100%,
            50% 50%,
            50% 50% !important;


        animation:
            wallpaperMove 20s ease-in-out infinite alternate !important;


        background-attachment:
            fixed !important;


        color: white;

        overflow-x: hidden !important;
    }


    /* ========================================================
       ZUSÄTZLICHE BEWEGTE LICHTWOLKEN
       ======================================================== */

    .stApp::before {

        content: "";

        position: fixed;

        width: 75vw;
        height: 75vh;

        left: -18vw;
        top: -18vh;

        pointer-events: none;

        z-index: 0;


        background:

            radial-gradient(
                ellipse at center,
                rgba(190, 55, 255, 0.55) 0%,
                rgba(145, 45, 240, 0.30) 28%,
                rgba(90, 35, 180, 0.12) 48%,
                transparent 72%
            );


        filter:
            blur(45px);


        opacity:
            0.95;


        animation:
            purpleCloudMove 16s ease-in-out infinite alternate;
    }


    .stApp::after {

        content: "";

        position: fixed;

        width: 80vw;
        height: 80vh;

        right: -20vw;
        bottom: -20vh;

        pointer-events: none;

        z-index: 0;


        background:

            radial-gradient(
                ellipse at center,
                rgba(55, 105, 255, 0.50) 0%,
                rgba(80, 65, 245, 0.32) 30%,
                rgba(125, 45, 235, 0.16) 50%,
                transparent 72%
            );


        filter:
            blur(55px);


        opacity:
            0.90;


        animation:
            blueCloudMove 19s ease-in-out infinite alternate;
    }


    /* ========================================================
       HAUPTBEWEGUNG DES WALLPAPERS
       ======================================================== */

    @keyframes wallpaperMove {

        0% {

            background-position:
                0% 0%,
                100% 0%,
                100% 100%,
                0% 100%,
                50% 50%,
                50% 50%;
        }


        20% {

            background-position:
                18% 12%,
                82% 8%,
                88% 82%,
                10% 88%,
                43% 56%,
                50% 50%;
        }


        40% {

            background-position:
                32% 25%,
                68% 20%,
                74% 68%,
                25% 76%,
                58% 44%,
                50% 50%;
        }


        60% {

            background-position:
                20% 38%,
                88% 32%,
                65% 86%,
                38% 62%,
                47% 60%,
                50% 50%;
        }


        80% {

            background-position:
                5% 25%,
                72% 42%,
                90% 72%,
                18% 92%,
                62% 48%,
                50% 50%;
        }


        100% {

            background-position:
                0% 40%,
                100% 30%,
                100% 100%,
                0% 70%,
                42% 55%,
                50% 50%;
        }
    }


    /* ========================================================
       VIOLETTE LICHTWOLKE
       ======================================================== */

    @keyframes purpleCloudMove {

        0% {

            transform:
                translate3d(
                    -5vw,
                    -3vh,
                    0
                )
                scale(0.95);
        }


        25% {

            transform:
                translate3d(
                    4vw,
                    2vh,
                    0
                )
                scale(1.05);
        }


        50% {

            transform:
                translate3d(
                    10vw,
                    8vh,
                    0
                )
                scale(1.10);
        }


        75% {

            transform:
                translate3d(
                    2vw,
                    13vh,
                    0
                )
                scale(1.04);
        }


        100% {

            transform:
                translate3d(
                    -7vw,
                    7vh,
                    0
                )
                scale(0.98);
        }
    }


    /* ========================================================
       BLAUE LICHTWOLKE
       ======================================================== */

    @keyframes blueCloudMove {

        0% {

            transform:
                translate3d(
                    6vw,
                    5vh,
                    0
                )
                scale(0.95);
        }


        25% {

            transform:
                translate3d(
                    -3vw,
                    -3vh,
                    0
                )
                scale(1.06);
        }


        50% {

            transform:
                translate3d(
                    -10vw,
                    -8vh,
                    0
                )
                scale(1.10);
        }


        75% {

            transform:
                translate3d(
                    -4vw,
                    4vh,
                    0
                )
                scale(1.04);
        }


        100% {

            transform:
                translate3d(
                    7vw,
                    8vh,
                    0
                )
                scale(0.98);
        }
    }
