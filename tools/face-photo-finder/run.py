import argparse

from face_finder.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--background-scan", action="store_true")
    arguments = parser.parse_args()
    main(background_mode=arguments.background_scan)
