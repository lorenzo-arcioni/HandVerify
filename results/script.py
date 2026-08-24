#!/usr/bin/env python3

import os
import pandas as pd

# Directory da cui partire
ROOT_DIR = "."

# File di output
OUTPUT_FILE = "final.txt"

# Pattern dei file da cercare
TARGET = "final_metrics.csv"


def find_metric_files(root):
    """
    Cerca ricorsivamente tutti i file che terminano con final_metrics.csv
    """
    files = []

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(TARGET):
                files.append(os.path.join(dirpath, filename))

    return sorted(files)


def process_file(filepath, output):
    """
    Legge il CSV, arrotonda i valori e scrive nel file cumulativo
    """

    filename = os.path.basename(filepath)

    output.write(f"=== {filename} ===\n\n")

    try:
        df = pd.read_csv(filepath)

        # Arrotonda tutte le colonne numeriche a 3 decimali
        df = df.round(3)

        # Scrive il dataframe in formato CSV
        df.to_csv(
            output,
            index=False
        )

    except Exception as e:
        output.write(f"ERRORE LETTURA FILE: {e}\n")

    output.write("\n\n")


def main():

    files = find_metric_files(ROOT_DIR)

    print(f"Trovati {len(files)} file:")

    for f in files:
        print(" -", f)

    # Sovrascrive final.txt
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for filepath in files:
            process_file(filepath, out)

    print(f"\nCreato {OUTPUT_FILE}")


if __name__ == "__main__":
    main()