import pandas as pd
import os


file_dir = "tmp"
files = [file_dir+"/"+file for file in os.listdir(file_dir)]

log = open("bad-files.txt", "w+")

for file in files:
    try:
        print(f"Checking {file}...")
        pd.read_csv(file, sep="|", encoding="iso-8859-1")
    except pd.errors.ParserError as e:
        print(f"Error in {file}")
        log.write(f"{file} {e} \n")

log.close()
