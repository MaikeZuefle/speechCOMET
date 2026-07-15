import csv
import os
import sys
from pathlib import Path


def extract(input_tsv: str, output_csv: str) -> None:
    with open(input_tsv, newline="", encoding="utf-8") as fin, \
         open(output_csv, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=["src", "mt", "score", "src_audio"])
        writer.writeheader()

        for row in reader:
            wav_path = f"../wav/{row['ID']}.wav"
            writer.writerow({"src": "", "mt": row["REF"],       "score": 100, "src_audio": wav_path})
            writer.writerow({"src": "", "mt": row["WRONG-REF"], "score": 0,   "src_audio": wav_path})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <input_tsv>")
        sys.exit(1)

    input_tsv = Path(sys.argv[1]) #./MONOLINGUAL.it_v1.2.tsv.1F
    base = input_tsv.name
    output_csv = os.path.join(os.path.dirname(input_tsv), base + ".csv")

    extract(input_tsv, output_csv)
    print(f"Written: {output_csv}")
