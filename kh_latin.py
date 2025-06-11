import csv
from khmer_latin_name_transformer import to_latin

input_file = "/kaggle/input/testing-dataset/khmer_names_split_all.txt"
output_file = "khmer_names_latin.csv"

# Read Khmer names from input
with open(input_file, "r", encoding="utf-8") as f:
    khmer_names = [line.strip() for line in f if line.strip()]

# Convert each to Latin
rows = [(khmer, to_latin(khmer)) for khmer in khmer_names]

# Save to CSV
with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["khmer_name", "latin_name"])  # Header
    writer.writerows(rows)

print(f"✅ Done! Saved to {output_file}")