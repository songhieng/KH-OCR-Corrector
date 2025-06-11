# split_and_deduplicate_khmer_names.py
from typing import Dict, Set, Tuple

def split_and_deduplicate(input_file: str, output_file: str):
    name_parts = set()

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            parts = line.strip().split()
            for part in parts:
                if part:
                    name_parts.add(part)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for name in sorted(name_parts):  # sorted optional
            outfile.write(name + '\n')

    print(f"✅ Done! Unique name parts saved to: {output_file}")

def split_and_deduplicate_with_translations(input_file: str, output_file: str):
    """
    Process a file containing both Khmer names and their Latin/English transliterations.
    Expected format: "khmer_name,latin_name" on each line
    """
    khmer_to_latin: Dict[str, str] = {}
    
    # Read the input file with name pairs
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if ',' in line:
                khmer, latin = line.split(',', 1)
                khmer = khmer.strip()
                latin = latin.strip()
                if khmer and latin:
                    # Split the Khmer name into parts
                    khmer_parts = khmer.split()
                    # Split the Latin name into parts
                    latin_parts = latin.split()
                    
                    # Ensure both have the same number of parts
                    if len(khmer_parts) == len(latin_parts):
                        for k_part, l_part in zip(khmer_parts, latin_parts):
                            khmer_to_latin[k_part] = l_part
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for khmer, latin in sorted(khmer_to_latin.items()):
            outfile.write(f"{khmer},{latin}\n")
    
    print(f"✅ Done! Unique name parts with translations saved to: {output_file}")

# Usage
# split_and_deduplicate("khmer_names.txt", "khmer_names_split_all.txt")
# split_and_deduplicate_with_translations("khmer_names_with_latin.txt", "khmer_latin_pairs.txt")