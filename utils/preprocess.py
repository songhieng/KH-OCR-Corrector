"""
Preprocessing utilities for Khmer name matching.
"""
import os
from typing import Dict, List, Tuple, Set, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("preprocess")

def split_and_deduplicate(input_file: str, output_file: str):
    """
    Split names into parts and deduplicate them.
    
    Args:
        input_file: Path to input file with one name per line
        output_file: Path to output file where unique name parts will be written
    """
    name_parts = set()

    logger.info(f"Processing file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            parts = line.strip().split()
            for part in parts:
                if part:
                    name_parts.add(part)

    logger.info(f"Writing {len(name_parts)} unique name parts to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for name in sorted(name_parts):  # sorted optional
            outfile.write(name + '\n')

    logger.info(f"✅ Done! Unique name parts saved to: {output_file}")

def split_and_deduplicate_with_translations(input_file: str, output_file: str):
    """
    Process a file containing both Khmer names and their Latin/English transliterations.
    
    Expected format: "latin_name,khmer_name" on each line
    
    Args:
        input_file: Path to input file with name pairs
        output_file: Path to output file where name part pairs will be written
    """
    khmer_to_latin: Dict[str, str] = {}
    
    logger.info(f"Processing file with translations: {input_file}")
    
    # Read the input file with name pairs
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if ',' in line:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    latin, khmer = parts  # First is Latin, second is Khmer
                    latin = latin.strip()
                    khmer = khmer.strip().rstrip(',')
                    
                    if khmer and latin:
                        # Split the Khmer name into parts
                        khmer_parts = khmer.split()
                        # Split the Latin name into parts
                        latin_parts = latin.split()
                        
                        # Ensure both have the same number of parts
                        if len(khmer_parts) == len(latin_parts):
                            for k_part, l_part in zip(khmer_parts, latin_parts):
                                khmer_to_latin[k_part] = l_part
    
    logger.info(f"Writing {len(khmer_to_latin)} name part pairs to {output_file}")
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for khmer, latin in sorted(khmer_to_latin.items()):
            outfile.write(f"{latin},{khmer}\n")
    
    logger.info(f"✅ Done! Unique name parts with translations saved to: {output_file}")

def validate_and_clean_data(input_file: str, output_file: Optional[str] = None) -> Tuple[int, int]:
    """
    Validate and clean a data file by removing invalid entries.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file (if None, validation only, no writing)
        
    Returns:
        Tuple of (total entries, valid entries)
    """
    valid_lines = []
    total = 0
    valid = 0
    
    logger.info(f"Validating file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(f, 1):
            total += 1
            line = line.strip()
            
            if not line:
                logger.warning(f"Line {line_num}: Empty line")
                continue
                
            # For Latin,Khmer pairs
            if ',' in line:
                parts = line.split(',', 1)
                if len(parts) < 2:
                    logger.warning(f"Line {line_num}: Invalid format - {line}")
                    continue
                    
                latin, khmer = parts
                latin = latin.strip()
                khmer = khmer.strip().rstrip(',')
                
                if not latin or not khmer:
                    logger.warning(f"Line {line_num}: Missing Latin or Khmer - {line}")
                    continue
                    
                valid_line = f"{latin},{khmer}"
            else:
                # For single names
                valid_line = line
                
            valid_lines.append(valid_line)
            valid += 1
    
    if output_file:
        logger.info(f"Writing {valid} valid entries to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in valid_lines:
                outfile.write(line + '\n')
    
    logger.info(f"Validation complete: {valid}/{total} entries are valid")
    return total, valid

def merge_data_files(input_files: List[str], output_file: str, remove_duplicates: bool = True):
    """
    Merge multiple data files into one.
    
    Args:
        input_files: List of input file paths
        output_file: Path to output file
        remove_duplicates: Whether to remove duplicate entries
    """
    entries = set() if remove_duplicates else []
    
    logger.info(f"Merging {len(input_files)} files into {output_file}")
    
    for input_file in input_files:
        logger.info(f"Processing: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if line:
                    if remove_duplicates:
                        entries.add(line)
                    else:
                        entries.append(line)
    
    logger.info(f"Writing {len(entries)} entries to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in sorted(entries) if remove_duplicates else entries:
            outfile.write(entry + '\n')
    
    logger.info(f"✅ Done! Merged data saved to: {output_file}")

def prepare_dataset(
    input_file: str,
    output_dir: str,
    split: bool = False,
    validate: bool = True
):
    """
    Prepare a dataset for use with the KhmerNameMatcher.
    
    Args:
        input_file: Path to input file
        output_dir: Directory to write output files
        split: Whether to split names into parts
        validate: Whether to validate and clean the data
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_file).split('.')[0]
    
    # Validate and clean if requested
    if validate:
        validated_file = os.path.join(output_dir, f"{base_name}_validated.txt")
        total, valid = validate_and_clean_data(input_file, validated_file)
        logger.info(f"Validation: {valid}/{total} entries are valid")
        working_file = validated_file
    else:
        working_file = input_file
    
    # Split if requested
    if split:
        if ',' in open(working_file, 'r', encoding='utf-8').readline():
            # File has Latin,Khmer pairs
            output_file = os.path.join(output_dir, f"{base_name}_split.txt")
            split_and_deduplicate_with_translations(working_file, output_file)
        else:
            # File has only names
            output_file = os.path.join(output_dir, f"{base_name}_split.txt")
            split_and_deduplicate(working_file, output_file)
    
    logger.info(f"✅ Dataset preparation complete")
    return os.path.join(output_dir, f"{base_name}_split.txt") if split else working_file 