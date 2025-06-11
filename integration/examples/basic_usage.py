"""
Basic usage example for the Khmer Name OCR Corrector API.
"""
import os
import sys

# Add parent directory to path to allow importing from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from integration.python_client import KhmerNameCorrectorClient

# Set up API client
api_url = os.environ.get("API_URL", "http://localhost:8000")
client = KhmerNameCorrectorClient(api_url)

def main():
    # Print API info
    try:
        info = client.get_info()
        print("API Info:")
        print(f"  Model: {info['name']}")
        print(f"  Device: {info['device']}")
        print(f"  Names in database: {info['num_names']}")
        print(f"  Latin names available: {info['has_latin_names']}")
        print()
    except Exception as e:
        print(f"Error getting API info: {str(e)}")
        print("Make sure the API is running at", api_url)
        return

    # Single match example
    print("Single match example:")
    khmer_name = "សុងហៀង"
    latin_name = "songhieng"
    
    print(f"Matching Khmer: '{khmer_name}', Latin: '{latin_name}'")
    
    try:
        result = client.match(khmer_name, latin_name)
        
        print(f"Input: {result['input_khmer']} / {result.get('input_latin', '')}")
        print("Results:")
        
        for i, match in enumerate(result['results'], 1):
            print(f"  {i}. {match['khmer_name']} ({match['latin_name']})")
            print(f"     Khmer score: {match['khmer_score']:.3f}")
            print(f"     Latin score: {match['latin_score']:.3f}")
            print(f"     Combined: {match['combined_score']:.3f}")
    except Exception as e:
        print(f"Error in match: {str(e)}")

    # Batch match example
    print("\nBatch match example:")
    batch_items = [
        {"khmer_text": "សុងហៀង", "latin_text": "songhieng"},
        {"khmer_text": "ដារា", "latin_text": "dara"},
        {"khmer_text": "សុខ", "latin_text": "sok"}
    ]
    
    try:
        batch_result = client.batch_match(batch_items)
        
        print(f"Processed {len(batch_result['results'])} items:")
        
        for i, item_result in enumerate(batch_result['results'], 1):
            top_match = item_result['results'][0] if item_result['results'] else None
            
            if top_match:
                print(f"  {i}. Input: {item_result['input_khmer']} / {item_result.get('input_latin', '')}")
                print(f"     Best match: {top_match['khmer_name']} ({top_match['latin_name']})")
                print(f"     Combined score: {top_match['combined_score']:.3f}")
            else:
                print(f"  {i}. No matches found for {item_result['input_khmer']}")
    except Exception as e:
        print(f"Error in batch match: {str(e)}")

    # Find best match example
    print("\nFind best match example:")
    
    try:
        best_match = client.find_best_match("សុងហៀង", "songhieng", min_score=0.8)
        
        if best_match:
            print(f"Found match: {best_match['khmer_name']} ({best_match['latin_name']})")
            print(f"Score: {best_match['combined_score']:.3f}")
        else:
            print("No match found above threshold")
    except Exception as e:
        print(f"Error finding best match: {str(e)}")

if __name__ == "__main__":
    main() 