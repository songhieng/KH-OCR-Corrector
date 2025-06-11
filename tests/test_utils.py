"""
Test utilities for Khmer Name OCR Corrector.
"""
import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the utils modules
from utils.matcher import KhmerNameMatcher
from utils.preprocess import clean_khmer_text, normalize_latin_text

class TestKhmerUtils(unittest.TestCase):
    """Test the Khmer utilities."""
    
    def setUp(self):
        """Set up test fixtures before running the tests."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create a temporary names file
        self.names_file = os.path.join(self.test_dir, "test_names.csv")
        with open(self.names_file, "w", encoding="utf-8") as f:
            f.write("songhieng,សុងហៀង\n")
            f.write("dara,ដារា\n")
            f.write("sok,សុខ\n")
            f.write("vichet,វិចិត្រ\n")
    
    def tearDown(self):
        """Tear down test fixtures after running the tests."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)
    
    def test_clean_khmer_text(self):
        """Test the clean_khmer_text function."""
        print("Testing clean_khmer_text...")
        
        # Test with valid Khmer text
        khmer_text = "សុងហៀង"
        cleaned = clean_khmer_text(khmer_text)
        self.assertEqual(cleaned, khmer_text)
        
        # Test with Khmer text containing spaces
        khmer_text = "សុង ហៀង"
        cleaned = clean_khmer_text(khmer_text)
        self.assertEqual(cleaned, "សុងហៀង")
        
        # Test with Khmer text containing commas
        khmer_text = "សុងហៀង,"
        cleaned = clean_khmer_text(khmer_text)
        self.assertEqual(cleaned, "សុងហៀង")
        
        print("✅ clean_khmer_text works")
    
    def test_normalize_latin_text(self):
        """Test the normalize_latin_text function."""
        print("Testing normalize_latin_text...")
        
        # Test with valid Latin text
        latin_text = "songhieng"
        normalized = normalize_latin_text(latin_text)
        self.assertEqual(normalized, latin_text)
        
        # Test with Latin text containing spaces
        latin_text = "song hieng"
        normalized = normalize_latin_text(latin_text)
        self.assertEqual(normalized, "songhieng")
        
        # Test with Latin text containing uppercase
        latin_text = "SongHieng"
        normalized = normalize_latin_text(latin_text)
        self.assertEqual(normalized, "songhieng")
        
        print("✅ normalize_latin_text works")
    
    def test_khmer_name_matcher_init(self):
        """Test the KhmerNameMatcher initialization."""
        print("Testing KhmerNameMatcher initialization...")
        
        try:
            matcher = KhmerNameMatcher(names_file=self.names_file)
            self.assertIsNotNone(matcher)
            self.assertTrue(matcher.has_latin_names)
            self.assertEqual(len(matcher.khmer_names), 4)
            self.assertEqual(len(matcher.latin_map), 4)
            print("✅ KhmerNameMatcher initialization works")
        except Exception as e:
            print(f"❌ Error initializing KhmerNameMatcher: {e}")
            raise
    
    def test_khmer_name_matcher_match(self):
        """Test the KhmerNameMatcher match method."""
        print("Testing KhmerNameMatcher match method...")
        
        matcher = KhmerNameMatcher(names_file=self.names_file)
        
        # Test with Khmer text only
        results = matcher.match(khmer_text="សុងហៀង")
        self.assertGreater(len(results), 0)
        first_result = results[0]
        self.assertEqual(first_result['khmer_name'], "សុងហៀង")
        self.assertEqual(first_result['latin_name'], "songhieng")
        self.assertGreater(first_result['khmer_score'], 0.9)  # Should be a high score
        
        # Test with Latin text only
        results = matcher.match(latin_text="songhieng")
        self.assertGreater(len(results), 0)
        first_result = results[0]
        self.assertEqual(first_result['khmer_name'], "សុងហៀង")
        self.assertEqual(first_result['latin_name'], "songhieng")
        self.assertGreater(first_result['latin_score'], 0.9)  # Should be a high score
        
        # Test with both Khmer and Latin text
        results = matcher.match(khmer_text="សុងហៀង", latin_text="songhieng")
        self.assertGreater(len(results), 0)
        first_result = results[0]
        self.assertEqual(first_result['khmer_name'], "សុងហៀង")
        self.assertEqual(first_result['latin_name'], "songhieng")
        self.assertGreater(first_result['khmer_score'], 0.9)
        self.assertGreater(first_result['latin_score'], 0.9)
        self.assertGreater(first_result['combined_score'], 0.9)
        
        print("✅ KhmerNameMatcher match method works")
    
    def test_khmer_name_matcher_batch_match(self):
        """Test the KhmerNameMatcher batch_match method."""
        print("Testing KhmerNameMatcher batch_match method...")
        
        matcher = KhmerNameMatcher(names_file=self.names_file)
        
        # Prepare batch items
        items = [
            {"khmer_text": "សុងហៀង", "latin_text": "songhieng"},
            {"khmer_text": "ដារា", "latin_text": "dara"}
        ]
        
        # Perform batch match
        results = matcher.batch_match(items)
        
        # Check results
        self.assertEqual(len(results), 2)
        
        # Check first result
        first_item_results = results[0]["results"]
        self.assertGreater(len(first_item_results), 0)
        first_result = first_item_results[0]
        self.assertEqual(first_result['khmer_name'], "សុងហៀង")
        self.assertEqual(first_result['latin_name'], "songhieng")
        
        # Check second result
        second_item_results = results[1]["results"]
        self.assertGreater(len(second_item_results), 0)
        first_result = second_item_results[0]
        self.assertEqual(first_result['khmer_name'], "ដារា")
        self.assertEqual(first_result['latin_name'], "dara")
        
        print("✅ KhmerNameMatcher batch_match method works")

def run_tests():
    """Run the tests."""
    print("Running utility tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\nAll utility tests completed.")

if __name__ == "__main__":
    run_tests() 