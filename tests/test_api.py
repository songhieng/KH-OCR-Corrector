"""
Test the API for Khmer Name OCR Corrector.
"""
import os
import sys
import unittest
import requests
import subprocess
import time
import signal
import atexit
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get API process (for cleanup)
api_process = None

def start_api_server(sample_file="sample.csv"):
    """Start the API server for testing."""
    print("Starting API server for testing...")
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_file_path = os.path.join(project_root, sample_file)
    
    # Make sure sample file exists
    if not os.path.exists(sample_file_path):
        print(f"Creating sample file at {sample_file_path}")
        with open(sample_file_path, "w", encoding="utf-8") as f:
            f.write("songhieng,សុងហៀង\n")
            f.write("dara,ដារា\n")
            f.write("sok,សុខ\n")
            f.write("vichet,វិចិត្រ\n")
    
    # Start the API server
    env = os.environ.copy()
    env["NAMES_FILE"] = sample_file_path
    
    global api_process
    api_process = subprocess.Popen(
        ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=project_root
    )
    
    # Give the server time to start
    print("Waiting for API server to start...")
    time.sleep(5)
    print("API server should be running now.")
    
    # Register cleanup
    atexit.register(stop_api_server)

def stop_api_server():
    """Stop the API server after testing."""
    global api_process
    if api_process:
        print("Stopping API server...")
        api_process.terminate()
        try:
            api_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_process.kill()
        api_process = None
        print("API server stopped.")

class TestAPI(unittest.TestCase):
    """Test the API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running the tests."""
        # API server should already be running
        cls.base_url = "http://localhost:8000"
        
        # Check if the API server is running
        try:
            response = requests.get(f"{cls.base_url}/")
            if response.status_code != 200:
                print(f"API server response code: {response.status_code}")
                print("API server may not be running correctly.")
        except requests.exceptions.ConnectionError:
            print("ERROR: Could not connect to API server. Make sure it's running.")
            sys.exit(1)
    
    def test_root(self):
        """Test the root endpoint."""
        print("Testing root endpoint...")
        try:
            response = requests.get(f"{self.base_url}/")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("name", data)
            self.assertIn("version", data)
            self.assertIn("status", data)
            print("✅ Root endpoint works")
        except Exception as e:
            print(f"❌ Error testing root endpoint: {e}")
            raise
    
    def test_info(self):
        """Test the info endpoint."""
        print("Testing info endpoint...")
        try:
            response = requests.get(f"{self.base_url}/info")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("name", data)
            self.assertIn("device", data)
            self.assertIn("has_latin_names", data)
            self.assertIn("num_names", data)
            print("✅ Info endpoint works")
        except Exception as e:
            print(f"❌ Error testing info endpoint: {e}")
            raise
    
    def test_match(self):
        """Test the match endpoint."""
        print("Testing match endpoint...")
        try:
            data = {
                "khmer_text": "សុងហៀង",
                "latin_text": "songhieng",
                "top_k": 3,
                "semantic_weight": 0.7,
                "latin_weight": 0.5
            }
            
            response = requests.post(f"{self.base_url}/match", json=data)
            self.assertEqual(response.status_code, 200)
            
            result = response.json()
            self.assertIn("results", result)
            self.assertGreater(len(result["results"]), 0)
            
            # Check the first result
            first_result = result["results"][0]
            self.assertIn("khmer_name", first_result)
            self.assertIn("latin_name", first_result)
            self.assertIn("khmer_score", first_result)
            self.assertIn("latin_score", first_result)
            self.assertIn("combined_score", first_result)
            
            print("✅ Match endpoint works")
        except Exception as e:
            print(f"❌ Error testing match endpoint: {e}")
            raise
    
    def test_batch_match(self):
        """Test the batch-match endpoint."""
        print("Testing batch-match endpoint...")
        try:
            data = {
                "items": [
                    {
                        "khmer_text": "សុងហៀង",
                        "latin_text": "songhieng"
                    },
                    {
                        "khmer_text": "ដារា",
                        "latin_text": "dara"
                    }
                ]
            }
            
            response = requests.post(f"{self.base_url}/batch-match", json=data)
            self.assertEqual(response.status_code, 200)
            
            result = response.json()
            self.assertIn("results", result)
            self.assertEqual(len(result["results"]), 2)
            
            print("✅ Batch-match endpoint works")
        except Exception as e:
            print(f"❌ Error testing batch-match endpoint: {e}")
            raise

def run_tests():
    """Run the API tests."""
    start_api_server()
    print("Running API tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\nAll API tests completed.")

if __name__ == "__main__":
    run_tests()