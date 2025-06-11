"""
Test the Streamlit app for Khmer Name OCR Corrector.
"""
import os
import sys
import unittest
import tempfile
import shutil
import time
import subprocess
import requests
import socket
from pathlib import Path
from contextlib import closing

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get app process (for cleanup)
app_process = None

def find_free_port():
    """Find a free port for the Streamlit app."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def start_streamlit_app(names_file="sample.csv"):
    """Start the Streamlit app for testing."""
    print("Starting Streamlit app for testing...")
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_file_path = os.path.join(project_root, names_file)
    
    # Make sure sample file exists
    if not os.path.exists(sample_file_path):
        print(f"Creating sample file at {sample_file_path}")
        with open(sample_file_path, "w", encoding="utf-8") as f:
            f.write("songhieng,សុងហៀង\n")
            f.write("dara,ដារា\n")
            f.write("sok,សុខ\n")
            f.write("vichet,វិចិត្រ\n")
    
    # Find a free port
    port = find_free_port()
    print(f"Found free port: {port}")
    
    # Start the Streamlit app
    env = os.environ.copy()
    env["NAMES_FILE"] = sample_file_path
    
    global app_process
    app_process = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", str(port), "--server.headless", "true"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=project_root
    )
    
    # Register a function to stop the app when the test is done
    import atexit
    atexit.register(stop_streamlit_app)
    
    # Give the app time to start
    print("Waiting for Streamlit app to start...")
    time.sleep(10)
    
    return port

def stop_streamlit_app():
    """Stop the Streamlit app after testing."""
    global app_process
    if app_process:
        print("Stopping Streamlit app...")
        app_process.terminate()
        try:
            app_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            app_process.kill()
        app_process = None
        print("Streamlit app stopped.")

class TestStreamlitApp(unittest.TestCase):
    """Test the Streamlit app."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running the tests."""
        # Start the Streamlit app
        cls.app_port = start_streamlit_app()
        cls.app_url = f"http://localhost:{cls.app_port}"
        
        # Check if the app is running
        try:
            response = requests.get(cls.app_url)
            if response.status_code != 200:
                print(f"Streamlit app response code: {response.status_code}")
                print("Streamlit app may not be running correctly.")
            else:
                print(f"Streamlit app is running at {cls.app_url}")
        except requests.exceptions.ConnectionError:
            print("ERROR: Could not connect to Streamlit app. Make sure it's running.")
            sys.exit(1)
    
    def test_app_running(self):
        """Test that the Streamlit app is running."""
        print("Testing if Streamlit app is running...")
        try:
            response = requests.get(self.app_url)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Streamlit", response.text)  # Streamlit should be in the HTML
            print("✅ Streamlit app is running")
        except Exception as e:
            print(f"❌ Error testing Streamlit app: {e}")
            raise

def run_tests():
    """Run the Streamlit app tests."""
    print("Running Streamlit app tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\nAll Streamlit app tests completed.")

if __name__ == "__main__":
    run_tests() 