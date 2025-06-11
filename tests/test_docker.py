"""
Test Docker setup for Khmer Name OCR Corrector.
"""
import os
import sys
import unittest
import subprocess
import time
import requests
import docker
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDocker(unittest.TestCase):
    """Test the Docker setup."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running the tests."""
        print("Setting up Docker tests...")
        
        # Get project root
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check if Docker is available
        try:
            client = docker.from_env()
            cls.client = client
            print("Docker is available.")
        except Exception as e:
            print(f"Docker is not available: {e}")
            print("Skipping Docker tests.")
            sys.exit(0)
        
        # Build the Docker image
        print("Building Docker image...")
        try:
            cls.image, _ = client.images.build(
                path=cls.project_root,
                tag="khmer-name-corrector:test",
                rm=True
            )
            print(f"Docker image built: {cls.image.id}")
        except Exception as e:
            print(f"Error building Docker image: {e}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test fixtures after running the tests."""
        print("Tearing down Docker tests...")
        
        # Clean up containers
        for container in cls.client.containers.list(all=True, filters={"name": "khmer-name-corrector-test"}):
            print(f"Removing container: {container.id}")
            container.remove(force=True)
        
        # Clean up images
        try:
            cls.client.images.remove("khmer-name-corrector:test")
            print("Removed test image.")
        except Exception as e:
            print(f"Error removing image: {e}")
    
    def test_docker_build(self):
        """Test that the Docker image builds successfully."""
        print("Testing Docker image build...")
        
        self.assertIsNotNone(self.image)
        self.assertEqual(self.image.tags[0], "khmer-name-corrector:test")
        
        print("✅ Docker image builds successfully")
    
    def test_docker_run_api(self):
        """Test that the Docker image runs the API correctly."""
        print("Testing Docker API container...")
        
        # Run the Docker container with the API
        container = self.client.containers.run(
            "khmer-name-corrector:test",
            command="python -m api.main",
            ports={"8000/tcp": 8000},
            environment={"NAMES_FILE": "/app/sample.csv"},
            detach=True,
            name="khmer-name-corrector-test-api"
        )
        
        try:
            # Wait for the API to start
            print("Waiting for API to start...")
            time.sleep(5)
            
            # Test the API
            response = requests.get("http://localhost:8000/")
            self.assertEqual(response.status_code, 200)
            
            print("✅ Docker API container runs correctly")
        finally:
            # Stop and remove the container
            container.stop()
            container.remove()

def run_tests():
    """Run the Docker tests."""
    print("Running Docker tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\nAll Docker tests completed.")

if __name__ == "__main__":
    run_tests() 