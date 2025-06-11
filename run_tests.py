#!/usr/bin/env python
"""
Test runner for Khmer Name OCR Corrector.
Runs all tests and reports results.
"""
import os
import sys
import time
import argparse
from importlib import import_module

def print_header(text):
    """Print a header for a test section."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80 + "\n")

def run_all_tests(args):
    """Run all tests."""
    # Track test results
    test_results = {}
    
    # Run tests in each category
    test_modules = [
        ("Utility Tests", "tests.test_utils"),
        ("API Tests", "tests.test_api"),
        ("App Tests", "tests.test_app"),
        ("Docker Tests", "tests.test_docker")
    ]
    
    for test_name, module_name in test_modules:
        if args.test and module_name.split('.')[-1] != args.test:
            continue
            
        print_header(f"Running {test_name}")
        
        try:
            # Import the module
            module = import_module(module_name)
            
            # Run the tests
            start_time = time.time()
            module.run_tests()
            duration = time.time() - start_time
            
            # Record result
            test_results[test_name] = {
                "status": "PASSED",
                "duration": duration
            }
            
        except Exception as e:
            # Record failure
            test_results[test_name] = {
                "status": "FAILED",
                "error": str(e)
            }
            
            if not args.continue_on_error:
                print(f"Error running {test_name}: {e}")
                print("Stopping tests due to failure. Use --continue-on-error to run all tests.")
                break
    
    # Print summary
    print_header("Test Summary")
    for test_name, result in test_results.items():
        status = result["status"]
        status_str = f"[✓] {status}" if status == "PASSED" else f"[✗] {status}"
        
        if status == "PASSED":
            duration = result.get("duration", 0)
            print(f"{test_name}: {status_str} in {duration:.2f}s")
        else:
            error = result.get("error", "Unknown error")
            print(f"{test_name}: {status_str} - {error}")
    
    # Determine overall success
    all_passed = all(result["status"] == "PASSED" for result in test_results.values())
    
    print("\nOverall Result:", "PASSED" if all_passed else "FAILED")
    return 0 if all_passed else 1

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run tests for Khmer Name OCR Corrector")
    parser.add_argument("--test", type=str, help="Run only a specific test module (e.g., test_utils)")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running tests even if one fails")
    
    args = parser.parse_args()
    
    return run_all_tests(args)

if __name__ == "__main__":
    sys.exit(main()) 