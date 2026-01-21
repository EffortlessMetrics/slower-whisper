import sys
import unittest
from unittest.mock import MagicMock, patch
from transcription import dogfood

class TestDogfoodSecurity(unittest.TestCase):
    @patch("subprocess.run")
    @patch("transcription.dogfood.print_diarization_stats")
    @patch("transcription.dogfood.compute_diarization_stats")
    @patch("transcription.dogfood.Path.exists")
    @patch("os.getenv")
    def test_argument_injection(self, mock_getenv, mock_exists, mock_stats, mock_print_stats, mock_run):
        # Setup mocks
        mock_getenv.return_value = "fake_key"  # ANTHROPIC_API_KEY

        # Path exists logic:
        # 1. args.file (-test.wav) -> True
        # 2. json_file (whisper_json/-test.json) -> True (for skip-transcribe)
        # 3. llm_example -> True
        def side_effect(self):
            return True
        mock_exists.return_value = True # Simplify: everything exists

        # Mock run result
        mock_run.return_value.stdout = "LLM Output"
        mock_run.return_value.returncode = 0

        # Run dogfood with malicious filename
        # We need to simulate the file path carefully because dogfood uses Path object
        # and checking args.file.exists()

        # Use --file=-test.wav to ensure argparse accepts it as a value, not a flag
        test_args = ["--file=-test.wav", "--skip-transcribe", "--root", "."]

        # We need to mock sys.argv or pass argv to main
        # dogfood.main takes argv

        ret = dogfood.main(test_args)

        # Check subprocess call
        # Expected: ["python", "examples/llm_integration/summarize_with_diarization.py", "whisper_json/-test.json"]
        # If vulnerable, it lacks "--" before the json file.

        found = False
        for call in mock_run.call_args_list:
            args = call[0][0] # First arg is the command list
            print(f"Subprocess called with: {args}")
            if "examples/llm_integration/summarize_with_diarization.py" in str(args[1]):
                found = True
                # Check for argument injection vulnerability
                # If secure, index 2 should be "--" and index 3 should be json file
                # If insecure, index 2 is json file
                if args[2] == "--":
                    print("Secure: Found '--' separator")
                    self.assertTrue(args[3].endswith("-test.json"))
                else:
                    print("Vulnerable: No '--' separator found")
                    self.fail("Missing '--' separator before filename")

        self.assertTrue(found, "LLM integration script was not called")

if __name__ == "__main__":
    unittest.main()
