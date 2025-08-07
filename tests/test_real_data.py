import unittest
import os
import pandas as pd
import shutil
import subprocess
import glob


def _get_tator_token(filepath: str = "api.txt"):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        token = f.read().strip()
    return token if token else None


@unittest.skipIf(
    _get_tator_token() is None,
    "A valid API token is required in 'api.txt' file in the project root",
)
class TestRealData(unittest.TestCase):
    def setUp(self):
        self.test_dir = "temp"
        self.site_name = "trinity-2_20250404T173830"
        self.image_name = "temp/images/trinity-2_220250404T173830/trinity-2_20250404T173830_Seymour_DSC02156.JPG"

        self.images_dir = os.path.join(self.test_dir, "images", self.site_name)
        self.results_dir = os.path.join(self.test_dir, "results")
        self.debug_dir = os.path.join(self.results_dir, "debug")

        self.test_data_path = "test_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_setup(self):
        tator_token = _get_tator_token()

        print("Running setup...")
        cmd_setup = [
            "python",
            "-m",
            "kelp-coverage",
            "setup",
            "--tator-csv",
            os.path.join(self.test_data, "seymour.csv"),
            "--images",
            "1",
            "--tator-token",
            tator_token,
        ]
        r1 = subprocess.run(
            cmd_setup, capture_output=True, text=True, cwd=self.test_dir
        )
        self.assertEqual(
            r1.returncode, 0, f"Setup script failed on first run:\n{r1.stderr}"
        )
        self.assertTrue(
            os.path.isdir(os.path.join(self.test_dir, "images")),
            "'images' directory was not created.",
        )
        self.assertTrue(
            os.path.isdir(os.path.join(self.test_dir, "results")),
            "'results' directory was not created.",
        )

        downloaded_files = os.listdir(self.images_dir)
        self.assertEqual(
            len(downloaded_files), 1, "Expected exactly one image to be downloaded."
        )
        downloaded_image_path = os.path.join(self.images_dir, downloaded_files[0])
        self.assertTrue(
            os.path.exists(downloaded_image_path), "Downloaded image file not found."
        )

        print("\nRunning setup (second pass)...")
        result2 = subprocess.run(
            cmd_setup, capture_output=True, text=True, cwd=self.test_dir
        )
        self.assertEqual(
            result2.returncode,
            0,
            f"Setup script failed on second run:\n{result2.stderr}",
        )

        self.assertIn(
            "Skipping",
            result2.stdout,
            "The 'Skipping' message was not found in the output on the second run.",
        )
        self.assertEqual(
            len(os.listdir(self.images_dir)),
            1,
            "No new images should have been downloaded.",
        )
        print("[SUCCESS] Download and skip logic test passed.")


if __name__ == "__main__":
    unittest.main()
