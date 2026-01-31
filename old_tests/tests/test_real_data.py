import unittest
import os
import pandas as pd
import shutil
import subprocess
import glob
import cv2
import numpy as np


def get_tator_token_from_file(filepath="api.txt"):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        token = f.read().strip()
    return token if token else None


class TestRealData(unittest.TestCase):
    def setUp(self):
        self.tator_token = get_tator_token_from_file("api.txt")
        if not self.tator_token:
            self.skipTest(
                "A valid Tator API token is required in an 'api.txt' file in the project root."
            )

        self.test_dir = "temp_real_data_test"
        self.site_name = "trinity-2_20250404T173830"
        self.image_name = "trinity-2_20250404T173830_Seymour_DSC02156.JPG"

        self.images_dir = os.path.join(self.test_dir, "images", self.site_name)
        self.results_dir = os.path.join(self.test_dir, "results")
        self.debug_dir = os.path.join(self.results_dir, "debug")

        self.test_data_path = os.path.abspath(os.path.join("tests", "test_data"))

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_setup_and_download_logic(self):
        print("\nInitial setup ...")

        cmd_setup = [
            "python",
            "-m",
            "kelp_coverage.cli",
            "setup",
            "--tator-csv",
            os.path.join(self.test_data_path, "seymour.csv"),
            "--start-idx",
            "0",
            "--end-idx",
            "1",
            "--tator-token",
            self.tator_token,
        ]

        result1 = subprocess.run(
            cmd_setup, capture_output=True, text=True, cwd=self.test_dir
        )
        self.assertEqual(
            result1.returncode,
            0,
            f"Setup script failed on first run:\n{result1.stderr}",
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

        print("\nValidating setup ...")
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
