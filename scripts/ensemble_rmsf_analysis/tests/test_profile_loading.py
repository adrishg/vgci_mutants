import pandas as pd
import unittest

from scripts.ensemble_rmsf_analysis.io import detect_profile_schema


class ProfileLoadingTests(unittest.TestCase):
    def test_schema_detection(self):
        frame = pd.DataFrame(columns=[
            "dataset", "sequence_condition", "protocol", "raw_residue_number",
            "ensemble_rmsf_A", "coverage_fraction",
        ])
        schema = detect_profile_schema(frame, "nav15")
        self.assertEqual(schema["residue"], "raw_residue_number")
        self.assertEqual(schema["rmsf"], "ensemble_rmsf_A")

    def test_missing_rmsf_fails_clearly(self):
        frame = pd.DataFrame(columns=[
            "dataset", "sequence_condition", "protocol", "raw_residue_number",
            "coverage_fraction",
        ])
        with self.assertRaisesRegex(ValueError, "rmsf"):
            detect_profile_schema(frame, "cav12")
