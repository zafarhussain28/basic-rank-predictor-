import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from predictor import (
    APP_DIR,
    FEATURE_NAME,
    _load_models,
    predict_scores,
    validate_raw_marks,
)


class PredictorTests(unittest.TestCase):
    def test_validate_raw_marks_accepts_supported_integer(self):
        self.assertEqual(validate_raw_marks(150), 150)

    def test_validate_raw_marks_rejects_non_integer_and_out_of_range(self):
        for value in (150.0, True, "150", -1, 301):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_raw_marks(value)

    def test_missing_model_file_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "percentile_model"):
                _load_models(Path(directory))

    def test_artifacts_load_and_predict(self):
        percentile_model, rank_model = _load_models(APP_DIR)
        for model in (percentile_model, rank_model):
            self.assertEqual(tuple(model.feature_names_in_), (FEATURE_NAME,))

        percentile, rank = predict_scores(150, percentile_model, rank_model)
        self.assertTrue(np.isfinite(percentile))
        self.assertTrue(np.isfinite(rank))

    def test_prediction_failure_is_propagated(self):
        failed_model = Mock()
        failed_model.predict.side_effect = RuntimeError("model inference failed")
        with self.assertRaisesRegex(RuntimeError, "model inference failed"):
            predict_scores(150, failed_model, Mock())


if __name__ == "__main__":
    unittest.main()