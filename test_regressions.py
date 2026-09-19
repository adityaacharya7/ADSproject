"""Run with: python -m unittest test_regressions -v"""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from model_engine import EmotionEngine
from text_preprocessing import apply_negation_tagging, clean_tweet_text


class ProductionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = EmotionEngine()

    def test_real_artifact_inference_does_not_fall_back(self):
        engine = self.engine
        self.assertIsNotNone(engine.pipeline)
        messages = ['Thank you for your excellent help!',
                    'This is not helpful at all, but the agent tried.',
                    'My account was hacked and I am scared!']
        with patch.object(engine, '_heuristic_probabilities',
                          side_effect=AssertionError('Unexpected fallback')):
            for text in messages:
                expected = engine.pipeline.predict_proba([text])[0]
                result = engine.predict(text)
                self.assertEqual(list(result['probabilities']), list(engine.pipeline.classes_))
                np.testing.assert_allclose(list(result['probabilities'].values()), expected, atol=0.00005)
                self.assertEqual(result['primary_emotion'], str(engine.pipeline.predict([text])[0]))

    def test_training_negation_and_cleaning(self):
        text = clean_tweet_text('@support I am not happy with this! https://example.com')
        self.assertEqual(apply_negation_tagging(text), 'i am not happy_NEG with_NEG this_NEG !')

    def test_drift_is_invariant_to_sample_repetition(self):
        texts = ['Thank you for your help!', 'My package is missing.', 'When will it arrive?']
        small = self.engine.check_data_drift(texts)
        large = self.engine.check_data_drift(texts * 100)
        self.assertEqual(small['feature_psi'], large['feature_psi'])
        self.assertEqual(small['status'], large['status'])

    def test_psi_detects_distribution_shift_with_equal_means(self):
        reference = {'cut_points': [1, 3], 'proportions': [0.5, 0, 0.5]}
        self.assertAlmostEqual(self.engine._population_stability_index([0, 4], reference), 0)
        self.assertGreater(self.engine._population_stability_index([2, 2], reference), 0.25)

    def test_psi_handles_empty_bins_and_extreme_values(self):
        score = self.engine._population_stability_index(
            [0, 10000], self.engine.drift_baseline['length'])
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0)

    def test_dashboard_renders_all_tabs(self):
        with patch.object(EmotionEngine, '_heuristic_probabilities',
                          side_effect=AssertionError('Dashboard used fallback')):
            app = AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=60)
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertEqual(len(app.tabs), 6)


if __name__ == '__main__':
    unittest.main()
