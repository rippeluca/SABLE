"""Token entropy and the five-draw summary. No API calls."""

from __future__ import annotations

import math
import unittest

from sable.uncertainty import distribution_entropy, mean_token_entropy, summarize


class UncertaintyTests(unittest.TestCase):
    def test_two_equal_outcomes_are_one_bit(self):
        half = math.log(0.5)
        self.assertAlmostEqual(distribution_entropy([half, half]), 1.0)

    def test_residual_mass_is_one_extra_outcome(self):
        self.assertAlmostEqual(distribution_entropy([math.log(0.5)]), 1.0)

    def test_a_certain_token_has_no_entropy(self):
        self.assertAlmostEqual(distribution_entropy([0.0]), 0.0)

    def test_mean_is_over_every_token(self):
        certain = [0.0]
        even = [math.log(0.5), math.log(0.5)]
        self.assertAlmostEqual(mean_token_entropy([certain, even]), 0.5)

    def test_summary_averages_only_the_majority_replies(self):
        samples = [
            {"label": "supported", "token_entropy": 0.2, "n_tokens": 10},
            {"label": "supported", "token_entropy": 0.4, "n_tokens": 30},
            {"label": "unclear", "token_entropy": 9.0, "n_tokens": 100},
            {"label": "supported", "token_entropy": 0.6, "n_tokens": 20},
            {"label": "supported", "token_entropy": 0.8, "n_tokens": 20},
        ]
        summary = summarize(samples)
        self.assertEqual(summary["majority"], "supported")
        self.assertAlmostEqual(summary["agreement"], 0.8)
        self.assertAlmostEqual(summary["token_entropy"], 0.5)
        self.assertEqual(summary["majority_draws"], 4)
        self.assertAlmostEqual(summary["mean_tokens"], 20.0)

    def test_a_tie_keeps_the_label_that_appeared_first(self):
        samples = [
            {"label": "unclear", "token_entropy": 1.0, "n_tokens": 5},
            {"label": "supported", "token_entropy": 2.0, "n_tokens": 5},
            {"label": "unclear", "token_entropy": 3.0, "n_tokens": 5},
            {"label": "supported", "token_entropy": 4.0, "n_tokens": 5},
            {"label": "not supported", "token_entropy": 5.0, "n_tokens": 5},
        ]
        summary = summarize(samples)
        self.assertEqual(summary["majority"], "unclear")
        self.assertAlmostEqual(summary["agreement"], 0.4)
        self.assertAlmostEqual(summary["token_entropy"], 2.0)
