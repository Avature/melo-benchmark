import unittest

from melo_benchmark.analysis.statistical_significance import \
    StatisticalSignificanceAnalyzer


# noinspection DuplicatedCode
class TestEscoLoader(unittest.TestCase):

    def test_wilcoxon(self):
        alpha = 0.05
        analyzer = StatisticalSignificanceAnalyzer()

        a_res = [
            0.7386,
            0.6814,
            0.7015,
            0.6794,
            0.6850,
            0.6832,
            0.6918,
            0.7038,
            0.6762,
            0.6709,
            0.6783
        ]
        b_res = [
            0.4955,
            0.3740,
            0.4359,
            0.4067,
            0.4057,
            0.3920,
            0.3840,
            0.4213,
            0.4047,
            0.3909,
            0.4693
        ]

        result = analyzer.wilcoxon_test(a_res, b_res)
        p_value = float(result.pvalue)

        self.assertLess(p_value, alpha)

        a_res = [
            0.7386,
            0.6814,
            0.7015,
            0.6794,
            0.6850,
            0.6832,
            0.6918,
            0.7038,
            0.6762,
            0.6709,
            0.6783
        ]
        b_res = [
            0.7384,
            0.6764,
            0.7105,
            0.6712,
            0.6957,
            0.6892,
            0.6820,
            0.7053,
            0.6778,
            0.6243,
            0.6860
        ]

        result = analyzer.wilcoxon_test(a_res, b_res)
        p_value = float(result.pvalue)

        self.assertGreater(p_value, alpha)
