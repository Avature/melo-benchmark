from typing import (
    Any,
    List
)

from scipy.stats import wilcoxon


class StatisticalSignificanceAnalyzer:

    @staticmethod
    def wilcoxon_test(
                method_a_results: List[float],
                method_b_results: List[float]
            ) -> Any:

        # Wilcoxon signed-rank test
        return wilcoxon(method_a_results, method_b_results)
