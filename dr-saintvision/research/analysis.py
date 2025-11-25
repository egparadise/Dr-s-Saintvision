"""
Statistical Analysis and Visualization for DR-Saintvision Research
Tools for analyzing experiment results and generating publication-quality figures

This module provides:
- Statistical significance testing (t-tests, ANOVA, effect sizes)
- Result visualization (bar charts, heatmaps, convergence plots)
- LaTeX table generation for papers
- Automated report generation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Result of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    significant: bool  # at alpha=0.05
    interpretation: str
    details: Dict[str, Any] = None


class StatisticalAnalyzer:
    """
    Statistical analysis tools for experiment results

    Provides:
    - Paired and independent t-tests
    - One-way ANOVA
    - Effect size calculations (Cohen's d, eta-squared)
    - Confidence interval estimation
    - Non-parametric alternatives (Mann-Whitney, Kruskal-Wallis)
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def paired_t_test(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2"
    ) -> StatisticalTestResult:
        """
        Paired t-test for comparing two related samples
        Use when: Same subjects tested under two conditions
        """
        arr1, arr2 = np.array(group1), np.array(group2)

        # Perform test
        statistic, p_value = stats.ttest_rel(arr1, arr2)

        # Calculate effect size (Cohen's d for paired samples)
        diff = arr1 - arr2
        effect_size = np.mean(diff) / np.std(diff, ddof=1)

        significant = p_value < self.alpha

        interpretation = self._interpret_effect_size(effect_size)
        if significant:
            interpretation += f" {group1_name}과 {group2_name} 간 통계적으로 유의미한 차이가 있음 (p={p_value:.4f})"
        else:
            interpretation += f" 통계적으로 유의미한 차이 없음 (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            significant=significant,
            interpretation=interpretation,
            details={
                "group1_mean": float(np.mean(arr1)),
                "group2_mean": float(np.mean(arr2)),
                "group1_std": float(np.std(arr1, ddof=1)),
                "group2_std": float(np.std(arr2, ddof=1)),
                "mean_difference": float(np.mean(diff)),
                "n": len(arr1)
            }
        )

    def independent_t_test(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2"
    ) -> StatisticalTestResult:
        """
        Independent samples t-test
        Use when: Two independent groups being compared
        """
        arr1, arr2 = np.array(group1), np.array(group2)

        # Perform test (Welch's t-test for unequal variances)
        statistic, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)

        # Calculate Cohen's d
        pooled_std = np.sqrt(
            ((len(arr1) - 1) * np.var(arr1, ddof=1) +
             (len(arr2) - 1) * np.var(arr2, ddof=1)) /
            (len(arr1) + len(arr2) - 2)
        )
        effect_size = (np.mean(arr1) - np.mean(arr2)) / pooled_std if pooled_std > 0 else 0

        significant = p_value < self.alpha

        interpretation = self._interpret_effect_size(effect_size)
        if significant:
            interpretation += f" {group1_name}과 {group2_name} 간 통계적으로 유의미한 차이가 있음"
        else:
            interpretation += " 통계적으로 유의미한 차이 없음"

        return StatisticalTestResult(
            test_name="Independent t-test (Welch's)",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            significant=significant,
            interpretation=interpretation,
            details={
                "group1_mean": float(np.mean(arr1)),
                "group2_mean": float(np.mean(arr2)),
                "group1_std": float(np.std(arr1, ddof=1)),
                "group2_std": float(np.std(arr2, ddof=1)),
                "n1": len(arr1),
                "n2": len(arr2)
            }
        )

    def one_way_anova(
        self,
        groups: Dict[str, List[float]]
    ) -> StatisticalTestResult:
        """
        One-way ANOVA for comparing multiple groups
        Use when: Comparing 3+ independent groups
        """
        group_names = list(groups.keys())
        group_values = [np.array(v) for v in groups.values()]

        # Perform ANOVA
        statistic, p_value = stats.f_oneway(*group_values)

        # Calculate eta-squared (effect size)
        all_values = np.concatenate(group_values)
        grand_mean = np.mean(all_values)

        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2
            for g in group_values
        )
        ss_total = np.sum((all_values - grand_mean) ** 2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        significant = p_value < self.alpha

        interpretation = self._interpret_eta_squared(eta_squared)
        if significant:
            interpretation += " 그룹 간 통계적으로 유의미한 차이가 있음"
        else:
            interpretation += " 그룹 간 통계적으로 유의미한 차이 없음"

        return StatisticalTestResult(
            test_name="One-way ANOVA",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(eta_squared),
            significant=significant,
            interpretation=interpretation,
            details={
                "group_means": {name: float(np.mean(groups[name])) for name in group_names},
                "group_stds": {name: float(np.std(groups[name], ddof=1)) for name in group_names},
                "group_sizes": {name: len(groups[name]) for name in group_names},
                "ss_between": float(ss_between),
                "ss_total": float(ss_total)
            }
        )

    def mann_whitney_u(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2"
    ) -> StatisticalTestResult:
        """
        Mann-Whitney U test (non-parametric alternative to t-test)
        Use when: Data is not normally distributed
        """
        arr1, arr2 = np.array(group1), np.array(group2)

        statistic, p_value = stats.mannwhitneyu(arr1, arr2, alternative='two-sided')

        # Calculate effect size (r = Z / sqrt(N))
        n = len(arr1) + len(arr2)
        z_score = stats.norm.ppf(1 - p_value / 2)
        effect_size = z_score / np.sqrt(n)

        significant = p_value < self.alpha

        interpretation = f"효과 크기 r={effect_size:.3f}."
        if significant:
            interpretation += f" {group1_name}과 {group2_name} 간 통계적으로 유의미한 차이가 있음"
        else:
            interpretation += " 통계적으로 유의미한 차이 없음"

        return StatisticalTestResult(
            test_name="Mann-Whitney U test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            significant=significant,
            interpretation=interpretation,
            details={
                "group1_median": float(np.median(arr1)),
                "group2_median": float(np.median(arr2)),
                "n1": len(arr1),
                "n2": len(arr2)
            }
        )

    def confidence_interval(
        self,
        data: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """Calculate confidence interval for a sample"""
        arr = np.array(data)
        mean = np.mean(arr)
        se = stats.sem(arr)
        ci = stats.t.interval(confidence, len(arr) - 1, loc=mean, scale=se)
        return float(mean), float(ci[0]), float(ci[1])

    def bootstrap_ci(
        self,
        data: List[float],
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """Bootstrap confidence interval (non-parametric)"""
        arr = np.array(data)
        means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            means.append(np.mean(sample))

        alpha = (1 - confidence) / 2
        lower = np.percentile(means, alpha * 100)
        upper = np.percentile(means, (1 - alpha) * 100)

        return float(np.mean(arr)), float(lower), float(upper)

    def normality_test(self, data: List[float]) -> Dict[str, Any]:
        """Test for normality using Shapiro-Wilk test"""
        arr = np.array(data)
        statistic, p_value = stats.shapiro(arr)

        return {
            "test": "Shapiro-Wilk",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > self.alpha,
            "interpretation": "정규분포" if p_value > self.alpha else "비정규분포"
        }

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        d = abs(d)
        if d < 0.2:
            return "효과 크기: 미미함 (negligible)."
        elif d < 0.5:
            return "효과 크기: 작음 (small)."
        elif d < 0.8:
            return "효과 크기: 중간 (medium)."
        else:
            return "효과 크기: 큼 (large)."

    def _interpret_eta_squared(self, eta: float) -> str:
        """Interpret eta-squared effect size"""
        if eta < 0.01:
            return "효과 크기: 미미함 (negligible)."
        elif eta < 0.06:
            return "효과 크기: 작음 (small)."
        elif eta < 0.14:
            return "효과 크기: 중간 (medium)."
        else:
            return "효과 크기: 큼 (large)."

    def compare_multi_agent_vs_single(
        self,
        multi_agent_results: List[float],
        single_model_results: List[float]
    ) -> Dict[str, Any]:
        """
        Complete statistical comparison between multi-agent and single model
        Returns comprehensive analysis suitable for paper
        """
        # Check normality
        ma_normality = self.normality_test(multi_agent_results)
        sm_normality = self.normality_test(single_model_results)

        both_normal = ma_normality["is_normal"] and sm_normality["is_normal"]

        # Choose appropriate test
        if both_normal:
            test_result = self.independent_t_test(
                multi_agent_results,
                single_model_results,
                "Multi-Agent",
                "Single Model"
            )
        else:
            test_result = self.mann_whitney_u(
                multi_agent_results,
                single_model_results,
                "Multi-Agent",
                "Single Model"
            )

        # Calculate CIs
        ma_ci = self.confidence_interval(multi_agent_results)
        sm_ci = self.confidence_interval(single_model_results)

        # Improvement percentage
        ma_mean = np.mean(multi_agent_results)
        sm_mean = np.mean(single_model_results)
        improvement = ((ma_mean - sm_mean) / sm_mean * 100) if sm_mean > 0 else 0

        return {
            "multi_agent": {
                "mean": float(ma_mean),
                "std": float(np.std(multi_agent_results, ddof=1)),
                "ci_95": (ma_ci[1], ma_ci[2]),
                "normality": ma_normality
            },
            "single_model": {
                "mean": float(sm_mean),
                "std": float(np.std(single_model_results, ddof=1)),
                "ci_95": (sm_ci[1], sm_ci[2]),
                "normality": sm_normality
            },
            "comparison": {
                "test_used": test_result.test_name,
                "statistic": test_result.statistic,
                "p_value": test_result.p_value,
                "effect_size": test_result.effect_size,
                "significant": test_result.significant,
                "improvement_percent": float(improvement)
            },
            "interpretation": test_result.interpretation
        }


class ResultVisualizer:
    """
    Visualization tools for experiment results

    Generates publication-quality figures:
    - Bar charts with error bars
    - Convergence plots
    - Heatmaps for confusion matrices
    - Box plots for distribution comparison
    - LaTeX tables for papers
    """

    def __init__(self, output_dir: str = "./figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Try to import matplotlib (optional dependency)
        self.plt = None
        self.sns = None
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            self.plt = plt

            try:
                import seaborn as sns
                self.sns = sns
                sns.set_style("whitegrid")
                sns.set_context("paper", font_scale=1.2)
            except ImportError:
                logger.warning("seaborn not installed - some visualizations unavailable")
        except ImportError:
            logger.warning("matplotlib not installed - visualizations unavailable")

    def bar_chart_comparison(
        self,
        data: Dict[str, float],
        errors: Optional[Dict[str, float]] = None,
        title: str = "Performance Comparison",
        ylabel: str = "Score",
        filename: str = "comparison.png"
    ) -> Optional[str]:
        """Create bar chart comparing different configurations"""
        if self.plt is None:
            logger.warning("matplotlib not available")
            return None

        fig, ax = self.plt.subplots(figsize=(10, 6))

        names = list(data.keys())
        values = list(data.values())
        error_values = [errors.get(n, 0) for n in names] if errors else None

        colors = self.sns.color_palette("husl", len(names)) if self.sns else None

        bars = ax.bar(names, values, yerr=error_values, capsize=5, color=colors)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.2)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

        self.plt.tight_layout()
        save_path = self.output_dir / filename
        self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.plt.close()

        logger.info(f"Saved bar chart to {save_path}")
        return str(save_path)

    def convergence_plot(
        self,
        rounds: List[int],
        scores: Dict[str, List[float]],
        title: str = "Convergence Over Debate Rounds",
        filename: str = "convergence.png"
    ) -> Optional[str]:
        """Plot convergence of metrics over debate rounds"""
        if self.plt is None:
            return None

        fig, ax = self.plt.subplots(figsize=(10, 6))

        for name, values in scores.items():
            ax.plot(rounds[:len(values)], values, marker='o', label=name, linewidth=2)

        ax.set_xlabel("Debate Round")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.plt.tight_layout()
        save_path = self.output_dir / filename
        self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.plt.close()

        return str(save_path)

    def box_plot_comparison(
        self,
        data: Dict[str, List[float]],
        title: str = "Distribution Comparison",
        ylabel: str = "Score",
        filename: str = "boxplot.png"
    ) -> Optional[str]:
        """Create box plot comparing distributions"""
        if self.plt is None:
            return None

        fig, ax = self.plt.subplots(figsize=(10, 6))

        if self.sns:
            import pandas as pd
            # Convert to long format for seaborn
            records = []
            for name, values in data.items():
                for v in values:
                    records.append({"Configuration": name, "Score": v})
            df = pd.DataFrame(records)
            self.sns.boxplot(data=df, x="Configuration", y="Score", ax=ax)
        else:
            ax.boxplot(data.values(), labels=data.keys())

        ax.set_ylabel(ylabel)
        ax.set_title(title)

        self.plt.tight_layout()
        save_path = self.output_dir / filename
        self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.plt.close()

        return str(save_path)

    def heatmap(
        self,
        matrix: List[List[float]],
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Heatmap",
        filename: str = "heatmap.png"
    ) -> Optional[str]:
        """Create heatmap visualization"""
        if self.plt is None or self.sns is None:
            return None

        fig, ax = self.plt.subplots(figsize=(10, 8))

        self.sns.heatmap(
            matrix,
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            ax=ax
        )

        ax.set_title(title)

        self.plt.tight_layout()
        save_path = self.output_dir / filename
        self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.plt.close()

        return str(save_path)

    def multi_metric_radar(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Multi-Metric Comparison",
        filename: str = "radar.png"
    ) -> Optional[str]:
        """Create radar/spider chart for multi-metric comparison"""
        if self.plt is None:
            return None

        # Get metrics (assume all configs have same metrics)
        first_config = list(data.values())[0]
        metrics = list(first_config.keys())
        num_metrics = len(metrics)

        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        fig, ax = self.plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = self.sns.color_palette("husl", len(data)) if self.sns else None

        for i, (config_name, metrics_dict) in enumerate(data.items()):
            values = [metrics_dict[m] for m in metrics]
            values += values[:1]  # Complete the loop

            color = colors[i] if colors else None
            ax.plot(angles, values, 'o-', linewidth=2, label=config_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        self.plt.tight_layout()
        save_path = self.output_dir / filename
        self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.plt.close()

        return str(save_path)

    def generate_latex_table(
        self,
        data: Dict[str, Dict[str, Any]],
        caption: str = "Experimental Results",
        label: str = "tab:results"
    ) -> str:
        """Generate LaTeX table from results"""
        if not data:
            return ""

        # Get all metrics
        first_config = list(data.values())[0]
        metrics = list(first_config.keys())
        configs = list(data.keys())

        # Build LaTeX table
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")

        # Column format
        col_format = "l" + "c" * len(metrics)
        latex.append(f"\\begin{{tabular}}{{{col_format}}}")
        latex.append("\\toprule")

        # Header
        header = "Configuration & " + " & ".join(metrics) + " \\\\"
        latex.append(header)
        latex.append("\\midrule")

        # Data rows
        for config in configs:
            values = []
            for metric in metrics:
                val = data[config].get(metric, 0)
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            row = f"{config} & " + " & ".join(values) + " \\\\"
            latex.append(row)

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        return "\n".join(latex)

    def generate_latex_comparison_table(
        self,
        comparison_result: Dict[str, Any],
        caption: str = "Multi-Agent vs Single Model Comparison",
        label: str = "tab:comparison"
    ) -> str:
        """Generate LaTeX table for multi-agent vs single model comparison"""
        ma = comparison_result["multi_agent"]
        sm = comparison_result["single_model"]
        comp = comparison_result["comparison"]

        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")
        latex.append("\\begin{tabular}{lccc}")
        latex.append("\\toprule")
        latex.append("Metric & Multi-Agent & Single Model & Improvement \\\\")
        latex.append("\\midrule")

        # Mean ± Std
        latex.append(
            f"Mean (SD) & {ma['mean']:.4f} ({ma['std']:.4f}) & "
            f"{sm['mean']:.4f} ({sm['std']:.4f}) & "
            f"{comp['improvement_percent']:.1f}\\% \\\\"
        )

        # 95% CI
        latex.append(
            f"95\\% CI & [{ma['ci_95'][0]:.4f}, {ma['ci_95'][1]:.4f}] & "
            f"[{sm['ci_95'][0]:.4f}, {sm['ci_95'][1]:.4f}] & - \\\\"
        )

        latex.append("\\midrule")

        # Statistical test
        sig_marker = "$^{***}$" if comp['significant'] else ""
        latex.append(
            f"Statistical Test & \\multicolumn{{3}}{{c}}{{{comp['test_used']}: "
            f"$p = {comp['p_value']:.4f}${sig_marker}, "
            f"effect size = {comp['effect_size']:.3f}}} \\\\"
        )

        latex.append("\\bottomrule")
        latex.append("\\multicolumn{4}{l}{\\footnotesize $^{***}p < 0.05$} \\\\")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        return "\n".join(latex)


class ReportGenerator:
    """
    Automated report generation for experiment results
    Combines statistical analysis and visualization
    """

    def __init__(
        self,
        analyzer: StatisticalAnalyzer,
        visualizer: ResultVisualizer,
        output_dir: str = "./reports"
    ):
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_experiment_report(
        self,
        experiment_name: str,
        results: Dict[str, Any],
        comparisons: Optional[Dict[str, List[float]]] = None
    ) -> str:
        """Generate comprehensive experiment report"""
        report = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report.append(f"# Experiment Report: {experiment_name}")
        report.append(f"Generated: {timestamp}\n")

        report.append("## Summary")
        for key, value in results.items():
            if isinstance(value, float):
                report.append(f"- **{key}**: {value:.4f}")
            elif isinstance(value, dict):
                report.append(f"- **{key}**:")
                for k, v in value.items():
                    if isinstance(v, float):
                        report.append(f"  - {k}: {v:.4f}")
                    else:
                        report.append(f"  - {k}: {v}")
            else:
                report.append(f"- **{key}**: {value}")

        if comparisons:
            report.append("\n## Statistical Comparisons")

            # If we have multi-agent vs single model
            if "multi_agent" in comparisons and "single_model" in comparisons:
                comparison = self.analyzer.compare_multi_agent_vs_single(
                    comparisons["multi_agent"],
                    comparisons["single_model"]
                )

                report.append("\n### Multi-Agent vs Single Model")
                report.append(f"- Multi-Agent Mean: {comparison['multi_agent']['mean']:.4f} "
                            f"(±{comparison['multi_agent']['std']:.4f})")
                report.append(f"- Single Model Mean: {comparison['single_model']['mean']:.4f} "
                            f"(±{comparison['single_model']['std']:.4f})")
                report.append(f"- Improvement: {comparison['comparison']['improvement_percent']:.1f}%")
                report.append(f"- Statistical Test: {comparison['comparison']['test_used']}")
                report.append(f"- p-value: {comparison['comparison']['p_value']:.4f}")
                report.append(f"- Effect Size: {comparison['comparison']['effect_size']:.3f}")
                report.append(f"- Significant: {'Yes' if comparison['comparison']['significant'] else 'No'}")

                # Generate LaTeX table
                latex_table = self.visualizer.generate_latex_comparison_table(comparison)
                report.append("\n### LaTeX Table")
                report.append("```latex")
                report.append(latex_table)
                report.append("```")

        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / f"{experiment_name}_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"Report saved to {report_path}")
        return report_text


# Convenience functions for quick analysis
def quick_comparison(
    multi_agent_scores: List[float],
    single_model_scores: List[float]
) -> Dict[str, Any]:
    """Quick statistical comparison between multi-agent and single model"""
    analyzer = StatisticalAnalyzer()
    return analyzer.compare_multi_agent_vs_single(multi_agent_scores, single_model_scores)


def visualize_results(
    results: Dict[str, float],
    errors: Optional[Dict[str, float]] = None,
    output_dir: str = "./figures"
) -> Optional[str]:
    """Quick visualization of results"""
    visualizer = ResultVisualizer(output_dir)
    return visualizer.bar_chart_comparison(results, errors)
