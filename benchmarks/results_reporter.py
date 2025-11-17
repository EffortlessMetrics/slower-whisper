"""
Results reporting module for audio enrichment benchmarks.

This module generates human-readable reports from benchmark results in multiple formats:
- CSV files for data analysis
- Markdown tables for documentation
- Summary statistics
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """Generate reports from benchmark results."""

    def __init__(self, results_file: Path):
        """
        Initialize reporter with results file.

        Args:
            results_file: Path to JSON results file
        """
        self.results_file = Path(results_file)
        self.output_dir = self.results_file.parent

        # Load results
        with open(self.results_file) as f:
            data = json.load(f)

        self.benchmark_info = data["benchmark_info"]
        self.results = data["results"]

        logger.info(f"Loaded {len(self.results)} benchmark results")

    def generate_csv_report(self, output_path: Path | None = None) -> Path:
        """
        Generate CSV report with all benchmark data.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to generated CSV file
        """
        if output_path is None:
            output_path = self.output_dir / f"{self.results_file.stem}.csv"

        # Define CSV columns
        columns = [
            "timestamp",
            "duration_seconds",
            "config_mode",
            "device",
            "execution_time_seconds",
            "peak_memory_mb",
            "avg_memory_mb",
            "segments_processed",
            "segments_per_second",
            "time_per_segment_ms",
            "prosody_time_ms",
            "emotion_time_ms",
            "success_count",
            "failure_count",
            "success_rate",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for result in self.results:
                config = result["config"]
                success_rate = (
                    result["success_count"] / result["segments_processed"] * 100
                    if result["segments_processed"] > 0
                    else 0
                )

                row = {
                    "timestamp": result["timestamp"],
                    "duration_seconds": config["duration_seconds"],
                    "config_mode": config["config_mode"],
                    "device": config["device"],
                    "execution_time_seconds": result["execution_time_seconds"],
                    "peak_memory_mb": result["peak_memory_mb"],
                    "avg_memory_mb": result["avg_memory_mb"],
                    "segments_processed": result["segments_processed"],
                    "segments_per_second": result["segments_per_second"],
                    "time_per_segment_ms": result["time_per_segment_ms"],
                    "prosody_time_ms": result["prosody_time_ms"] or "",
                    "emotion_time_ms": result["emotion_time_ms"] or "",
                    "success_count": result["success_count"],
                    "failure_count": result["failure_count"],
                    "success_rate": f"{success_rate:.1f}%",
                }
                writer.writerow(row)

        logger.info(f"CSV report saved to: {output_path}")
        return output_path

    def generate_markdown_report(self, output_path: Path | None = None) -> Path:
        """
        Generate markdown report with formatted tables and analysis.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to generated markdown file
        """
        if output_path is None:
            output_path = self.output_dir / f"{self.results_file.stem}.md"

        # Group results by configuration
        results_by_config = {}
        for result in self.results:
            config_key = (result["config"]["config_mode"], result["config"]["device"])
            if config_key not in results_by_config:
                results_by_config[config_key] = []
            results_by_config[config_key].append(result)

        # Generate markdown
        md_lines = []

        # Header
        md_lines.append("# Audio Enrichment Benchmark Results")
        md_lines.append("")
        md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append(f"**Total Benchmarks:** {len(self.results)}")
        md_lines.append("")

        # System information
        md_lines.append("## System Information")
        md_lines.append("")
        system_info = self.benchmark_info["system_info"]
        md_lines.append(
            f"- **CPU:** {system_info.get('cpu_count', 'N/A')} cores "
            f"({system_info.get('cpu_count_logical', 'N/A')} logical)"
        )
        md_lines.append(f"- **RAM:** {system_info.get('total_ram_gb', 0):.1f} GB")
        md_lines.append(f"- **CUDA Available:** {system_info.get('cuda_available', False)}")
        if system_info.get("cuda_device"):
            md_lines.append(f"- **GPU:** {system_info['cuda_device']}")
        md_lines.append(f"- **Python:** {system_info.get('python_version', 'N/A')}")
        md_lines.append(f"- **PyTorch:** {system_info.get('torch_version', 'N/A')}")
        md_lines.append("")

        # Summary statistics
        md_lines.append("## Summary Statistics")
        md_lines.append("")
        md_lines.extend(self._generate_summary_stats())
        md_lines.append("")

        # Detailed results by configuration
        md_lines.append("## Detailed Results by Configuration")
        md_lines.append("")

        for (config_mode, device), config_results in sorted(results_by_config.items()):
            md_lines.append(f"### {config_mode.title()} - {device.upper()}")
            md_lines.append("")
            md_lines.extend(self._generate_config_table(config_results))
            md_lines.append("")

        # Performance comparison
        md_lines.append("## Performance Comparison")
        md_lines.append("")
        md_lines.extend(self._generate_comparison_analysis())
        md_lines.append("")

        # Recommendations
        md_lines.append("## Recommendations")
        md_lines.append("")
        md_lines.extend(self._generate_recommendations())
        md_lines.append("")

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Markdown report saved to: {output_path}")
        return output_path

    def _generate_summary_stats(self) -> list[str]:
        """Generate summary statistics section."""
        lines = []

        # Overall metrics
        total_time = sum(r["execution_time_seconds"] for r in self.results)
        total_segments = sum(r["segments_processed"] for r in self.results)
        avg_segments_per_sec = np.mean([r["segments_per_second"] for r in self.results])

        lines.append(f"- **Total Processing Time:** {total_time:.1f} seconds")
        lines.append(f"- **Total Segments Processed:** {total_segments}")
        lines.append(f"- **Average Throughput:** {avg_segments_per_sec:.2f} segments/second")

        # Memory stats
        peak_memory = max(r["peak_memory_mb"] for r in self.results)
        avg_memory = np.mean([r["avg_memory_mb"] for r in self.results])

        lines.append(f"- **Peak Memory Usage:** {peak_memory:.1f} MB")
        lines.append(f"- **Average Memory Usage:** {avg_memory:.1f} MB")

        # Success rate
        total_success = sum(r["success_count"] for r in self.results)
        success_rate = (total_success / total_segments * 100) if total_segments > 0 else 0

        lines.append(f"- **Overall Success Rate:** {success_rate:.1f}%")

        return lines

    def _generate_config_table(self, results: list[dict[str, Any]]) -> list[str]:
        """Generate a table for a specific configuration."""
        lines = []

        # Sort by duration
        results = sorted(results, key=lambda r: r["config"]["duration_seconds"])

        # Table header
        lines.append(
            "| Duration | Exec Time (s) | Segments/s | Time/Segment (ms) | Memory (MB) | Success Rate |"
        )
        lines.append(
            "|----------|---------------|------------|-------------------|-------------|--------------|"
        )

        # Table rows
        for result in results:
            duration = result["config"]["duration_seconds"]
            exec_time = result["execution_time_seconds"]
            seg_per_sec = result["segments_per_second"]
            time_per_seg = result["time_per_segment_ms"]
            memory = result["peak_memory_mb"]
            success_rate = (
                result["success_count"] / result["segments_processed"] * 100
                if result["segments_processed"] > 0
                else 0
            )

            # Format duration nicely
            if duration < 60:
                dur_str = f"{int(duration)}s"
            elif duration < 3600:
                dur_str = f"{int(duration / 60)}m"
            else:
                dur_str = f"{int(duration / 3600)}h"

            lines.append(
                f"| {dur_str:8s} | {exec_time:13.2f} | {seg_per_sec:10.2f} | "
                f"{time_per_seg:17.1f} | {memory:11.1f} | {success_rate:11.1f}% |"
            )

        return lines

    def _generate_comparison_analysis(self) -> list[str]:
        """Generate performance comparison analysis."""
        lines = []

        # Compare configurations
        config_modes = {r["config"]["config_mode"] for r in self.results}

        if len(config_modes) > 1:
            lines.append("### Configuration Comparison (10s audio)")
            lines.append("")

            # Get 10s results for each config
            ten_sec_results = [r for r in self.results if r["config"]["duration_seconds"] == 10]

            if ten_sec_results:
                lines.append("| Configuration | Device | Time/Segment (ms) | Memory (MB) |")
                lines.append("|---------------|--------|-------------------|-------------|")

                for result in sorted(
                    ten_sec_results,
                    key=lambda r: (r["config"]["config_mode"], r["config"]["device"]),
                ):
                    config = result["config"]["config_mode"]
                    device = result["config"]["device"]
                    time_per_seg = result["time_per_segment_ms"]
                    memory = result["peak_memory_mb"]

                    lines.append(
                        f"| {config:13s} | {device:6s} | {time_per_seg:17.1f} | {memory:11.1f} |"
                    )

                lines.append("")

        # CPU vs GPU comparison
        cpu_results = [r for r in self.results if r["config"]["device"] == "cpu"]
        gpu_results = [r for r in self.results if r["config"]["device"] == "cuda"]

        if cpu_results and gpu_results:
            lines.append("### CPU vs GPU Performance")
            lines.append("")

            cpu_avg_time = np.mean([r["time_per_segment_ms"] for r in cpu_results])
            gpu_avg_time = np.mean([r["time_per_segment_ms"] for r in gpu_results])
            speedup = cpu_avg_time / gpu_avg_time if gpu_avg_time > 0 else 0

            lines.append(f"- **CPU Average:** {cpu_avg_time:.1f} ms/segment")
            lines.append(f"- **GPU Average:** {gpu_avg_time:.1f} ms/segment")
            lines.append(f"- **GPU Speedup:** {speedup:.2f}x")
            lines.append("")

        # Scaling analysis
        lines.append("### Scaling Analysis")
        lines.append("")

        # Group by config and device, analyze scaling
        for config_mode in sorted(config_modes):
            for device in ["cpu", "cuda"]:
                config_results = [
                    r
                    for r in self.results
                    if r["config"]["config_mode"] == config_mode and r["config"]["device"] == device
                ]

                if len(config_results) >= 2:
                    config_results = sorted(
                        config_results, key=lambda r: r["config"]["duration_seconds"]
                    )

                    # Check if time per segment is relatively constant
                    times = [r["time_per_segment_ms"] for r in config_results]
                    time_std = np.std(times)
                    time_mean = np.mean(times)
                    cv = (time_std / time_mean * 100) if time_mean > 0 else 0

                    scaling = "linear" if cv < 10 else "sublinear" if cv < 20 else "poor"

                    lines.append(
                        f"- **{config_mode} ({device}):** {scaling} scaling (CV: {cv:.1f}%)"
                    )

        return lines

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on results."""
        lines = []

        # Find best performing configuration
        best_result = min(self.results, key=lambda r: r["time_per_segment_ms"])
        best_config = best_result["config"]

        lines.append("### Optimal Configuration")
        lines.append("")
        lines.append("For best performance, use:")
        lines.append(f"- **Configuration:** {best_config['config_mode']}")
        lines.append(f"- **Device:** {best_config['device']}")
        lines.append(
            f"- **Expected Performance:** {best_result['time_per_segment_ms']:.1f} ms/segment"
        )
        lines.append("")

        # Memory recommendations
        max_memory = max(r["peak_memory_mb"] for r in self.results)
        lines.append("### Memory Requirements")
        lines.append("")
        lines.append(
            f"- **Minimum RAM Recommended:** {max_memory * 2:.0f} MB "
            f"({max_memory * 2 / 1024:.1f} GB)"
        )
        lines.append(f"- **Peak Memory Observed:** {max_memory:.1f} MB")
        lines.append("")

        # Configuration-specific recommendations
        lines.append("### Configuration-Specific Notes")
        lines.append("")

        prosody_results = [r for r in self.results if r["config"]["config_mode"] == "prosody"]
        emotion_results = [
            r
            for r in self.results
            if r["config"]["config_mode"] in ["emotion", "full", "categorical"]
        ]

        if prosody_results:
            avg_prosody_time = np.mean([r["time_per_segment_ms"] for r in prosody_results])
            lines.append(
                f"- **Prosody-only:** ~{avg_prosody_time:.1f} ms/segment "
                f"(lightweight, CPU-friendly)"
            )

        if emotion_results:
            avg_emotion_time = np.mean([r["time_per_segment_ms"] for r in emotion_results])
            lines.append(
                f"- **With Emotion Models:** ~{avg_emotion_time:.1f} ms/segment "
                f"(benefits from GPU acceleration)"
            )

        # GPU recommendation
        cpu_results = [r for r in self.results if r["config"]["device"] == "cpu"]
        gpu_results = [r for r in self.results if r["config"]["device"] == "cuda"]

        if cpu_results and gpu_results:
            cpu_avg = np.mean([r["time_per_segment_ms"] for r in cpu_results])
            gpu_avg = np.mean([r["time_per_segment_ms"] for r in gpu_results])

            if gpu_avg < cpu_avg * 0.7:  # GPU is significantly faster
                speedup = cpu_avg / gpu_avg
                lines.append("")
                lines.append(
                    f"**GPU Usage Recommended:** GPU provides {speedup:.1f}x speedup "
                    f"for emotion recognition models"
                )

        return lines

    def generate_all_reports(self):
        """Generate all report formats."""
        csv_path = self.generate_csv_report()
        md_path = self.generate_markdown_report()

        logger.info("\nReports generated:")
        logger.info(f"  CSV: {csv_path}")
        logger.info(f"  Markdown: {md_path}")

        return {"csv": csv_path, "markdown": md_path}


def generate_reports(results_file: Path) -> dict[str, Path]:
    """
    Convenience function to generate all reports.

    Args:
        results_file: Path to JSON results file

    Returns:
        Dictionary mapping report type to file path
    """
    reporter = BenchmarkReporter(results_file)
    return reporter.generate_all_reports()


# CLI for standalone usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate reports from benchmark results")
    parser.add_argument("results_file", type=Path, help="Path to JSON results file")
    parser.add_argument("--csv-only", action="store_true", help="Generate only CSV report")
    parser.add_argument(
        "--markdown-only", action="store_true", help="Generate only Markdown report"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    reporter = BenchmarkReporter(args.results_file)

    if args.csv_only:
        reporter.generate_csv_report()
    elif args.markdown_only:
        reporter.generate_markdown_report()
    else:
        reporter.generate_all_reports()
