"""
Printing utilities for visualization and reporting of model evaluation results.
"""

from colorama import Fore, Style


# Define color formatting constants
HEADER = Fore.CYAN + Style.BRIGHT  # Cyan, bold
VALUE = Fore.GREEN + Style.BRIGHT  # Green, bold
RESET = Style.RESET_ALL  # Reset to default
BOLD = Style.BRIGHT  # Bold


def _print_header(header: str, with_frame: bool = False):
    """
    Print a formatted header, optionally with a frame.

    Args:
        header: Header text to display
        with_frame: Whether to add a frame around the header
    """
    if with_frame:
        print("\n" + "=" * 80)
    print(f"\n{HEADER}{BOLD}{header}{RESET}")
    if with_frame:
        print("=" * 80)


def print_summary_statistics(metrics):
    """
    Print summary statistics to the terminal for the whole model.

    Args:
        metrics: Original metrics dictionary
    """
    # Print a header separator
    _print_header("EEVIT MODEL EVALUATION RESULTS", with_frame=True)

    # 1. Overall model statistics
    _print_header("OVERALL MODEL STATISTICS:")
    print(f"- Overall Accuracy: {VALUE}{metrics['overall_accuracy']:.2f}%{RESET}")
    print(f"- Total Samples: {VALUE}{metrics['total_samples']}{RESET}")
    print(f"- Speedup Factor: {VALUE}{metrics['speedup']:.2f}x{RESET}")
    print(f"- Computation Saved: {VALUE}{metrics['expected_saving']:.2f}%{RESET}")


def print_class_statistics(metrics, top_n=5):
    """
    Print detailed statistics about class performance.

    Args:
        metrics: Dictionary containing evaluation metrics
        top_n: Number of top/bottom classes to display
    """
    if "class_statistics" not in metrics or not metrics["class_statistics"]:
        raise ValueError("Class statistics not found in metrics file")

    class_stats = metrics["class_statistics"]

    # Print header
    _print_header("CLASS PERFORMANCE STATISTICS", with_frame=True)

    # 1. General class statistics
    _print_header("OVERALL CLASS STATISTICS:")
    print(f"- Total Classes: {VALUE}{len(class_stats)}{RESET}")
    print(f"- Overall Accuracy: {VALUE}{metrics['overall_accuracy']:.2f}%{RESET}")

    # 2. Sort classes by accuracy
    sorted_by_acc = sorted(
        class_stats.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True
    )

    # 3. Sort classes by inference time (speed)
    sorted_by_speed = sorted(
        class_stats.items(),
        key=lambda x: x[1].get("avg_inference_time_ms", float("inf")),
    )

    # 4. Print classes by accuracy (top and bottom together)
    _print_header("CLASSES BY ACCURACY:")
    print("-" * 80)
    print(
        f"{'Rank':<6} {'Class ID':<8} {'Class Name':<25} {'Accuracy':<10} {'Latency (ms)':<15} {'Mode Exit':<10}"
    )
    print("-" * 80)

    # Print top classes
    for i, (class_id, stats) in enumerate(sorted_by_acc[:top_n]):
        class_name = stats.get("name", f"Class {class_id}")
        accuracy = stats.get("accuracy", 0)
        mode_exit = stats.get("mode_exit_layer", 0)
        avg_time = stats.get("avg_inference_time_ms", 0)

        print(
            f"Top {i + 1:<3} {class_id:<8} {class_name[:25]:<25} {accuracy:>8.2f}%  {avg_time:>12.2f}   {mode_exit:>8}"
        )

    # Add separator between top and bottom
    print("-" * 80)

    # Print bottom classes
    for i, (class_id, stats) in enumerate(sorted_by_acc[-top_n:]):
        class_name = stats.get("name", f"Class {class_id}")
        accuracy = stats.get("accuracy", 0)
        mode_exit = stats.get("mode_exit_layer", 0)
        avg_time = stats.get("avg_inference_time_ms", 0)

        print(
            f"Bot {i + 1:<3} {class_id:<8} {class_name[:25]:<25} {accuracy:>8.2f}%  {avg_time:>12.2f}   {mode_exit:>8}"
        )

    # 5. Print classes by inference time (fastest and slowest together)
    _print_header("CLASSES BY INFERENCE TIME:")
    print("-" * 80)
    print(
        f"{'Rank':<6} {'Class ID':<8} {'Class Name':<25} {'Accuracy':<10} {'Latency (ms)':<15} {'Mode Exit':<10}"
    )
    print("-" * 80)

    # Print fastest classes
    for i, (class_id, stats) in enumerate(sorted_by_speed[:top_n]):
        class_name = stats.get("name", f"Class {class_id}")
        accuracy = stats.get("accuracy", 0)
        mode_exit = stats.get("mode_exit_layer", 0)
        avg_time = stats.get("avg_inference_time_ms", 0)

        print(
            f"Fast {i + 1:<2} {class_id:<8} {class_name[:25]:<25} {accuracy:>8.2f}%  {avg_time:>12.2f}   {mode_exit:>8}"
        )

    # Add separator between fastest and slowest
    print("-" * 80)

    # Print slowest classes
    for i, (class_id, stats) in enumerate(sorted_by_speed[-top_n:]):
        class_name = stats.get("name", f"Class {class_id}")
        accuracy = stats.get("accuracy", 0)
        mode_exit = stats.get("mode_exit_layer", 0)
        avg_time = stats.get("avg_inference_time_ms", 0)

        print(
            f"Slow {i + 1:<2} {class_id:<8} {class_name[:25]:<25} {accuracy:>8.2f}%  {avg_time:>12.2f}   {mode_exit:>8}"
        )

    print("\n" + "=" * 80)


def print_exit_statistics(metrics):
    """
    Print exit-related statistics to the terminal.

    Args:
        metrics: Metrics dictionary containing exit statistics
    """
    # Exit point statistics
    _print_header("EXIT POINT STATISTICS:")
    print("-" * 80)
    print(
        f"{'Exit Point':<15} {'Samples':<10} {'Percentage':<12} {'Accuracy':<12} {'Latency (ms)':<15} {'Exit Layer':<10}"
    )
    print("-" * 80)

    # Sort exit points for consistent display
    def exit_sort_key(exit_name):
        if exit_name == "final":
            return float("inf")  # Final exit should appear last
        else:
            parts = exit_name.split("_")
            return int(parts[1]) if len(parts) > 1 else float("inf")

    exit_stats = metrics["exit_statistics"]
    for exit_key in sorted(exit_stats.keys(), key=exit_sort_key):
        stats = exit_stats[exit_key]

        # Format exit name for display
        if exit_key == "final":
            exit_name = "Final Layer"
        else:
            parts = exit_key.split("_")
            exit_name = f"Exit {parts[1]}" if len(parts) > 1 else exit_key

        # Get statistics
        sample_count = stats.get("count", 0)
        percentage = stats.get("percentage_samples", 0)
        accuracy = stats.get("accuracy", 0)
        latency = stats.get("avg_inference_time_ms", 0)
        std_latency = stats.get("std_inference_time_ms", 0)

        # Print formatted row
        print(
            f"{exit_name:<15} {sample_count:<10} {percentage:>8.1f}%     {accuracy:>8.2f}%     {latency:>6.2f} ± {std_latency:<6.2f}"
        )

    print("-" * 80)


def print_sample_distribution(metrics):
    """
    Print information about sample distribution across exit points.

    Args:
        metrics: Metrics dictionary containing exit statistics
    """
    exit_stats = metrics["exit_statistics"]

    _print_header("SAMPLE DISTRIBUTION SUMMARY:")
    early_exit_samples = metrics["total_samples"] - (
        exit_stats.get("final", {}).get("count", 0)
    )
    early_exit_percentage = (
        (early_exit_samples / metrics["total_samples"]) * 100
        if metrics["total_samples"] > 0
        else 0
    )

    print(
        f"  - Early Exit Samples: {VALUE}{early_exit_samples}{RESET} ({early_exit_percentage:.1f}% of total)"
    )
    print(
        f"  - Final Layer Samples: {VALUE}{exit_stats.get('final', {}).get('count', 0)}{RESET} ({100 - early_exit_percentage:.1f}% of total)"
    )


def print_class_summary(metrics):
    """
    Print a summary of class statistics.

    Args:
        metrics: Metrics dictionary containing class statistics
    """
    if "class_statistics" not in metrics:
        return

    class_stats = metrics["class_statistics"]
    _print_header("CLASS STATISTICS SUMMARY:")
    print(f"  - Total Classes: {VALUE}{len(class_stats)}{RESET}")

    # Find classes with highest and lowest accuracy
    if not class_stats:
        raise ValueError("Class statistics not found in metrics file")

    sorted_by_acc = sorted(
        class_stats.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True
    )
    highest_acc_class = sorted_by_acc[0]
    lowest_acc_class = sorted_by_acc[-1]

    print(
        f"  - Highest Accuracy Class: {VALUE}{highest_acc_class[1]['name']}{RESET} ({highest_acc_class[1]['accuracy']:.2f}%)"
    )
    print(
        f"  - Lowest Accuracy Class: {VALUE}{lowest_acc_class[1]['name']}{RESET} ({lowest_acc_class[1]['accuracy']:.2f}%)"
    )

    print("\n" + "=" * 80)


def print_performance_summary(metrics):
    """
    Print a summary of performance metrics.

    Args:
        metrics: Metrics dictionary containing performance statistics
    """
    _print_header("PERFORMANCE SUMMARY:")
    print(
        f"  - Average Inference Time: {VALUE}{metrics.get('avg_inference_time_ms', 'N/A'):.2f} ms{RESET}"
    )

    print("\n" + "=" * 80)


def print_overall_statistics(metrics):
    """
    Print overall statistics to the terminal for the whole model.

    Args:
        metrics: Original metrics dictionary
    """
    # Print a header separator
    _print_header("EEVIT MODEL EVALUATION RESULTS", with_frame=True)

    # 1. Overall model statistics
    _print_header("OVERALL MODEL STATISTICS:")
    print(f"- Overall Accuracy: {VALUE}{metrics['overall_accuracy']:.2f}%{RESET}")
    print(f"- Total Samples: {VALUE}{metrics['total_samples']}{RESET}")
    print(f"- Speedup Factor: {VALUE}{metrics['speedup']:.2f}x{RESET}")
    print(f"- Computation Saved: {VALUE}{metrics['expected_saving']:.2f}%{RESET}")


def print_statistics(metrics, detailed: bool = True, top_n: int = 5):
    """
    Print detailed statistics to the terminal for each exit and the whole model.

    Args:
        metrics: Original metrics dictionary
    """

    print_overall_statistics(metrics)

    print_class_summary(metrics)

    print_performance_summary(metrics)

    if detailed:
        print_class_statistics(metrics, top_n)

        print_exit_statistics(metrics)

        print_sample_distribution(metrics)
