import json
import os
import re
import argparse
from collections import defaultdict
import statistics

"""
This script processes PyTorch profiler trace files to calculate the average attention layer 
durations in milliseconds and saves the results to a JSON file. The script groups attention layers 
into predefined groups and computes the average duration for each group across multiple trace files.

The script calculates both individual file statistics and the average across all runs 
for each layer group, saving everything to a single consolidated JSON file.

Usage:
    Run the script from the command line with the following options:
        -d, --directory: Directory containing profiler trace files (default: current directory).
        -O, --output: Output JSON file name (default: attention_statistics.json).
"""


# Define layer groups for averaging
LAYER_GROUPS = {
    "Layers_0_2": range(0, 3),
    "Layers_3_6": range(3, 7),
    "Layers_7_10": range(7, 11),
}


def extract_attention_durations(file_path):
    """Extract 'dur' values from trace events where 'name' matches 'nn.Module: Attention_x'."""
    with open(file_path, "r") as f:
        data = json.load(f)

    durations = defaultdict(list)

    for event in data.get("traceEvents", []):
        name = event.get("name", "")
        dur = event.get("dur", None)

        # Match "nn.Module: Attention_x" where x is a number
        match = re.match(r"nn\.Module: Attention_(\d+)", name)
        if match and dur is not None:
            layer = int(match.group(1))  # Extract layer number

            # Assign to the correct group
            for group_name, layer_range in LAYER_GROUPS.items():
                if layer in layer_range:
                    durations[group_name].append(dur)

    return durations


def compute_averages(trace_files):
    """Compute average durations for each layer group across multiple trace files."""
    all_averages = []
    # Track all durations for each group across all files for overall averages
    all_group_durations = defaultdict(list)

    for file in trace_files:
        durations = extract_attention_durations(file)

        averages = {
            group: (sum(values) / len(values)) if values else 0
            for group, values in durations.items()
        }

        # from microseconds to milliseconds and round to 3 decimal places
        averages = {group: round(avg / 1000, 3) for group, avg in averages.items()}

        # Store results with filename reference
        result = {"trace_file": os.path.basename(file)}
        result.update(averages)
        all_averages.append(result)

        # Collect all durations for each group for overall statistics
        for group, values in durations.items():
            # Convert microseconds to milliseconds
            all_group_durations[group].extend([v / 1000 for v in values])

    # Calculate overall averages and standard deviations across all files
    overall_stats = {}
    for group, durations in all_group_durations.items():
        if durations:
            overall_stats[group] = {
                "avg_ms": round(statistics.mean(durations), 3),
                "std_dev_ms": round(statistics.stdev(durations), 3)
                if len(durations) > 1
                else 0,
                "min_ms": round(min(durations), 3),
                "max_ms": round(max(durations), 3),
                "count": len(durations),
            }
        else:
            overall_stats[group] = {
                "avg_ms": 0,
                "std_dev_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "count": 0,
            }

    return all_averages, overall_stats


def save_to_json(file_results, overall_stats, output_file):
    """Save both individual file results and overall statistics to a single JSON file."""
    # Create consolidated data structure
    consolidated_data = {
        "individual_runs": file_results,
        "overall_statistics": overall_stats,
    }

    with open(output_file, "w") as f:
        json.dump(consolidated_data, f, indent=4)


def print_overall_statistics(overall_stats):
    """Print the overall statistics in a readable format."""
    print("\nOverall Statistics Across All Runs:")
    print("-" * 80)
    print(
        f"{'Layer Group':<15} {'Average (ms)':<15} {'Std Dev (ms)':<15} {'Min (ms)':<15} {'Max (ms)':<15} {'Sample Count'}"
    )
    print("-" * 80)

    for group, stats in overall_stats.items():
        print(
            f"{group:<15} {stats['avg_ms']:<15.3f} {stats['std_dev_ms']:<15.3f} {stats['min_ms']:<15.3f} {stats['max_ms']:<15.3f} {stats['count']}"
        )


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Process PyTorch profiler trace files and calculate attention layer durations."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        help="Directory containing profiler trace files (default: current directory)",
    )
    parser.add_argument(
        "-O",
        "--output",
        type=str,
        default="attention_statistics.json",
        help="Output JSON file name (default: attention_statistics.json)",
    )
    args = parser.parse_args()

    # Create full path for the output file
    output_json_path = os.path.join(args.directory, args.output)

    # Find all 'pytorch_profiler_trace_x.json' files in the specified directory
    trace_files = sorted(
        [
            os.path.join(args.directory, f)
            for f in os.listdir(args.directory)
            if f.startswith("pytorch_profiler_trace_") and f.endswith(".json")
        ]
    )

    if not trace_files:
        print(f"No trace files found in {args.directory}")
        return

    print(f"Processing {len(trace_files)} trace files from {args.directory}...")
    results, overall_stats = compute_averages(trace_files)

    # Save both individual file averages and overall statistics to a single JSON file
    save_to_json(results, overall_stats, output_json_path)
    print(f"Results saved to {output_json_path}")

    # Print overall statistics
    print_overall_statistics(overall_stats)


if __name__ == "__main__":
    main()
