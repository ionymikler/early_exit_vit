import json
import os
import re
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics
import numpy as np

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
        --plot: Enable timeline plotting for a specific layer.
"""


# Define layer groups for averaging
LAYER_GROUPS = {
    "No early-exit": list(range(0, 3)) + [11],  # Include the last layer as well
    "LPH": range(3, 7),
    "GAH": range(7, 11),
}

# Define individual layers for detailed analysis
INDIVIDUAL_LAYERS = {f"Layer_{i}": [i] for i in range(12)}


def extract_attention_durations(file_path):
    """Extract 'dur' values from trace events where 'name' matches 'nn.Module: Attention_x'."""
    with open(file_path, "r") as f:
        data = json.load(f)

    durations = defaultdict(list)
    individual_layer_durations = defaultdict(list)

    # Also extract timestamps for timeline plotting
    timestamps = defaultdict(list)

    for event in data.get("traceEvents", []):
        name = event.get("name", "")
        dur = event.get("dur", None)
        ts = event.get("ts", None)  # Get timestamp

        # Match "nn.Module: Attention_x" where x is a number
        match = re.match(r"nn\.Module: Attention_(\d+)", name)
        if match and dur is not None:
            layer = int(match.group(1))  # Extract layer number

            # For individual layer statistics
            layer_key = f"Layer_{layer}"
            individual_layer_durations[layer_key].append(dur)

            # Store timestamp with duration for timeline plotting
            if ts is not None:
                timestamps[layer_key].append((ts, dur))

            # Assign to the correct group
            for group_name, layer_range in LAYER_GROUPS.items():
                if layer in layer_range:
                    durations[group_name].append(dur)

    # Merge both dictionaries
    durations.update(individual_layer_durations)
    return durations, timestamps


def compute_averages(trace_files):
    """Compute average durations for each layer group across multiple trace files."""
    all_averages = []
    # Track all durations for each group across all files for overall averages
    all_group_durations = defaultdict(list)

    # Store timestamps for timeline plotting
    all_timestamps = defaultdict(list)
    trace_file_numbers = {}  # To track the run number for each file

    for i, file in enumerate(trace_files):
        durations, timestamps = extract_attention_durations(file)

        # Store run number for this file
        file_basename = os.path.basename(file)
        # Try to extract run number from filename (e.g., pytorch_profiler_trace_1.json -> 1)
        match = re.search(r"_(\d+)\.json$", file_basename)
        run_number = int(match.group(1)) if match else i + 1
        trace_file_numbers[file_basename] = run_number

        averages = {
            group: (sum(values) / len(values)) if values else 0
            for group, values in durations.items()
        }

        # from microseconds to milliseconds and round to 3 decimal places
        averages = {group: round(avg / 1000, 3) for group, avg in averages.items()}

        # Store results with filename reference
        result = {"trace_file": file_basename, "run_number": run_number}
        result.update(averages)
        all_averages.append(result)

        # Collect all durations for each group for overall statistics
        for group, values in durations.items():
            # Convert microseconds to milliseconds
            all_group_durations[group].extend([v / 1000 for v in values])

        # Store timestamps with run number for timeline plotting
        for layer, ts_list in timestamps.items():
            for ts, dur in ts_list:
                all_timestamps[layer].append(
                    (run_number, ts, dur / 1000)
                )  # Convert dur to ms

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

    return all_averages, overall_stats, all_timestamps


def save_to_json(file_results, overall_stats, output_file):
    """Save both individual file results and overall statistics to a single JSON file."""
    # Organize the overall statistics section into individual layers and layer groups
    organized_stats = {"individual_layers": {}, "layer_groups": {}}

    # Sort statistics into the appropriate categories
    for key, value in overall_stats.items():
        if key.startswith("Layer_"):
            organized_stats["individual_layers"][key] = value
        elif key.startswith("Layers_"):
            organized_stats["layer_groups"][key] = value

    # Create consolidated data structure
    consolidated_data = {
        "individual_runs": file_results,
        "overall_statistics": organized_stats,
    }

    with open(output_file, "w") as f:
        json.dump(consolidated_data, f, indent=4)


def print_overall_statistics(overall_stats):
    """Print the overall statistics in a readable format."""
    print("\nOverall Statistics Across All Runs:")
    print("-" * 95)
    print(
        f"{'Layer':<15} {'Average (ms)':<15} {'Std Dev (ms)':<15} {'Min (ms)':<15} {'Max (ms)':<15} {'Sample Count':<15}"
    )
    print("-" * 95)

    # First print individual layer statistics (sorted by layer number)
    individual_layers = sorted(
        [(k, v) for k, v in overall_stats.items() if k.startswith("Layer_")],
        key=lambda x: int(x[0].split("_")[1]),
    )

    for layer_name, stats in individual_layers:
        print(
            f"{layer_name:<15} {stats['avg_ms']:<15.3f} {stats['std_dev_ms']:<15.3f} {stats['min_ms']:<15.3f} {stats['max_ms']:<15.3f} {stats['count']:<15}"
        )

    # Print a separator
    print("-" * 95)

    # Then print layer group statistics
    group_stats = [(k, v) for k, v in overall_stats.items() if k.startswith("Layers_")]
    for group_name, stats in group_stats:
        print(
            f"{group_name:<15} {stats['avg_ms']:<15.3f} {stats['std_dev_ms']:<15.3f} {stats['min_ms']:<15.3f} {stats['max_ms']:<15.3f} {stats['count']:<15}"
        )


def prompt_for_layer(all_timestamps):
    """Prompt the user to select one or multiple layers for timeline plotting."""
    # Get all available layers
    layers = [layer for layer in all_timestamps.keys() if layer.startswith("Layer_")]
    layers.sort(key=lambda x: int(x.split("_")[1]))

    print("\nAvailable layers for timeline plotting:")
    for i, layer in enumerate(layers):
        print(f"{i}. {layer}")

    while True:
        choice = input(
            "\nSelect layers (options: layer numbers/names separated by spaces, 'all' for all layers): "
        )

        if choice.lower() == "all":
            return layers

        # Split input by spaces to handle multiple selections
        selections = choice.split()
        selected_layers = []

        for selection in selections:
            # Try to interpret as a number (index)
            try:
                idx = int(selection)
                if 0 <= idx < len(layers):
                    selected_layers.append(layers[idx])
                else:
                    print(
                        f"Invalid layer number: {idx}. Please enter numbers between 0 and {len(layers)-1}"
                    )
                    break
            except ValueError:
                # Try to interpret as a layer name
                if selection in layers:
                    selected_layers.append(selection)
                # Try to interpret as "Layer_X"
                elif selection.isdigit() and f"Layer_{selection}" in layers:
                    selected_layers.append(f"Layer_{selection}")
                else:
                    print(
                        f"Layer '{selection}' not found. Please select from the list."
                    )
                    break

        # If we successfully processed all selections, return the result
        if len(selected_layers) == len(selections) and selected_layers:
            return selected_layers

        # If we got here, there was an error in the selections - try again


def plot_layer_timeline(all_timestamps, layers_to_plot, output_dir=None):
    """Create a timeline plot showing latency over different runs for selected layers."""
    plt.figure(figsize=(12, 8))

    # Define a colormap for multiple layers
    colors = plt.cm.tab10.colors

    for i, layer in enumerate(layers_to_plot):
        # Extract data for this layer
        if layer not in all_timestamps:
            print(f"Warning: No data found for {layer}")
            continue

        data = all_timestamps[layer]
        if not data:
            print(f"Warning: Empty data for {layer}")
            continue

        # Sort by run number
        data.sort(key=lambda x: x[0])

        # Extract run numbers and durations
        runs = [item[0] for item in data]
        durations = [item[2] for item in data]  # Already in ms

        # Plot line
        color = colors[i % len(colors)]
        plt.plot(
            runs, durations, "o-", label=layer, color=color, markersize=8, linewidth=2
        )

        # Calculate and plot the trend line
        if len(runs) > 1:
            z = np.polyfit(runs, durations, 1)
            p = np.poly1d(z)
            plt.plot(runs, p(runs), "--", color=color, alpha=0.7)

    plt.xlabel("Run Number", fontsize=14)
    plt.ylabel("Latency (ms)", fontsize=14)
    plt.title("Layer Latency Timeline Across Runs", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Optimize x-axis to show integer run numbers
    ax = plt.gca()
    try:
        # Find min and max run numbers across all selected layers
        all_runs = []
        for layer in layers_to_plot:
            if layer in all_timestamps and all_timestamps[layer]:
                all_runs.extend([item[0] for item in all_timestamps[layer]])

        if all_runs:
            min_run = min(all_runs)
            max_run = max(all_runs)
            ax.set_xticks(range(min_run, max_run + 1))
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not optimize x-axis ticks: {e}")

    plt.tight_layout()

    # Ask if the user wants to save the figure
    save_fig = False
    if output_dir is not None:
        save_choice = input(
            "\nWould you like to save the timeline figure? (y/n): "
        ).lower()
        save_fig = save_choice.startswith("y")

    if save_fig:
        # Create a descriptive filename
        if len(layers_to_plot) == 1:
            filename = f"{layers_to_plot[0]}_timeline.png"
        else:
            layers_str = "".join([layer.split("_")[1] for layer in layers_to_plot])
            filename = f"layers_{layers_str}_timeline.png"

        # Save the figure
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()


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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable timeline plotting for specific layers",
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
    results, overall_stats, all_timestamps = compute_averages(trace_files)

    # Save both individual file averages and overall statistics to a single JSON file
    save_to_json(results, overall_stats, output_json_path)
    print(f"Results saved to {output_json_path}")

    # Print overall statistics
    print_overall_statistics(overall_stats)

    # Plot timeline if requested
    if args.plot and all_timestamps:
        layers_to_plot = prompt_for_layer(all_timestamps)
        if layers_to_plot:
            plot_layer_timeline(all_timestamps, layers_to_plot, args.directory)


if __name__ == "__main__":
    main()
