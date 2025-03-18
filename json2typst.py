#!/usr/bin/env python3
"""
JSON to Typst Tables Converter

This script converts attention statistics JSON to a Typst file containing formatted tables.
It creates two tables: one for individual layer statistics and one for layer group statistics.

Usage:
    python json_to_typst.py [-d DIRECTORY] [-f FILENAME]

Arguments:
    -d, --directory  Directory containing the JSON file (default: current directory)
    -f, --filename   Input JSON filename (default: attention_statistics.json)
"""

import json
import argparse
import os
from datetime import datetime


def format_value(value, precision=3):
    """Format numeric values with consistent precision."""
    if isinstance(value, (int, float)):
        if isinstance(value, int):
            return str(value)
        else:
            return f"{value:.{precision}f}"
    return str(value)


def create_typst_table(data, title, columns, column_headers):
    """
    Create a Typst table from data.

    Args:
        data: List of dictionaries containing the data
        title: Title for the table
        columns: Number of columns in the table
        column_headers: List of header names

    Returns:
        Formatted Typst table as a string
    """
    # Start with table definition and headers
    typst_table = "#figure(\n"
    typst_table += "  table(\n"
    typst_table += f"    columns: {columns},\n"

    # Add headers
    header_str = "    table.header("
    for header in column_headers:
        header_str += f"[{header}], "
    header_str = header_str.rstrip(", ") + "),\n"
    typst_table += header_str

    # Add data rows
    for row in data:
        row_str = "    "
        for value in row:
            row_str += f"[{value}], "
        typst_table += row_str.rstrip(", ") + ",\n"

    # Close the table and add caption
    typst_table += "  ),\n"
    typst_table += f"  caption: [{title}],\n"
    typst_table += ")\n\n"

    return typst_table


def json_to_typst(input_file, output_file):
    """
    Convert JSON statistics to Typst tables.

    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output Typst file
    """
    try:
        # Load the JSON data
        with open(input_file, "r") as f:
            stats = json.load(f)

        # Extract layer statistics
        individual_layer_stats = stats.get("overall_statistics", {}).get(
            "individual_layers", {}
        )
        group_stats = stats.get("overall_statistics", {}).get("layer_groups", {})

        # Get timestamp from one of the trace files if available
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create Typst document
        typst_content = "#align(center)[= Attention Layer Statistics]\n\n"
        typst_content += f"Generated on {timestamp}\n\n"

        # Section for individual layer statistics
        typst_content += "Individual Layer Statistics\n\n"

        # Prepare data for individual layers table
        layer_data = []
        column_headers = [
            "Layer",
            "Avg (ms)",
            "Std Dev",
            "Min (ms)",
            "Max (ms)",
            "Count",
        ]

        # Sort layers by layer number
        sorted_layers = sorted(
            individual_layer_stats.keys(),
            key=lambda x: int(x.split("_")[1]) if "_" in x else 0,
        )

        for layer in sorted_layers:
            stats = individual_layer_stats[layer]
            layer_data.append(
                [
                    layer,
                    format_value(stats.get("avg_ms", 0)),
                    format_value(stats.get("std_dev_ms", 0)),
                    format_value(stats.get("min_ms", 0)),
                    format_value(stats.get("max_ms", 0)),
                    str(stats.get("count", 0)),
                ]
            )

        # Create individual layers table
        individual_layers_table = create_typst_table(
            layer_data, "Attention Latency by Individual Layer", 6, column_headers
        )
        typst_content += individual_layers_table

        # Section for layer group statistics
        typst_content += "Layer Group Statistics\n\n"

        # Prepare data for layer groups table
        group_data = []

        # Sort groups in a logical order
        sorted_groups = sorted(group_stats.keys())

        for group in sorted_groups:
            stats = group_stats[group]
            group_data.append(
                [
                    group,
                    format_value(stats.get("avg_ms", 0)),
                    format_value(stats.get("std_dev_ms", 0)),
                    format_value(stats.get("min_ms", 0)),
                    format_value(stats.get("max_ms", 0)),
                    str(stats.get("count", 0)),
                ]
            )

        # Create layer groups table
        groups_table = create_typst_table(
            group_data, "Attention Latency by Layer Group", 6, column_headers
        )
        typst_content += groups_table

        # Add metadata about the source
        typst_content += f"Source: `{os.path.basename(input_file)}`\n"

        # Write to output file
        with open(output_file, "w") as f:
            f.write(typst_content)

        print(f"Successfully converted {input_file} to {output_file}")
        return True

    except Exception as e:
        print(f"Error converting JSON to Typst: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert attention statistics JSON to Typst tables"
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        help="Directory containing the JSON file (default: current directory)",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="attention_statistics.json",
        help="Input JSON filename (default: attention_statistics.json)",
    )

    args = parser.parse_args()

    # Construct input file path
    input_file = os.path.join(args.directory, args.filename)

    # Create output file path in the same directory
    input_base = os.path.splitext(args.filename)[0]
    output_file = os.path.join(args.directory, f"{input_base}.typ")

    # Verify input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return

    json_to_typst(input_file, output_file)


if __name__ == "__main__":
    main()
