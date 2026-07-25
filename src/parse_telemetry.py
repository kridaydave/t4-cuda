#!/usr/bin/env python3
import csv
import json
import os
import sys


def parse_telemetry(csv_path: str, output_json_path: str) -> None:
    if not os.path.exists(csv_path):
        print(f"[Telemetry Parser] Error: '{csv_path}' not found.")
        sys.exit(1)

    clocks = []
    powers = []
    temps = []
    throttle_events = {}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("[Telemetry Parser] Error: CSV file is empty.")
            sys.exit(1)

        for row in reader:
            if len(row) < 4:
                continue
            try:
                sm_clk = float(row[0].replace("MHz", "").strip())
                power = float(row[1].replace("W", "").strip())
                temp = float(row[2].replace("C", "").strip())
                throttle = row[3].strip()
            except (ValueError, IndexError):
                continue

            clocks.append(sm_clk)
            powers.append(power)
            temps.append(temp)

            if throttle and throttle not in ("0x0000000000000000", "N/A", "[N/A]", ""):
                throttle_events[throttle] = throttle_events.get(throttle, 0) + 1

    if not clocks:
        print("[Telemetry Parser] No valid telemetry samples found.")
        sys.exit(1)

    summary = {
        "total_samples": len(clocks),
        "sm_clock_mhz": {
            "avg": round(sum(clocks) / len(clocks), 2),
            "max": max(clocks),
            "min": min(clocks),
        },
        "power_draw_watts": {
            "avg": round(sum(powers) / len(powers), 2),
            "max": max(powers),
            "min": min(powers),
        },
        "temperature_celsius": {
            "avg": round(sum(temps) / len(temps), 2),
            "max": max(temps),
            "min": min(temps),
        },
        "active_throttle_reasons": throttle_events,
    }

    print("=" * 74)
    print("  NVIDIA Tesla T4 Hardware Telemetry Summary Report")
    print("=" * 74)
    print(f"  Total Samples          : {summary['total_samples']}")
    print(f"  SM Boost Clock (MHz)   : Avg = {summary['sm_clock_mhz']['avg']}  "
          f"Max = {summary['sm_clock_mhz']['max']}")
    print(f"  Power Consumption (W)  : Avg = {summary['power_draw_watts']['avg']}  "
          f"Max = {summary['power_draw_watts']['max']}  (TDP: 70W)")
    print(f"  GPU Temperature (C)    : Avg = {summary['temperature_celsius']['avg']}  "
          f"Max = {summary['temperature_celsius']['max']}")
    print(f"  Active Throttle Flags  : {summary['active_throttle_reasons']}")
    print("=" * 74)

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f">> Saved summary to '{output_json_path}'.")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/t4_telemetry.csv"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "telemetry_summary.json"
    parse_telemetry(csv_file, out_json)
