import json
import csv

# Input and output file paths
input_file = "mini_data2.txt"  # Change this to your actual input file path
output_file = "eye_tracking_data.csv"

# Open input file and output CSV file
with open(input_file, "r") as infile, open(output_file, "a", newline="") as csvfile:
    fieldnames = [
        "timestamp", "time", "fixation",
        "avg_x", "avg_y",
        "left_avg_x", "left_avg_y", "left_pcenter_x", "left_pcenter_y", "left_psize",
        "right_avg_x", "right_avg_y", "right_pcenter_x", "right_pcenter_y", "right_psize"
    ]
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for line in infile:
        try:
            data = json.loads(line.strip())  # Load JSON from each line
            if data.get("category") == "heartbeat":  # Ignore heartbeat entries
                continue
            
            values = data.get("values", {})
            frame = values.get("frame", {})

            row = {
                "timestamp": frame.get("timestamp", ""),
                "time": frame.get("time", ""),
                "fixation": frame.get("fix", ""),
                "avg_x": frame.get("avg", {}).get("x", ""),
                "avg_y": frame.get("avg", {}).get("y", ""),
                "left_avg_x": frame.get("lefteye", {}).get("avg", {}).get("x", ""),
                "left_avg_y": frame.get("lefteye", {}).get("avg", {}).get("y", ""),
                "left_pcenter_x": frame.get("lefteye", {}).get("pcenter", {}).get("x", ""),
                "left_pcenter_y": frame.get("lefteye", {}).get("pcenter", {}).get("y", ""),
                "left_psize": frame.get("lefteye", {}).get("psize", ""),
                "right_avg_x": frame.get("righteye", {}).get("avg", {}).get("x", ""),
                "right_avg_y": frame.get("righteye", {}).get("avg", {}).get("y", ""),
                "right_pcenter_x": frame.get("righteye", {}).get("pcenter", {}).get("x", ""),
                "right_pcenter_y": frame.get("righteye", {}).get("pcenter", {}).get("y", ""),
                "right_psize": frame.get("righteye", {}).get("psize", ""),
            }

            writer.writerow(row)
        
        except json.JSONDecodeError:
            print("Skipping invalid JSON line:", line.strip())

print(f"CSV file '{output_file}' generated successfully!")
