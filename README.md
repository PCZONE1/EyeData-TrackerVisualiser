# Eye Tracking Visualization Tool

A Flask-based web application for generating interactive visualizations from eye tracking data. This tool processes CSV data from eye trackers and generates fixation maps and pupil size heatmaps, with optional background image support.

## Features

- **CSV Data Processing**: Upload and process eye tracking data in CSV format
- **Background Image Support**: Add context to your visualizations by overlaying eye tracking data on images
- **Fixation Map**: Visualize where users focused their attention, with circle size indicating fixation duration
- **Pupil Size Heatmap**: Analyze cognitive load through pupil dilation visualization
- **Interactive UI**: User-friendly interface with drag-and-drop file uploads and previews
- **Instant Downloads**: Generated visualizations are immediately available for download

## Demo

Here's what the visualizations look like:

| Fixation Map | Pupil Size Heatmap |
|--------------|-------------------|
| ![Fixation Map](docs/fixation_map.png) | ![Pupil Heatmap](docs/pupil_size_heatmap.png) |

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/aliladakhi/eye-tracking-visualization.git
   cd eye-tracking-visualization
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create required directories:
   ```bash
   mkdir -p uploads static
   ```

## Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:10000
   ```

3. Upload your eye tracking CSV data and an optional background image

4. View and download the generated visualizations

## CSV Data Format

Your eye tracking CSV should include the following columns:

- `avg_x`: X-coordinate of gaze point (required)
- `avg_y`: Y-coordinate of gaze point (required)
- `fixation`: Boolean (TRUE/FALSE) indicating if the point is a fixation
- `left_psize`: Left pupil size (required for pupil heatmap)
- `right_psize`: Right pupil size (required for pupil heatmap)
- `time`: Timestamp or duration information

Example CSV format:
```
avg_x,avg_y,fixation,left_psize,right_psize,time
512,384,TRUE,3.2,3.1,1000
640,400,TRUE,3.5,3.4,1002
...
```

## Data Processing Details

The application performs the following processing on eye tracking data:

1. **Coordinate Transformation**: The application rotates coordinates 90 degrees clockwise to match standard eye tracker orientation
2. **Spatial Clustering**: DBSCAN clustering algorithm groups nearby fixation points
3. **Pupil Size Averaging**: The mean of left and right pupil sizes is calculated for cognitive load analysis
4. **Temporal Sequencing**: Fixation points are color-coded to show the viewing sequence

## Background Image Requirements

- Supported formats: JPG, PNG, JPEG
- Images will be automatically resized to match visualization dimensions (1920x1080)
- For best results, use images with the same aspect ratio as your eye tracking screen (typically 16:9)

## Customization

You can modify the visualization parameters in `app.py`:

- `distance_threshold`: Adjust the clustering distance for fixation points
- Visualization colors and transparency can be modified in the plotting functions
- Plot dimensions can be adjusted to match your specific screen resolution

## Troubleshooting

- **File Upload Issues**: Ensure your CSV has the correct column names and format
- **Image Not Appearing**: Verify the image format is supported (JPG, PNG, JPEG)
- **Plot Generation Error**: Check the CSV for missing or non-numeric data
- **Server Error**: Look for error messages in the console output

