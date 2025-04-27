UPLOAD_FOLDER = 'uploads'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from flask import Flask, render_template, request, redirect, url_for
import os
import time
import uuid



def clean_old_files(folder, age_seconds=3600):
    """
    Delete files older than age_seconds from the given folder.
    Default is 1 hour (3600 seconds).
    """
    now = time.time()
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            if now - os.path.getmtime(filepath) > age_seconds:
                os.remove(filepath)


def create_fixation_map(csv_file, distance_threshold=50):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert Fixation Column (TRUE/FALSE → 1/0)
    df["fixation"] = df["fixation"].astype(bool).astype(int)
    
    # Convert Coordinates to Numeric
    df["avg_x"] = pd.to_numeric(df["avg_x"], errors="coerce").fillna(0)
    df["avg_y"] = pd.to_numeric(df["avg_y"], errors="coerce").fillna(0)
    
    # Rotate coordinates 90 degrees clockwise and maintain scale
    # New x = y
    # New y = 1080 - x (to keep points within bounds and correct orientation)
    df["rotated_x"] = df["avg_y"]
    df["rotated_y"] = 1080 - df["avg_x"]
    
    # Filter Only Fixation Points (Fixation == 1)
    fixation_data = df[df["fixation"] == 1]
    
    # Perform spatial clustering using DBSCAN
    coordinates = np.column_stack((fixation_data['rotated_x'], fixation_data['rotated_y']))
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(coordinates)
    fixation_data['cluster'] = clustering.labels_
    
    # Aggregate data by cluster
    clustered_data = fixation_data.groupby('cluster').agg({
        'rotated_x': 'mean',
        'rotated_y': 'mean',
        'time': 'count'  # Count number of points in each cluster
    }).reset_index()
    
    # Create figure and axis with white background
    # Convert inches to pixels (assuming 100 DPI)
    width_inches = 1920/100
    height_inches = 1080/100
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), facecolor='white')
    ax.set_facecolor('white')
    
    # Set the plot limits to match the screen resolution
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    
    # Plot lines between consecutive clusters
    for i in range(len(clustered_data) - 1):
        ax.plot([clustered_data['rotated_x'].iloc[i], clustered_data['rotated_x'].iloc[i + 1]], 
               [clustered_data['rotated_y'].iloc[i], clustered_data['rotated_y'].iloc[i + 1]], 
               color='blue', alpha=0.3, linewidth=1)
    
    # Calculate sizes based on number of points in each cluster
    max_points = clustered_data['time'].max()
    min_points = clustered_data['time'].min()
    
    # Adjust the size range to make differences more visible
    # Changed from (50, 500) to (100, 2000) for more dramatic size differences
    min_size = 100
    max_size = 2000
    
    # Scale sizes with a logarithmic transformation to better show differences
    sizes = min_size + ((np.log1p(clustered_data['time']) - np.log1p(min_points)) / 
                       (np.log1p(max_points) - np.log1p(min_points))) * (max_size - min_size)
    
    # Plot circles for fixation points
    scatter = ax.scatter(clustered_data['rotated_x'], 
                        clustered_data['rotated_y'], 
                        s=sizes,  # Using our new sizes
                        alpha=0.5, 
                        c=range(len(clustered_data)),  # Color based on sequence
                        cmap='viridis')
    
    # Add a small colorbar to show temporal sequence
    cbar = plt.colorbar(scatter, label='Fixation Sequence', shrink=0.6)
    cbar.ax.tick_params(labelsize=8)
    
    # Remove axes, title, and labels
    ax.set_axis_off()
    
    # Make sure the figure is tight
    plt.tight_layout()
    
    # Save the plot without any borders or padding
    filename = str(uuid.uuid4())
    plt.savefig(f'static/{filename}_fixation_map.png', dpi=300, bbox_inches='tight', pad_inches=0, 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def create_pupil_heatmap(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert Fixation Column (TRUE/FALSE → 1/0)
    df["fixation"] = df["fixation"].astype(bool).astype(int)
    
    # Convert Coordinates to Numeric
    df["avg_x"] = pd.to_numeric(df["avg_x"], errors="coerce").fillna(0)
    df["avg_y"] = pd.to_numeric(df["avg_y"], errors="coerce").fillna(0)
    
    # Convert pupil sizes to numeric and calculate average
    df['right_psize'] = pd.to_numeric(df['right_psize'], errors="coerce").fillna(0)
    df['left_psize'] = pd.to_numeric(df['left_psize'], errors="coerce").fillna(0)
    df['avg_pupil_size'] = (df['right_psize'] + df['left_psize']) / 2
    
    # Rotate coordinates 90 degrees clockwise and maintain scale
    df["rotated_x"] = df["avg_y"]
    df["rotated_y"] = 1080 - df["avg_x"]
    
    # Filter Only Fixation Points (Fixation == 1)
    fixation_data = df[df["fixation"] == 1]
    
    # Create figure and axis with white background
    # Convert inches to pixels (assuming 100 DPI)
    width_inches = 1920/100
    height_inches = 1080/100
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), facecolor='white')
    ax.set_facecolor('white')
    
    # Set the plot limits to match the screen resolution
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    
    # Plot heatmap points
    scatter = ax.scatter(fixation_data['rotated_x'], 
                        fixation_data['rotated_y'], 
                        s=100,  # Fixed size for all points
                        alpha=0.6, 
                        c=fixation_data['avg_pupil_size'],  # Color based on average pupil size
                        cmap='YlOrRd',  # Yellow-Orange-Red colormap
                        norm=plt.Normalize(vmin=fixation_data['avg_pupil_size'].min(), 
                                        vmax=fixation_data['avg_pupil_size'].max()))
    
    # Add colorbar to show pupil size scale
    cbar = plt.colorbar(scatter, label='Average Pupil Size')
    cbar.ax.tick_params(labelsize=8)
    
    # Remove axes
    ax.set_axis_off()
    
    # Make sure the figure is tight
    plt.tight_layout()
    
    # Save the plot
    filename = str(uuid.uuid4())
    plt.savefig(f'static/{filename}_pupil_size_heatmap.png', dpi=300, bbox_inches='tight', 
                pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in request', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if not file.filename.lower().endswith('.csv'):
        return 'Only CSV files are allowed', 400

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Run plotting
        create_fixation_map(filepath, distance_threshold=25)
        create_pupil_heatmap(filepath)
        
        # Delete uploaded CSV after plots are created
        os.remove(filepath)
        clean_old_files('static', age_seconds=3600)  # Clean static folder (files older than 1 hour)
        return redirect(url_for('download_page'))
    except Exception as e:
        return f"Something went wrong: {str(e)}", 500


@app.route('/download')
def download_page():
    return render_template('download.html')


if __name__ == '__main__':
    app.run(debug=True)
