UPLOAD_FOLDER = 'uploads'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
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


def create_fixation_map(csv_file, background_image=None, distance_threshold=50):
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
    
    # Add background image if provided
    if background_image:
        try:
            img = Image.open(background_image)
            # Resize image to match 1920x1080 dimensions
            img = img.resize((1920, 1080), Image.LANCZOS)
            # Add the background image to the plot
            ax.imshow(np.array(img), extent=[0, 1920, 0, 1080])
        except Exception as e:
            print(f"Error loading background image: {str(e)}")
    
    # Plot lines between consecutive clusters
    for i in range(len(clustered_data) - 1):
        ax.plot([clustered_data['rotated_x'].iloc[i], clustered_data['rotated_x'].iloc[i + 1]], 
               [clustered_data['rotated_y'].iloc[i], clustered_data['rotated_y'].iloc[i + 1]], 
               color='blue', alpha=0.5, linewidth=1.5)
    
    # Calculate sizes based on number of points in each cluster
    max_points = clustered_data['time'].max()
    min_points = clustered_data['time'].min()
    
    # Adjust the size range to make differences more visible
    min_size = 100
    max_size = 2000
    
    # Scale sizes with a logarithmic transformation to better show differences
    sizes = min_size + ((np.log1p(clustered_data['time']) - np.log1p(min_points)) / 
                       (np.log1p(max_points) - np.log1p(min_points))) * (max_size - min_size)
    
    # Plot circles for fixation points
    scatter = ax.scatter(clustered_data['rotated_x'], 
                        clustered_data['rotated_y'], 
                        s=sizes,  # Using our new sizes
                        alpha=0.6, 
                        c=range(len(clustered_data)),  # Color based on sequence
                        cmap='viridis',
                        edgecolors='white',  # Add white edge for better visibility on background
                        linewidths=1)
    
    # Add a small colorbar to show temporal sequence
    cbar = plt.colorbar(scatter, label='Fixation Sequence', shrink=0.6)
    cbar.ax.tick_params(labelsize=8)
    
    # Remove axes, title, and labels
    ax.set_axis_off()
    
    # Make sure the figure is tight
    plt.tight_layout()
    
    # Generate unique filename
    filename = str(uuid.uuid4())
    output_path = f'static/{filename}_fixation_map.png'
    
    # Save the plot without any borders or padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    return output_path


def create_pupil_heatmap(csv_file, background_image=None):
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
    
    # Add background image if provided
    if background_image:
        try:
            img = Image.open(background_image)
            # Resize image to match 1920x1080 dimensions
            img = img.resize((1920, 1080), Image.LANCZOS)
            # Add the background image to the plot
            ax.imshow(np.array(img), extent=[0, 1920, 0, 1080])
        except Exception as e:
            print(f"Error loading background image: {str(e)}")
    
    # Create a scatter plot colored by pupil size
    scatter = ax.scatter(fixation_data['rotated_x'], 
                        fixation_data['rotated_y'], 
                        s=150,  # Increased size for better visibility
                        alpha=0.7, 
                        c=fixation_data['avg_pupil_size'],  # Color based on average pupil size
                        cmap='YlOrRd',  # Yellow-Orange-Red colormap
                        edgecolors='white',  # Add white edge for better visibility
                        linewidths=0.5,
                        norm=plt.Normalize(vmin=fixation_data['avg_pupil_size'].min(), 
                                        vmax=fixation_data['avg_pupil_size'].max()))
    
    # Add colorbar to show pupil size scale
    cbar = plt.colorbar(scatter, label='Average Pupil Size')
    cbar.ax.tick_params(labelsize=8)
    
    # Remove axes
    ax.set_axis_off()
    
    # Make sure the figure is tight
    plt.tight_layout()
    
    # Generate unique filename
    filename = str(uuid.uuid4())
    output_path = f'static/{filename}_pupil_size_heatmap.png'
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    return output_path


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    # Check if CSV file is present
    if 'file' not in request.files:
        return 'No CSV file part in request', 400

    csv_file = request.files['file']
    if csv_file.filename == '':
        return 'No selected CSV file', 400

    if not csv_file.filename.lower().endswith('.csv'):
        return 'Only CSV files are allowed', 400
    
    # Save CSV file
    csv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{csv_file.filename}")
    csv_file.save(csv_filepath)
    
    # Check if image file is present
    background_image_path = None
    if 'image_file' in request.files:
        image_file = request.files['image_file']
        if image_file.filename != '':
            # Check file extension
            allowed_extensions = {'.jpg', '.jpeg', '.png'}
            file_ext = os.path.splitext(image_file.filename.lower())[1]
            
            if file_ext in allowed_extensions:
                # Save image file
                image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{image_file.filename}")
                image_file.save(image_filepath)
                background_image_path = image_filepath
    
    try:
        # Generate plots
        fixation_map_path = create_fixation_map(csv_filepath, background_image_path, distance_threshold=25)
        pupil_heatmap_path = create_pupil_heatmap(csv_filepath, background_image_path)
        
        # Store paths in session or a temporary file
        with open('static/latest_plots.txt', 'w') as f:
            f.write(f"{fixation_map_path}\n{pupil_heatmap_path}")
        
        # Clean up uploaded files
        os.remove(csv_filepath)
        if background_image_path:
            os.remove(background_image_path)
            
        # Clean old static files
        clean_old_files('static', age_seconds=3600)  # Clean files older than 1 hour
        
        return redirect(url_for('download_page'))
    except Exception as e:
        return f"Something went wrong: {str(e)}", 500


@app.route('/download')
def download_page():
    # Read the latest plot paths from the file
    try:
        with open('static/latest_plots.txt', 'r') as f:
            lines = f.readlines()
            fixation_map_path = lines[0].strip()
            pupil_heatmap_path = lines[1].strip()
    except:
        # Fallback to default paths if file doesn't exist
        fixation_map_path = 'static/fixation_map.png'
        pupil_heatmap_path = 'static/pupil_size_heatmap.png'
    
    return render_template('download.html', 
                          fixation_map=fixation_map_path,
                          pupil_heatmap=pupil_heatmap_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)