UPLOAD_FOLDER = 'uploads'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from sklearn.cluster import DBSCAN
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import time
import uuid
import json
import math


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


def create_fixation_map(dataframe, background_image=None, distance_threshold=50):
    """
    Create a fixation map from the provided dataframe.
    """
    df = dataframe.copy()
    
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
    
    # Check if we have enough data points for clustering
    if len(coordinates) > 0:
        clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(coordinates)
        fixation_data['cluster'] = clustering.labels_
        
        # Aggregate data by cluster
        clustered_data = fixation_data.groupby('cluster').agg({
            'rotated_x': 'mean',
            'rotated_y': 'mean',
            'time': 'count'  # Count number of points in each cluster
        }).reset_index()
    else:
        # Create empty clustered data if no fixation points
        clustered_data = pd.DataFrame(columns=['cluster', 'rotated_x', 'rotated_y', 'time'])
    
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
    
    # Only plot lines and clusters if we have data
    if len(clustered_data) > 1:
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
    else:
        # Add a text message if there are no fixation points
        ax.text(960, 540, 'No fixation points in this session', 
                fontsize=20, ha='center', va='center', color='gray')
    
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


def create_pupil_heatmap(dataframe, background_image=None):
    """
    Create a pupil size heatmap from the provided dataframe.
    """
    df = dataframe.copy()
    
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
    
    # Only create scatter plot if we have data
    if len(fixation_data) > 0:
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
    else:
        # Add a text message if there are no fixation points
        ax.text(960, 540, 'No pupil data in this session', 
                fontsize=20, ha='center', va='center', color='gray')
    
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


def split_dataframe_by_percentages(df, percentages):
    """
    Split a dataframe into multiple parts based on percentage values.
    
    Args:
        df: Pandas DataFrame to split
        percentages: List of percentage values (should sum to 100)
    
    Returns:
        List of DataFrames
    """
    total_rows = len(df)
    result = []
    start_idx = 0
    
    for percentage in percentages:
        num_rows = math.ceil((percentage / 100) * total_rows)
        end_idx = min(start_idx + num_rows, total_rows)
        
        # Handle edge case for last session
        if percentage == percentages[-1]:
            end_idx = total_rows
            
        session_df = df.iloc[start_idx:end_idx].copy()
        result.append(session_df)
        start_idx = end_idx
        
    return result


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
    
    try:
        # Get session information
        session_count = 0
        session_percentages = []
        session_image_paths = []
        
        # Count the number of sessions based on percentage inputs
        while f'session_percentage_{session_count + 1}' in request.form:
            session_count += 1
        
        # Collect session percentages and image files
        for i in range(1, session_count + 1):
            percentage_key = f'session_percentage_{i}'
            image_key = f'image_file_{i}'
            
            percentage = int(request.form.get(percentage_key, 0))
            session_percentages.append(percentage)
            
            # Check if image file exists for this session
            image_path = None
            if image_key in request.files:
                image_file = request.files[image_key]
                if image_file.filename != '':
                    # Save image file
                    image_filepath = os.path.join(
                        app.config['UPLOAD_FOLDER'], 
                        f"{uuid.uuid4()}_{image_file.filename}"
                    )
                    image_file.save(image_filepath)
                    image_path = image_filepath
            
            session_image_paths.append(image_path)
        
        # Verify total percentage is 100%
        total_percentage = sum(session_percentages)
        if total_percentage != 100:
            return f"Session percentages must sum to 100%. Current sum: {total_percentage}%", 400
        
        # Read the CSV data
        df = pd.read_csv(csv_filepath)
        
        # Split the dataframe based on percentages
        session_dataframes = split_dataframe_by_percentages(df, session_percentages)
        
        # Process each session
        session_results = []
        
        for i, session_df in enumerate(session_dataframes):
            session_id = i + 1
            session_percentage = session_percentages[i]
            background_image = session_image_paths[i]
            
            # Generate visualizations for this session
            fixation_map_path = create_fixation_map(
                session_df, 
                background_image,
                distance_threshold=25
            )
            
            pupil_heatmap_path = create_pupil_heatmap(
                session_df, 
                background_image
            )
            
            # Collect session info
            session_info = {
                'id': session_id,
                'percentage': session_percentage,
                'data_count': len(session_df),
                'fixation_map': fixation_map_path,
                'pupil_heatmap': pupil_heatmap_path
            }
            
            session_results.append(session_info)
        
        # Store session results for download page
        with open('static/session_results.json', 'w') as f:
            json.dump(session_results, f)
        
        # Clean up uploaded files
        os.remove(csv_filepath)
        for image_path in session_image_paths:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
            
        # Clean old static files
        clean_old_files('static', age_seconds=3600)  # Clean files older than 1 hour
        
        return redirect(url_for('download_page'))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Something went wrong: {str(e)}", 500


@app.route('/download')
def download_page():
    # Read the session results from the JSON file
    try:
        with open('static/session_results.json', 'r') as f:
            sessions = json.load(f)
    except:
        # Fallback to empty sessions if file doesn't exist
        sessions = []
    
    return render_template('download.html', sessions=sessions)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)