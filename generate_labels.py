# generate_labels.py
import numpy as np
import sys
sys.path.append("/home/tom/github/landing-segmentation")  # Path to pydem

import pydem.HazardDetection as hd
from tqdm import tqdm
import os

# Define base directory
BASE_DIR = '/home/tom/github/landing-segmentation/data'

def create_labels_from_dem(dem_path, output_path, params):
    """
    Create safety labels from DEM using pydem.HazardDetection
    
    Parameters:
    -----------
    dem_path : str
        Path to the DEM file (.npy)
    output_path : str
        Path to save the label file
    params : dict
        Parameters for hazard detection
    """
    # Load DEM
    dem = np.load(dem_path)
    
    # Run hazard detection
    fpmap, site_slope, site_rghns, pix_rghns, site_safe, indef = hd.dhd(
        dem,
        rmpp=params['rmpp'],
        negative_rghns_unsafe=False,
        lander_type='square',
        dl=params['dl'],
        dp=params['dp'],
        scrit=params['scrit'],
        rcrit=params['rcrit']
    )
    
    # Save label
    np.save(output_path, site_safe)
    return site_safe

def process_split(split_name, params):
    """Process one split of the dataset"""
    input_dir = os.path.join(BASE_DIR, split_name, 'depth_maps')
    output_dir = os.path.join(BASE_DIR, split_name, 'label', 'is_safe')
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of DEM files
    dem_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    
    if not dem_files:
        print(f"No .npy files found in {input_dir}")
        return
    
    print(f"\nProcessing {split_name} split...")
    for dem_file in tqdm(dem_files, desc=f"{split_name}"):
        dem_path = os.path.join(input_dir, dem_file)
        label_path = os.path.join(output_dir, dem_file)  # Use same filename
        
        try:
            create_labels_from_dem(dem_path, label_path, params)
        except Exception as e:
            print(f"Error processing {dem_file}: {str(e)}")

def main():
    # Check if data directory exists
    if not os.path.exists(BASE_DIR):
        print(f"Data directory not found: {BASE_DIR}")
        return
    
    # Parameters for hazard detection
    params = {
        'rmpp': 0.3,              # Resolution in meters per pixel
        'dl': 3.0,                # Lander diameter in meters
        'dp': 0.3,                # Pixel size in meters
        'scrit': 10*np.pi/180,    # Critical slope angle in radians
        'rcrit': 0.3              # Critical roughness threshold
    }
    
    # Process each split
    splits = ['training', 'validation', 'test']
    
    for split in splits:
        split_dir = os.path.join(BASE_DIR, split)
        if os.path.exists(split_dir):
            process_split(split, params)
            
            # Print summary
            depth_maps_dir = os.path.join(split_dir, 'depth_maps')
            labels_dir = os.path.join(split_dir, 'label', 'is_safe')
            
            if os.path.exists(depth_maps_dir) and os.path.exists(labels_dir):
                depth_count = len([f for f in os.listdir(depth_maps_dir) if f.endswith('.npy')])
                label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.npy')])
                print(f"\n{split.capitalize()} set summary:")
                print(f"  Depth maps: {depth_count}")
                print(f"  Labels generated: {label_count}")
                
                if depth_count != label_count:
                    print(f"  Warning: Number of labels doesn't match number of depth maps!")
        else:
            print(f"\nSkipping {split} split - directory not found: {split_dir}")

if __name__ == "__main__":
    main()