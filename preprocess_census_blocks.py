import rasterio
import numpy as np
import json
import os
import geopandas as gpd
import requests
from rasterio.mask import mask
from rasterio.features import rasterize
from shapely.geometry import mapping
import warnings
warnings.filterwarnings('ignore')

def load_census_block_groups():
    """Load census block groups for Mecklenburg County, NC (Charlotte area)"""
    print("Loading census block groups for Mecklenburg County, NC...")
    
    # Load the existing GeoJSON file
    geojson_path = "Data/Census/mecklenburg_block_groups.geojson"
    
    if os.path.exists(geojson_path):
        gdf = gpd.read_file(geojson_path)
        print(f"Loaded {len(gdf)} block groups from {geojson_path}")
        return gdf
    else:
        print(f"Census file not found: {geojson_path}")
        print("Please ensure the census data has been downloaded.")
        return None

def extract_raster_stats_for_polygons(raster_path, polygons_gdf):
    """Extract raster statistics for each polygon"""
    print(f"Extracting statistics from {raster_path}...")
    
    stats_list = []
    
    try:
        with rasterio.open(raster_path) as src:
            # Reproject polygons to match raster CRS
            if polygons_gdf.crs != src.crs:
                polygons_reproj = polygons_gdf.to_crs(src.crs)
            else:
                polygons_reproj = polygons_gdf
            
            for idx, row in polygons_reproj.iterrows():
                try:
                    # Get the geometry
                    geometry = [mapping(row.geometry)]
                    
                    # Mask the raster with the polygon
                    masked_data, masked_transform = mask(src, geometry, crop=True, nodata=np.nan)
                    masked_array = masked_data[0]  # Get first band
                    
                    # Calculate statistics (excluding NaN values)
                    valid_data = masked_array[~np.isnan(masked_array)]
                    
                    # Get geometry in original CRS (WGS84) - use the same index from original dataframe
                    original_row = polygons_gdf.iloc[idx]
                    original_geom = mapping(original_row.geometry)
                    
                    if len(valid_data) > 0:
                        stats = {
                            'GEOID': original_row['GEOID'],
                            'NAME': getattr(original_row, 'NAME', f"Block Group {idx}"),
                            'mean_temp': float(np.mean(valid_data)),
                            'min_temp': float(np.min(valid_data)),
                            'max_temp': float(np.max(valid_data)),
                            'std_temp': float(np.std(valid_data)),
                            'pixel_count': int(len(valid_data)),
                            'geometry': original_geom
                        }
                    else:
                        # No valid data in this block group
                        stats = {
                            'GEOID': original_row['GEOID'],
                            'NAME': getattr(original_row, 'NAME', f"Block Group {idx}"),
                            'mean_temp': None,
                            'min_temp': None,
                            'max_temp': None,
                            'std_temp': None,
                            'pixel_count': 0,
                            'geometry': original_geom
                        }
                    
                    stats_list.append(stats)
                    
                except Exception as e:
                    print(f"Error processing polygon {idx}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error opening raster {raster_path}: {e}")
        return []
    
    print(f"Processed {len(stats_list)} block groups")
    return stats_list

def process_census_block_groups():
    """Main function to process census block groups with temperature data"""
    
    # Load census data
    block_groups_gdf = load_census_block_groups()
    
    if block_groups_gdf is None:
        print("Failed to load census block groups")
        return
    
    # Process each time period
    time_periods = ['am', 'af', 'pm']
    output_dir = "preprocessed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = {}
    
    for time_period in time_periods:
        print(f"\nProcessing {time_period.upper()} data...")
        
        raster_path = f"Data/Rasters/charlotte-north-carolina_{time_period}_temp_f.tif"
        
        if not os.path.exists(raster_path):
            print(f"Raster file not found: {raster_path}")
            continue
        
        # Extract statistics
        stats = extract_raster_stats_for_polygons(raster_path, block_groups_gdf)
        
        if stats:
            # Include ALL block groups, marking which ones have temperature data
            all_data[time_period] = stats
            
            # Save individual file
            output_file = f"{output_dir}/census_blocks_{time_period}.json"
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Count those with valid temperature data
            valid_stats = [s for s in stats if s['mean_temp'] is not None]
            print(f"Saved {len(stats)} total block groups ({len(valid_stats)} with temperature data) to {output_file}")
            
            # Print summary
            if valid_stats:
                temps = [s['mean_temp'] for s in valid_stats]
                print(f"Temperature range: {min(temps):.1f}F - {max(temps):.1f}F")
        else:
            print(f"No valid statistics generated for {time_period}")
    
    # Save combined file
    if all_data:
        combined_file = f"{output_dir}/census_blocks_all.json"
        with open(combined_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        print(f"\nSaved combined data to {combined_file}")
        
        # Print overall summary
        print("\nSummary:")
        for period, data in all_data.items():
            if data:
                valid_temps = [s['mean_temp'] for s in data if s['mean_temp'] is not None]
                print(f"{period.upper()}: {len(data)} total block groups, {len(valid_temps)} with temperature data")
                if valid_temps:
                    print(f"  Temperature range: {min(valid_temps):.1f}F - {max(valid_temps):.1f}F")
    
    print("\nCensus block group processing complete!")

if __name__ == "__main__":
    process_census_block_groups() 