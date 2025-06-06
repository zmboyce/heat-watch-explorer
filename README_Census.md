# Heat Watch Explorer - Census Block Groups

A streamlined web application for exploring temperature data in Charlotte, NC using real Census Block Groups instead of artificial hexagon grids.

## Features

### ✅ Census Block Group Integration
- **Real Administrative Boundaries**: Uses official US Census Bureau block groups for Mecklenburg County, NC
- **288 Block Groups**: Covers all block groups with temperature data in the Charlotte area
- **Perfect Tessellation**: No overlapping boundaries - proper administrative polygons

### 🌡️ Temperature Analysis
- **Multiple Variables**: Mean, min, max, standard deviation, and temperature range
- **Time Periods**: Morning (AM), Afternoon (AF), and Evening (PM) data
- **Statistical Summaries**: Comprehensive statistics for each block group
- **Interactive Filtering**: Real-time temperature range filtering

### 📊 Visualization Features
- **Choropleth Maps**: Color-coded census block groups based on temperature
- **Blue-to-Red Color Scheme**: Consistent color mapping across all visualizations
- **Interactive Charts**: Temperature distributions, spatial scatter plots, and time comparisons
- **Data Export**: Download filtered data as CSV

## Data Sources

### Temperature Rasters
- **Source**: Charlotte, NC temperature measurements
- **Files**: `charlotte-north-carolina_{am,af,pm}_temp_f.tif`
- **Format**: GeoTIFF raster files with temperature in Fahrenheit

### Census Block Groups
- **Source**: US Census Bureau TIGER/Line Shapefiles 2023
- **Coverage**: Mecklenburg County, NC (FIPS: 37119)
- **Download**: Automated via `preprocess_census_blocks.py`

## Data Processing Results

### Temperature Statistics by Time Period

| Time Period | Block Groups | Temperature Range | Mean Temp |
|-------------|-------------|-------------------|-----------|
| Morning (AM) | 288 | 74.4°F - 81.4°F | ~77.9°F |
| Afternoon (AF) | 288 | 88.8°F - 99.0°F | ~93.9°F |
| Evening (PM) | 288 | 88.8°F - 96.5°F | ~92.7°F |

### Data Quality
- **Valid Coverage**: 288 out of 386 total block groups have temperature data
- **High Resolution**: Average of ~150 pixels per block group
- **No Overlaps**: Perfect tessellation with no gaps or overlaps
- **Coordinate Accuracy**: Proper CRS transformation from UTM to WGS84

## Installation & Setup

### Prerequisites
```bash
pip install streamlit plotly pandas numpy rasterio geopandas shapely requests
```

### Quick Start
1. **Generate Census Data** (one-time setup):
   ```bash
   python preprocess_census_blocks.py
   ```

2. **Run the Application**:
   ```bash
   streamlit run app_census.py --server.port 8510
   ```

3. **Access the App**: Open http://localhost:8510 in your browser

## File Structure

```
HeatExplorer/
├── app_census.py                    # Main Streamlit application
├── preprocess_census_blocks.py      # Data preprocessing script
├── requirements.txt                 # Python dependencies
├── README_Census.md                 # This documentation
├── Data/
│   ├── Rasters/                     # Temperature raster files
│   │   ├── charlotte-north-carolina_am_temp_f.tif
│   │   ├── charlotte-north-carolina_af_temp_f.tif
│   │   └── charlotte-north-carolina_pm_temp_f.tif
│   └── Census/                      # Census data (auto-downloaded)
│       ├── nc_block_groups.zip
│       ├── extracted/
│       └── mecklenburg_block_groups.geojson
└── preprocessed_data/               # Processed temperature statistics
    ├── census_blocks_am.json
    ├── census_blocks_af.json
    ├── census_blocks_pm.json
    └── census_blocks_all.json
```

## How It Works

### 1. Data Download
- Downloads 2023 Census TIGER/Line block groups for North Carolina
- Filters to Mecklenburg County (Charlotte area)
- Saves as GeoJSON for easy processing

### 2. Temperature Extraction
- Opens each temperature raster file
- Reprojects census polygons to match raster CRS
- Uses rasterio.mask to extract pixel values within each block group
- Calculates statistics (mean, min, max, std) for each polygon

### 3. Visualization
- Loads preprocessed JSON data for fast rendering
- Creates choropleth maps using Plotly ScatterMapbox
- Applies consistent blue-to-red color schemes
- Provides interactive filtering and analysis tools

## Advantages Over Hexagon Approach

### ✅ Perfect Tessellation
- No overlapping or gaps between boundaries
- Follows natural administrative divisions
- Familiar geographic units for residents

### ✅ Real-World Relevance
- Census block groups are meaningful administrative units
- Can be linked to demographic and socioeconomic data
- Useful for urban planning and policy decisions

### ✅ Better Performance
- Pre-processed data loads instantly
- No complex tessellation calculations
- Efficient polygon rendering

### ✅ Data Quality
- Complete coverage of study area
- Statistically robust sample sizes
- Consistent boundary definitions

## Future Enhancements

### Demographic Integration
- Link to Census demographic data (population, income, etc.)
- Analyze temperature vs. socioeconomic factors
- Environmental justice analysis

### Traverse Data Integration
- Add point measurements from traverse shapefiles
- Compare raster vs. point temperature data
- Validation and uncertainty analysis

### Time Series Analysis
- Multi-day temperature trends
- Seasonal variation analysis
- Heat island evolution over time

## Technical Notes

### Coordinate Systems
- **Input Rasters**: UTM projection (meters)
- **Census Data**: WGS84 (degrees)
- **Display**: WGS84 for web mapping compatibility

### Data Size
- Raw census data: ~2MB per time period
- Preprocessed JSON: ~3-4MB per file
- App loads data efficiently with caching

### Performance Optimization
- Streamlit caching for data loading
- Pre-computed statistics for fast filtering
- Efficient polygon rendering with ScatterMapbox

## Troubleshooting

### Missing Temperature Data
- Some block groups outside raster coverage show as "no overlap"
- This is expected for areas beyond the study area
- 288/386 coverage rate is excellent for urban analysis

### Slow Loading
- First run downloads 50MB+ census data
- Subsequent runs use cached data
- Consider pre-downloading census data for deployment

### Color Scale Issues
- Blue-to-red color scheme is consistent across all visualizations
- Temperature normalization handles extreme values well
- Color bar shows actual temperature ranges

## Contact & Support

This Heat Watch Explorer demonstrates the power of combining real administrative boundaries with high-resolution temperature data for meaningful urban heat analysis.

Built with ❤️ using Streamlit, Plotly, and open data sources. 