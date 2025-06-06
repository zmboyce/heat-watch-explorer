# Heat Watch Explorer

An interactive web application for exploring temperature data in Charlotte, NC using US Census Block Groups and raster temperature data.

## Features

- **Real Administrative Boundaries**: Uses official US Census Bureau block groups for Mecklenburg County, NC
- **Temperature Analysis**: Multiple temperature variables (mean, min, max, standard deviation, range)
- **Time Period Comparison**: Morning (AM), Afternoon (AF), and Evening (PM) temperature data
- **Interactive Visualization**: 
  - Choropleth maps with temperature-based coloring
  - Temperature distribution histograms
  - Spatial scatter plots
  - Interactive filtering and analysis
- **Data Export**: Download filtered datasets as CSV

## Data Sources

- **Temperature Rasters**: Charlotte, NC temperature data in GeoTIFF format
- **Census Boundaries**: US Census Bureau TIGER/Line Shapefiles (2023 Block Groups)
- **Geographic Coverage**: Mecklenburg County, North Carolina (Charlotte metropolitan area)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/heat-watch-explorer.git
   cd heat-watch-explorer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data:**
   - Place temperature raster files (.tif) in `Data/Rasters/`
   - Files should be named: `charlotte-north-carolina_{am,af,pm}_temp_f.tif`
   - Census data will be downloaded automatically on first run

## Usage

### Run the Census Block Groups Application

```bash
streamlit run app_census.py --server.port 8514
```

Then open your browser to `http://localhost:8514`

### Data Preprocessing

To regenerate census statistics from raster data:

```bash
python preprocess_census_blocks.py
```

This will:
- Download US Census block groups for Mecklenburg County
- Extract temperature statistics for each block group
- Generate processed data files in `preprocessed_data/`

## Project Structure

```
heat-watch-explorer/
├── app_census.py                    # Main Streamlit application
├── preprocess_census_blocks.py      # Data preprocessing script
├── requirements.txt                 # Python dependencies
├── Data/
│   ├── Rasters/                    # Temperature raster files (.tif)
│   └── Census/                     # Census boundary data
├── preprocessed_data/              # Processed temperature statistics
└── README.md
```

## Technical Details

### Data Processing Pipeline

1. **Census Data**: Downloads 2023 TIGER/Line block groups for North Carolina, filters to Mecklenburg County
2. **Raster Analysis**: Extracts temperature statistics using zonal statistics for each census polygon
3. **Coordinate Systems**: Handles CRS transformations between UTM and WGS84
4. **Statistical Calculations**: Computes mean, min, max, standard deviation, and pixel counts

### Visualization Components

- **Plotly Maps**: Interactive choropleth maps with census block polygons
- **Temperature Filtering**: Real-time filtering with temperature range sliders
- **Statistical Charts**: Distribution histograms and spatial scatter plots
- **Color Schemes**: Consistent blue-to-red temperature color mapping

## Data Coverage

- **Total Census Block Groups**: 624 in Mecklenburg County
- **With Temperature Data**: ~288 block groups (46% coverage)
- **Temperature Ranges**:
  - AM: 74.4°F - 81.4°F
  - AF: 88.8°F - 99.0°F  
  - PM: 88.8°F - 96.5°F

## Dependencies

- `streamlit` - Web application framework
- `plotly` - Interactive visualization
- `geopandas` - Geospatial data processing
- `rasterio` - Raster data analysis
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `requests` - HTTP requests for data download

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- US Census Bureau for TIGER/Line boundary data
- Temperature data providers (specify your source)
- Streamlit and Plotly communities for excellent documentation

## Contact

Your Name - your.email@example.com
Project Link: [https://github.com/yourusername/heat-watch-explorer](https://github.com/yourusername/heat-watch-explorer) 