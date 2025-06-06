import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
import numpy as np
from shapely.geometry import shape
import requests
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Heat Watch Explorer - Comprehensive Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_census_data(time_period):
    """Load census block group data with temperature statistics"""
    census_file = f"preprocessed_data/census_blocks_{time_period}.json"
    
    if not os.path.exists(census_file):
        return None
    
    try:
        with open(census_file, 'r') as f:
            census_data = json.load(f)
        return census_data
    except Exception as e:
        st.error(f"Error loading census data: {e}")
        return None

def census_data_to_dataframe(census_data):
    """Convert census data to pandas DataFrame"""
    if not census_data:
        return pd.DataFrame()
    
    df_data = []
    for block in census_data:
        # Calculate center point for scatter plots
        geom = shape(block['geometry'])
        centroid = geom.centroid
        
        df_data.append({
            'GEOID': block['GEOID'],
            'NAME': block.get('NAME', 'Unknown'),
            'mean_temp': block['mean_temp'],
            'min_temp': block['min_temp'],
            'max_temp': block['max_temp'],
            'std_temp': block['std_temp'],
            'pixel_count': block['pixel_count'],
            'lat': centroid.y,
            'lng': centroid.x,
            'temp_range': block['max_temp'] - block['min_temp'] if block['max_temp'] is not None and block['min_temp'] is not None else None
        })
    
    return pd.DataFrame(df_data)

def create_census_map_plotly(census_data, temp_filter_min, temp_filter_max, color_by='mean_temp'):
    """Create Plotly choropleth map with census block groups"""
    
    # Filter data by temperature
    filtered_data = [b for b in census_data if b.get(color_by) is not None and temp_filter_min <= b[color_by] <= temp_filter_max]
    
    if not filtered_data:
        return go.Figure()
    
    # Create the figure
    fig = go.Figure()
    
    # Get temperature range for color normalization
    temps = [b[color_by] for b in filtered_data]
    temp_min, temp_max = min(temps), max(temps)
    
    # Add each census block as a choropleth polygon
    for block in filtered_data:
        temp = block[color_by]
        
        # Normalize temperature for color mapping
        if temp_max > temp_min:
            norm_temp = (temp - temp_min) / (temp_max - temp_min)
        else:
            norm_temp = 0.5
        
        # Get color from colorscale
        color = px.colors.sample_colorscale('RdYlBu_r', [norm_temp])[0]
        
        # Extract coordinates from geometry
        geom = block['geometry']
        if geom['type'] == 'Polygon':
            coords = geom['coordinates'][0]
        else:  # MultiPolygon - use first polygon
            coords = geom['coordinates'][0][0]
        
        # Convert to lat/lng lists
        lats = [coord[1] for coord in coords] + [coords[0][1]]  # Close polygon
        lons = [coord[0] for coord in coords] + [coords[0][0]]
        
        # Add census block polygon
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='lines',
            fill='toself',
            fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.7)'),
            line=dict(width=0.5, color='white'),
            hovertemplate=f'''
            <b>Block Group: {block["GEOID"]}</b><br>
            Mean Temp: {block["mean_temp"]:.1f}°F<br>
            Min Temp: {block["min_temp"]:.1f}°F<br>
            Max Temp: {block["max_temp"]:.1f}°F<br>
            Std Dev: {block["std_temp"]:.1f}°F<br>
            Pixels: {block["pixel_count"]}
            <extra></extra>''',
            showlegend=False,
            name=""
        ))
    
    # Add a dummy trace for the colorbar
    if filtered_data:
        fig.add_trace(go.Scattermapbox(
            lat=[filtered_data[0]['geometry']['coordinates'][0][0][1]],
            lon=[filtered_data[0]['geometry']['coordinates'][0][0][0]],
            mode='markers',
            marker=dict(
                size=0,
                color=temps,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(
                    title=dict(text=f"{color_by.replace('_', ' ').title()} (°F)", side="right"),
                    x=1.02
                )
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=35.19, lon=-80.79),
            zoom=10
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=600
    )
    
    return fig

def create_temperature_distribution(df, time_period, temp_col='mean_temp'):
    """Create temperature distribution plot with blue-to-red color scheme"""
    if df.empty:
        return go.Figure()
    
    # Create histogram with color mapping
    fig = go.Figure()
    
    # Create histogram bins
    hist_data = df[temp_col].values
    n_bins = 20
    
    counts, bin_edges = np.histogram(hist_data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Color mapping - normalize bin centers to 0-1 range for colorscale
    temp_min, temp_max = df[temp_col].min(), df[temp_col].max()
    if temp_max > temp_min:
        normalized_temps = (bin_centers - temp_min) / (temp_max - temp_min)
    else:
        normalized_temps = np.ones_like(bin_centers) * 0.5
    
    # Create color array using the same colorscale as the map
    colors = px.colors.sample_colorscale('RdYlBu_r', normalized_temps)
    
    # Add histogram bars
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        width=(bin_edges[1] - bin_edges[0]) * 0.9,
        marker=dict(
            color=colors,
            line=dict(color='white', width=0.5)
        ),
        name='Temperature Distribution',
        hovertemplate=f'{temp_col.replace("_", " ").title()}: %{{x:.1f}}°F<br>Count: %{{y}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{temp_col.replace("_", " ").title()} Distribution - {time_period.upper()}',
        xaxis_title=f'{temp_col.replace("_", " ").title()} (°F)',
        yaxis_title='Block Groups',
        height=300,
        margin=dict(r=0, t=40, l=0, b=0),
        showlegend=False
    )
    
    return fig

def create_scatter_plot(df, time_period, temp_col='mean_temp'):
    """Create scatter plot of temperature vs spatial position"""
    if df.empty:
        return go.Figure()
    
    fig = px.scatter(
        df,
        x='lng',
        y='lat',
        color=temp_col,
        size='pixel_count',
        size_max=15,
        color_continuous_scale='RdYlBu_r',
        title=f'{temp_col.replace("_", " ").title()} by Location - {time_period.upper()}',
        labels={'lng': 'Longitude', 'lat': 'Latitude', temp_col: f'{temp_col.replace("_", " ").title()} (°F)'},
        hover_data=['GEOID', 'min_temp', 'max_temp', 'std_temp']
    )
    
    fig.update_layout(
        height=400,
        margin=dict(r=0, t=40, l=0, b=0)
    )
    
    return fig

# Bus Stops Integration Classes
class BusStopsTemperatureAnalyzer:
    def __init__(self):
        self.bus_stops_url = "https://gis.charlottenc.gov/arcgis/rest/services/HNS/HousingLocationalToolLayers/MapServer/16/query"
    
    @st.cache_data(ttl=3600)
    def fetch_bus_stops(_self):
        """Fetch bus stops data from Charlotte ArcGIS REST API"""
        params = {
            'where': '1=1',  # Get all records
            'outFields': '*',
            'f': 'json',
            'returnGeometry': 'true'
        }
        
        try:
            response = requests.get(_self.bus_stops_url, params=params, timeout=30)
            data = response.json()
            
            if 'features' in data:
                bus_stops = []
                for feature in data['features']:
                    attrs = feature['attributes']
                    geom = feature['geometry']
                    
                    bus_stops.append({
                        'StopID': attrs.get('StopID'),
                        'StopDesc': attrs.get('StopDesc'),
                        'Nearest_Intersection': attrs.get('Nearest_In'),
                        'Bench': attrs.get('Bench') == 'Yes',
                        'Shelter': attrs.get('Shelter') == 'Yes',
                        'Routes': attrs.get('routes'),
                        'Frequency_AM_Peak': attrs.get('Frequency_AM_Peak', 0) or 0,
                        'Frequency_Midday': attrs.get('Frquency_Midday', 0) or 0,
                        'Frequency_PM_Peak': attrs.get('Frequency_PM_Peak', 0) or 0,
                        'Frequency_Evening': attrs.get('Frequency_Evening', 0) or 0,
                        'lat': geom['y'],
                        'lng': geom['x']
                    })
                
                return pd.DataFrame(bus_stops)
            
        except Exception as e:
            st.error(f"Error fetching bus stops: {e}")
            return pd.DataFrame()
    
    def interpolate_temperature_at_stops(self, bus_stops_df, census_temperature_df):
        """Interpolate temperature values at bus stop locations"""
        if bus_stops_df.empty or census_temperature_df.empty:
            return bus_stops_df
        
        # Create coordinate arrays
        bus_coords = bus_stops_df[['lat', 'lng']].values
        census_coords = census_temperature_df[['lat', 'lng']].values
        
        # Calculate distances between bus stops and census points
        distances = cdist(bus_coords, census_coords)
        
        # Use inverse distance weighting for temperature interpolation
        temperatures = []
        for i, stop_distances in enumerate(distances):
            # Find nearest census points (within reasonable distance)
            nearby_indices = np.where(stop_distances < 0.01)[0]  # ~1km radius
            
            if len(nearby_indices) > 0:
                nearby_temps = census_temperature_df.iloc[nearby_indices]['mean_temp'].values
                nearby_dists = stop_distances[nearby_indices]
                
                # Inverse distance weighting
                if len(nearby_temps) == 1:
                    temp = nearby_temps[0]
                else:
                    weights = 1 / (nearby_dists + 1e-10)  # Avoid division by zero
                    temp = np.average(nearby_temps, weights=weights)
                
                temperatures.append(temp)
            else:
                # Use nearest point if none within radius
                nearest_idx = np.argmin(stop_distances)
                temperatures.append(census_temperature_df.iloc[nearest_idx]['mean_temp'])
        
        bus_stops_df['estimated_temp'] = temperatures
        return bus_stops_df
    
    def calculate_heat_exposure_risk(self, bus_stops_df):
        """Calculate heat exposure risk based on temperature and infrastructure"""
        def calculate_risk(row):
            base_risk = row['estimated_temp']
            
            # Adjust risk based on infrastructure
            if row['Shelter']:
                base_risk -= 5  # Shelter provides significant cooling
            elif row['Bench']:
                base_risk -= 2  # Bench provides some relief
            
            # Higher frequency stops = more exposure time
            avg_frequency = (row['Frequency_AM_Peak'] + row['Frequency_Midday'] + 
                           row['Frequency_PM_Peak'] + row['Frequency_Evening']) / 4
            
            if avg_frequency > 4:  # High frequency (>4 buses/hour)
                risk_multiplier = 1.0
            elif avg_frequency > 2:  # Medium frequency
                risk_multiplier = 1.1
            else:  # Low frequency (long wait times)
                risk_multiplier = 1.2
            
            return base_risk * risk_multiplier
        
        bus_stops_df['heat_risk_score'] = bus_stops_df.apply(calculate_risk, axis=1)
        
        # Categorize risk levels
        def categorize_risk(score):
            if score >= 95:
                return 'Extreme'
            elif score >= 85:
                return 'High'
            elif score >= 75:
                return 'Moderate'
            else:
                return 'Low'
        
        bus_stops_df['risk_category'] = bus_stops_df['heat_risk_score'].apply(categorize_risk)
        return bus_stops_df
    
    def create_bus_stops_map(self, bus_stops_df, time_period='all'):
        """Create interactive map of bus stops with heat risk"""
        if bus_stops_df.empty:
            return go.Figure()
        
        # Color mapping for risk categories
        color_map = {
            'Low': '#2E8B57',      # Sea Green
            'Moderate': '#FFD700',  # Gold
            'High': '#FF6347',      # Tomato
            'Extreme': '#DC143C'    # Crimson
        }
        
        fig = px.scatter_mapbox(
            bus_stops_df,
            lat='lat',
            lon='lng',
            color='risk_category',
            size='heat_risk_score',
            size_max=15,
            color_discrete_map=color_map,
            hover_data={
                'StopID': True,
                'StopDesc': True,
                'estimated_temp': ':.1f',
                'heat_risk_score': ':.1f',
                'Shelter': True,
                'Bench': True,
                'Routes': True
            },
            title=f'Bus Stop Heat Risk Analysis - {time_period.upper()}',
            mapbox_style="open-street-map",
            zoom=10,
            center=dict(lat=35.19, lon=-80.79)
        )
        
        fig.update_layout(
            height=600,
            margin=dict(r=0, t=40, l=0, b=0)
        )
        
        return fig

# Temperature Forecasting Classes
class TemperatureForecastingSystem:
    def __init__(self, openweather_api_key=None):
        self.api_key = openweather_api_key
        self.charlotte_coords = {'lat': 35.19, 'lng': -80.79}
        self.weather_stations = [
            {'name': 'Charlotte Douglas', 'lat': 35.214, 'lng': -80.943},
            {'name': 'Charlotte Downtown', 'lat': 35.227, 'lng': -80.843},
            {'name': 'Charlotte University', 'lat': 35.308, 'lng': -80.734},
            {'name': 'Charlotte South', 'lat': 35.087, 'lng': -80.792},
            {'name': 'Charlotte East', 'lat': 35.192, 'lng': -80.627}
        ]
    
    @st.cache_data(ttl=3600)
    def fetch_current_weather_data(_self):
        """Fetch current weather data from multiple points around Charlotte"""
        if not _self.api_key:
            return _self._generate_mock_weather_data()
        
        weather_data = []
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        
        for station in _self.weather_stations:
            try:
                params = {
                    'lat': station['lat'],
                    'lon': station['lng'],
                    'appid': _self.api_key,
                    'units': 'imperial'
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                data = response.json()
                
                if response.status_code == 200:
                    weather_data.append({
                        'name': station['name'],
                        'lat': station['lat'],
                        'lng': station['lng'],
                        'temperature': data['main']['temp'],
                        'humidity': data['main']['humidity'],
                        'pressure': data['main']['pressure'],
                        'wind_speed': data['wind']['speed'],
                        'clouds': data['clouds']['all'],
                        'timestamp': datetime.now()
                    })
                
            except Exception as e:
                st.warning(f"Failed to fetch data for {station['name']}: {e}")
        
        return pd.DataFrame(weather_data)
    
    def _generate_mock_weather_data(self):
        """Generate mock weather data for demonstration"""
        np.random.seed(42)
        base_temp = 85  # Summer temperature
        
        mock_data = []
        for i, station in enumerate(self.weather_stations):
            # Add some spatial variation
            temp_variation = np.random.normal(0, 3)
            mock_data.append({
                'name': station['name'],
                'lat': station['lat'],
                'lng': station['lng'],
                'temperature': base_temp + temp_variation,
                'humidity': np.random.randint(45, 75),
                'pressure': np.random.normal(1013, 5),
                'wind_speed': np.random.exponential(5),
                'clouds': np.random.randint(10, 90),
                'timestamp': datetime.now()
            })
        
        return pd.DataFrame(mock_data)
    
    @st.cache_data(ttl=3600)
    def fetch_forecast_data(_self, hours=24):
        """Fetch weather forecast data"""
        if not _self.api_key:
            return _self._generate_mock_forecast_data(hours)
        
        forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
        
        try:
            params = {
                'lat': _self.charlotte_coords['lat'],
                'lon': _self.charlotte_coords['lng'],
                'appid': _self.api_key,
                'units': 'imperial'
            }
            
            response = requests.get(forecast_url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                forecast_data = []
                for item in data['list'][:hours//3]:  # 3-hour intervals
                    forecast_data.append({
                        'datetime': datetime.fromtimestamp(item['dt']),
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'pressure': item['main']['pressure'],
                        'wind_speed': item['wind']['speed'],
                        'clouds': item['clouds']['all'],
                        'description': item['weather'][0]['description']
                    })
                
                return pd.DataFrame(forecast_data)
        
        except Exception as e:
            st.error(f"Error fetching forecast data: {e}")
            return pd.DataFrame()
    
    def _generate_mock_forecast_data(self, hours):
        """Generate mock forecast data"""
        np.random.seed(42)
        
        forecast_data = []
        current_time = datetime.now()
        base_temp = 85
        
        for i in range(hours):
            # Simulate daily temperature cycle
            hour_angle = (i % 24) * 2 * np.pi / 24
            daily_variation = 8 * np.sin(hour_angle - np.pi/3)  # Peak at ~2 PM
            random_variation = np.random.normal(0, 2)
            
            forecast_data.append({
                'datetime': current_time + timedelta(hours=i),
                'temperature': base_temp + daily_variation + random_variation,
                'humidity': max(30, min(90, 60 + np.random.normal(0, 15))),
                'pressure': 1013 + np.random.normal(0, 5),
                'wind_speed': max(0, np.random.exponential(5)),
                'clouds': max(0, min(100, 50 + np.random.normal(0, 20))),
                'description': np.random.choice(['clear sky', 'few clouds', 'scattered clouds', 'broken clouds'])
            })
        
        return pd.DataFrame(forecast_data)

# Main app
st.title("🌡️ Heat Watch Explorer")
st.markdown("**Comprehensive Temperature Analysis & Heat Risk Assessment for Charlotte, NC**")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Census Analysis", "🚌 Bus Stop Heat Risk", "🌡️ Temperature Forecasting"])

with tab1:
    st.header("Census Block Groups Temperature Analysis")
    st.markdown("**Interactive Temperature Analysis using Census Block Groups**")
    
    # Sidebar controls
    st.sidebar.header("Census Analysis Controls")
    
    # Time period selection
    time_options = {
        'am': 'Morning (AM)',
        'af': 'Afternoon (AF)', 
        'pm': 'Evening (PM)'
    }
    
    selected_time = st.sidebar.selectbox(
        "Select Time Period:",
        options=list(time_options.keys()),
        format_func=lambda x: time_options[x],
        index=1  # Default to afternoon
    )
    
    # Load data
    census_data = load_census_data(selected_time)
    
    if census_data:
        df = census_data_to_dataframe(census_data)
        
        # Temperature variable selection
        st.sidebar.subheader("Temperature Variable")
        temp_variable = st.sidebar.selectbox(
            "Variable to Display:",
            options=['mean_temp', 'min_temp', 'max_temp', 'std_temp', 'temp_range'],
            format_func=lambda x: {
                'mean_temp': 'Mean Temperature',
                'min_temp': 'Minimum Temperature', 
                'max_temp': 'Maximum Temperature',
                'std_temp': 'Temperature Std Dev',
                'temp_range': 'Temperature Range'
            }[x],
            index=0
        )
        
        # Temperature filtering
        st.sidebar.subheader("Temperature Filter")
        
        temp_min = float(df[temp_variable].min())
        temp_max = float(df[temp_variable].max())
        
        temp_range = st.sidebar.slider(
            f"{temp_variable.replace('_', ' ').title()} Range (°F)",
            min_value=temp_min,
            max_value=temp_max,
            value=(temp_min, temp_max),
            step=0.1
        )
        
        temp_filter_min, temp_filter_max = temp_range
        
        # Filter dataframe
        filtered_df = df[(df[temp_variable] >= temp_filter_min) & (df[temp_variable] <= temp_filter_max)]
        
        # Display metrics
        st.sidebar.metric("Total Block Groups", len(df))
        st.sidebar.metric("Filtered Block Groups", len(filtered_df))
        st.sidebar.metric("Average Pixel Count", f"{df['pixel_count'].mean():.0f}")
        
        if not filtered_df.empty:
            st.sidebar.metric(f"Avg {temp_variable.replace('_', ' ').title()}", f"{filtered_df[temp_variable].mean():.1f}°F")
        
        # Main layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"{temp_variable.replace('_', ' ').title()} Map - {time_options[selected_time]}")
            
            # Create and display the map
            fig_map = create_census_map_plotly(census_data, temp_filter_min, temp_filter_max, temp_variable)
            st.plotly_chart(fig_map, use_container_width=True)
        
        with col2:
            st.subheader("Data Analysis")
            
            # Temperature distribution
            fig_dist = create_temperature_distribution(filtered_df, selected_time, temp_variable)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Data summary
            if not filtered_df.empty:
                st.markdown("**Statistics**")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(f"Min {temp_variable.replace('_', ' ')}", f"{filtered_df[temp_variable].min():.1f}°F")
                    st.metric(f"Max {temp_variable.replace('_', ' ')}", f"{filtered_df[temp_variable].max():.1f}°F")
                
                with col_b:
                    st.metric(f"Mean {temp_variable.replace('_', ' ')}", f"{filtered_df[temp_variable].mean():.1f}°F")
                    st.metric(f"Std Dev", f"{filtered_df[temp_variable].std():.1f}°F")
            
        # Full width scatter plot
        st.subheader("Spatial Temperature Analysis")
        fig_scatter = create_scatter_plot(filtered_df, selected_time, temp_variable)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Comparison across time periods
        st.subheader("Temperature Comparison Across Time Periods")
        
        # Load all time periods for comparison
        all_time_data = {}
        for period in time_options.keys():
            period_data = load_census_data(period)
            if period_data:
                period_df = census_data_to_dataframe(period_data)
                all_time_data[period] = period_df
        
        if len(all_time_data) > 1:
            # Create comparison plot
            comparison_data = []
            for period, data_df in all_time_data.items():
                comparison_data.extend([
                    {
                        'Time Period': time_options[period],
                        'GEOID': row['GEOID'],
                        'Temperature': row[temp_variable],
                        'Variable': temp_variable.replace('_', ' ').title()
                    }
                    for _, row in data_df.iterrows()
                ])
            
            comparison_df = pd.DataFrame(comparison_data)
            
            if not comparison_df.empty:
                fig_box = px.box(
                    comparison_df,
                    x='Time Period',
                    y='Temperature',
                    title=f'{temp_variable.replace("_", " ").title()} Distribution by Time Period',
                    color='Time Period',
                    color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12']
                )
                
                fig_box.update_layout(
                    height=400,
                    margin=dict(r=0, t=40, l=0, b=0)
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
        
        # Data table (expandable)
        with st.expander("View Block Group Data"):
            st.dataframe(
                filtered_df.round(2),
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name=f"census_blocks_{selected_time}_{temp_variable}_{temp_filter_min:.1f}_{temp_filter_max:.1f}F.csv",
                mime="text/csv"
            )
    
    else:
        st.error("No census block group data found. Please run the preprocessing script.")
        st.info("Run `python preprocess_census_blocks.py` to generate the required data files.")

with tab2:
    st.header("🚌 Bus Stop Heat Risk Analysis")
    st.markdown("**Assess heat exposure risk at bus stops using real-time infrastructure data**")
    
    analyzer = BusStopsTemperatureAnalyzer()
    
    # Fetch bus stops data
    with st.spinner("Loading bus stops data..."):
        bus_stops_df = analyzer.fetch_bus_stops()
    
    if not bus_stops_df.empty:
        st.success(f"Loaded {len(bus_stops_df)} bus stops")
        
        # Time period selection for bus stops
        bus_time_period = st.selectbox(
            "Select Time Period for Heat Analysis:",
            options=['am', 'af', 'pm'],
            format_func=lambda x: {'am': 'Morning', 'af': 'Afternoon', 'pm': 'Evening'}[x],
            index=1,
            key="bus_time_period"
        )
        
        # Load corresponding temperature data
        census_data_for_stops = load_census_data(bus_time_period)
        if census_data_for_stops:
            census_df_for_stops = census_data_to_dataframe(census_data_for_stops)
            
            # Interpolate temperature at bus stops
            bus_stops_with_temp = analyzer.interpolate_temperature_at_stops(bus_stops_df, census_df_for_stops)
            
            # Calculate heat risk
            bus_stops_with_risk = analyzer.calculate_heat_exposure_risk(bus_stops_with_temp)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Bus Stops", len(bus_stops_with_risk))
            with col2:
                sheltered = bus_stops_with_risk['Shelter'].sum()
                st.metric("Stops with Shelter", f"{sheltered} ({sheltered/len(bus_stops_with_risk)*100:.1f}%)")
            with col3:
                high_risk = len(bus_stops_with_risk[bus_stops_with_risk['risk_category'].isin(['High', 'Extreme'])])
                st.metric("High/Extreme Risk Stops", f"{high_risk} ({high_risk/len(bus_stops_with_risk)*100:.1f}%)")
            with col4:
                avg_temp = bus_stops_with_risk['estimated_temp'].mean()
                st.metric("Avg. Estimated Temp", f"{avg_temp:.1f}°F")
            
            # Display map
            fig_bus_map = analyzer.create_bus_stops_map(bus_stops_with_risk, bus_time_period)
            st.plotly_chart(fig_bus_map, use_container_width=True)
            
            # Risk analysis charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk category pie chart
                risk_counts = bus_stops_with_risk['risk_category'].value_counts()
                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title='Bus Stops by Heat Risk Category',
                    color_discrete_map={
                        'Low': '#2E8B57',
                        'Moderate': '#FFD700',
                        'High': '#FF6347',
                        'Extreme': '#DC143C'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Infrastructure vs Risk scatter
                fig_scatter = px.scatter(
                    bus_stops_with_risk,
                    x='estimated_temp',
                    y='heat_risk_score',
                    color='Shelter',
                    symbol='Bench',
                    size='Frequency_AM_Peak',
                    hover_data=['StopID', 'StopDesc'],
                    title='Temperature vs Heat Risk Score',
                    labels={
                        'estimated_temp': 'Estimated Temperature (°F)',
                        'heat_risk_score': 'Heat Risk Score'
                    }
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # High-risk stops table
            high_risk_stops = bus_stops_with_risk[
                bus_stops_with_risk['risk_category'].isin(['High', 'Extreme'])
            ].sort_values('heat_risk_score', ascending=False)
            
            if not high_risk_stops.empty:
                st.subheader("🚨 High Risk Bus Stops")
                st.dataframe(
                    high_risk_stops[['StopID', 'StopDesc', 'estimated_temp', 'heat_risk_score', 
                                   'risk_category', 'Shelter', 'Bench', 'Routes']].round(2),
                    use_container_width=True
                )
            
            # Data table (expandable)
            with st.expander("View All Bus Stops Data"):
                st.dataframe(bus_stops_with_risk.round(2), use_container_width=True)
                
                # Download button
                csv = bus_stops_with_risk.to_csv(index=False)
                st.download_button(
                    label="Download Bus Stops Risk Data as CSV",
                    data=csv,
                    file_name=f"bus_stops_heat_risk_{bus_time_period}.csv",
                    mime="text/csv"
                )
        
        else:
            st.error("No temperature data available for selected time period.")
    
    else:
        st.error("Unable to load bus stops data. Please check your internet connection.")

with tab3:
    st.header("🌡️ Temperature Forecasting")
    st.markdown("**Real-time weather data and area-wide temperature predictions**")
    
    # API key input
    api_key = st.text_input(
        "OpenWeatherMap API Key (optional - uses demo data if not provided):",
        type="password",
        help="Get your free API key from https://openweathermap.org/api"
    )
    
    forecaster = TemperatureForecastingSystem(api_key if api_key else None)
    
    # Forecast parameters
    col1, col2 = st.columns(2)
    with col1:
        forecast_hours = st.slider("Forecast Hours", 6, 72, 24)
    with col2:
        update_interval = st.selectbox(
            "Update Interval", 
            ["Real-time", "Every 30 min", "Every hour"], 
            index=1
        )
    
    # Fetch current weather
    with st.spinner("Fetching current weather data..."):
        current_weather = forecaster.fetch_current_weather_data()
    
    if not current_weather.empty:
        # Current conditions metrics
        st.subheader("Current Conditions")
        avg_temp = current_weather['temperature'].mean()
        avg_humidity = current_weather['humidity'].mean()
        avg_wind = current_weather['wind_speed'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Temperature", f"{avg_temp:.1f}°F")
        with col2:
            st.metric("Avg Humidity", f"{avg_humidity:.0f}%")
        with col3:
            st.metric("Avg Wind Speed", f"{avg_wind:.1f} mph")
        with col4:
            temp_range = current_weather['temperature'].max() - current_weather['temperature'].min()
            st.metric("Temperature Range", f"{temp_range:.1f}°F")
        
        # Current conditions map
        fig_current = px.scatter_mapbox(
            current_weather,
            lat='lat',
            lon='lng',
            color='temperature',
            size='temperature',
            size_max=20,
            color_continuous_scale='RdYlBu_r',
            hover_data=['name', 'humidity', 'wind_speed'],
            title='Current Temperature Conditions',
            mapbox_style="open-street-map",
            zoom=9,
            center=dict(lat=35.19, lon=-80.79)
        )
        
        fig_current.update_layout(height=500)
        st.plotly_chart(fig_current, use_container_width=True)
    
    # Fetch forecast
    with st.spinner("Generating forecast..."):
        forecast_data = forecaster.fetch_forecast_data(forecast_hours)
    
    if not forecast_data.empty:
        st.subheader("Temperature Forecast")
        
        # Forecast time series
        fig_forecast = px.line(
            forecast_data,
            x='datetime',
            y='temperature',
            title='Temperature Forecast - Charlotte Area',
            markers=True,
            line_shape='spline'
        )
        fig_forecast.update_layout(
            xaxis_title='Time',
            yaxis_title='Temperature (°F)',
            height=400
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly breakdown
            forecast_data['hour'] = forecast_data['datetime'].dt.hour
            hourly_avg = forecast_data.groupby('hour')['temperature'].mean().reset_index()
            
            fig_hourly = px.bar(
                hourly_avg,
                x='hour',
                y='temperature',
                title='Average Temperature by Hour',
                color='temperature',
                color_continuous_scale='RdYlBu_r'
            )
            fig_hourly.update_layout(height=300)
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Temperature vs humidity
            fig_humidity = px.scatter(
                forecast_data,
                x='humidity',
                y='temperature',
                color='wind_speed',
                size='clouds',
                title='Temperature vs Humidity',
                labels={'humidity': 'Humidity (%)', 'temperature': 'Temperature (°F)'}
            )
            fig_humidity.update_layout(height=300)
            st.plotly_chart(fig_humidity, use_container_width=True)
        
        # Heat alerts
        high_temp_forecasts = forecast_data[forecast_data['temperature'] > 90]
        if not high_temp_forecasts.empty:
            st.warning(f"🌡️ **Heat Alert**: {len(high_temp_forecasts)} forecast periods show temperatures above 90°F")
            
            with st.expander("View Heat Alert Details"):
                alert_data = high_temp_forecasts[['datetime', 'temperature', 'humidity', 'description']]
                alert_data['datetime'] = alert_data['datetime'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(alert_data)
        
        # Forecast table
        with st.expander("Detailed Forecast Data"):
            display_forecast = forecast_data.copy()
            display_forecast['datetime'] = display_forecast['datetime'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(display_forecast.round(2))

# Footer
st.markdown("---")
st.markdown("**🌡️ Heat Watch Explorer** • Comprehensive heat analysis for Charlotte, NC • Built with Streamlit & Plotly") 