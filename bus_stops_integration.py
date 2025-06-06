import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from scipy.spatial.distance import cdist

class BusStopsTemperatureAnalyzer:
    def __init__(self):
        self.bus_stops_url = "https://gis.charlottenc.gov/arcgis/rest/services/HNS/HousingLocationalToolLayers/MapServer/16/query"
        self.weather_api_key = None  # User would need to provide
    
    @st.cache_data
    def fetch_bus_stops(self):
        """Fetch bus stops data from Charlotte ArcGIS REST API"""
        params = {
            'where': '1=1',  # Get all records
            'outFields': '*',
            'f': 'json',
            'returnGeometry': 'true'
        }
        
        try:
            response = requests.get(self.bus_stops_url, params=params)
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
                        'Frequency_AM_Peak': attrs.get('Frequency_AM_Peak', 0),
                        'Frequency_Midday': attrs.get('Frquency_Midday', 0),
                        'Frequency_PM_Peak': attrs.get('Frequency_PM_Peak', 0),
                        'Frequency_Evening': attrs.get('Frequency_Evening', 0),
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
    
    def create_risk_summary_charts(self, bus_stops_df):
        """Create summary charts for risk analysis"""
        if bus_stops_df.empty:
            return go.Figure(), go.Figure()
        
        # Risk category distribution
        risk_counts = bus_stops_df['risk_category'].value_counts()
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
        
        # Infrastructure vs. Risk
        fig_scatter = px.scatter(
            bus_stops_df,
            x='estimated_temp',
            y='heat_risk_score',
            color='Shelter',
            symbol='Bench',
            size='Frequency_AM_Peak',
            hover_data=['StopID', 'StopDesc'],
            title='Temperature vs Heat Risk Score (by Infrastructure)',
            labels={
                'estimated_temp': 'Estimated Temperature (°F)',
                'heat_risk_score': 'Heat Risk Score'
            }
        )
        
        return fig_pie, fig_scatter

# Streamlit app integration example
def create_bus_stops_tab():
    """Create a new tab for bus stops analysis"""
    st.subheader("🚌 Bus Stop Heat Risk Analysis")
    
    analyzer = BusStopsTemperatureAnalyzer()
    
    # Fetch bus stops data
    with st.spinner("Loading bus stops data..."):
        bus_stops_df = analyzer.fetch_bus_stops()
    
    if not bus_stops_df.empty:
        st.success(f"Loaded {len(bus_stops_df)} bus stops")
        
        # Time period selection
        time_period = st.selectbox(
            "Select Time Period:",
            options=['am', 'af', 'pm'],
            format_func=lambda x: {'am': 'Morning', 'af': 'Afternoon', 'pm': 'Evening'}[x]
        )
        
        # Load corresponding temperature data (from existing census analysis)
        # This would integrate with your existing load_census_data function
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Bus Stops", len(bus_stops_df))
        with col2:
            sheltered = bus_stops_df['Shelter'].sum()
            st.metric("Stops with Shelter", f"{sheltered} ({sheltered/len(bus_stops_df)*100:.1f}%)")
        with col3:
            high_freq = len(bus_stops_df[bus_stops_df['Frequency_AM_Peak'] > 4])
            st.metric("High Frequency Stops", high_freq)
        with col4:
            avg_temp = bus_stops_df['estimated_temp'].mean() if 'estimated_temp' in bus_stops_df.columns else 0
            st.metric("Avg. Estimated Temp", f"{avg_temp:.1f}°F")
        
        # Display map and charts
        if 'estimated_temp' in bus_stops_df.columns:
            fig_map = analyzer.create_bus_stops_map(bus_stops_df, time_period)
            st.plotly_chart(fig_map, use_container_width=True)
            
            col1, col2 = st.columns(2)
            fig_pie, fig_scatter = analyzer.create_risk_summary_charts(bus_stops_df)
            
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Data table
        with st.expander("View Bus Stops Data"):
            st.dataframe(bus_stops_df)
    
    else:
        st.error("Unable to load bus stops data")

if __name__ == "__main__":
    create_bus_stops_tab() 