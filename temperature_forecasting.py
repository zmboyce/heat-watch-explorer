import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class TemperatureForecastingSystem:
    """
    Temperature forecasting system that integrates with weather APIs
    and provides area-wide temperature predictions for Charlotte, NC
    """
    
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
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_current_weather_data(self):
        """Fetch current weather data from multiple points around Charlotte"""
        if not self.api_key:
            # Return mock data for demonstration
            return self._generate_mock_weather_data()
        
        weather_data = []
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        
        for station in self.weather_stations:
            try:
                params = {
                    'lat': station['lat'],
                    'lon': station['lng'],
                    'appid': self.api_key,
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
    
    @st.cache_data(ttl=3600)
    def fetch_forecast_data(self, hours=24):
        """Fetch weather forecast data"""
        if not self.api_key:
            return self._generate_mock_forecast_data(hours)
        
        forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
        
        try:
            params = {
                'lat': self.charlotte_coords['lat'],
                'lon': self.charlotte_coords['lng'],
                'appid': self.api_key,
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
    
    def create_temperature_surface(self, weather_df, grid_resolution=50):
        """Create interpolated temperature surface for the Charlotte area"""
        if weather_df.empty:
            return None, None, None
        
        # Define grid bounds (Charlotte metro area)
        lat_min, lat_max = 35.0, 35.4
        lng_min, lng_max = -81.1, -80.5
        
        # Create grid
        lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
        lng_grid = np.linspace(lng_min, lng_max, grid_resolution)
        lat_mesh, lng_mesh = np.meshgrid(lat_grid, lng_grid)
        
        # Interpolate temperature values
        points = weather_df[['lat', 'lng']].values
        values = weather_df['temperature'].values
        
        # Use griddata for spatial interpolation
        temp_grid = griddata(
            points, values, 
            (lat_mesh, lng_mesh), 
            method='cubic',
            fill_value=values.mean()
        )
        
        return lat_mesh, lng_mesh, temp_grid
    
    def create_forecast_surface_animation(self, forecast_df):
        """Create animated forecast surface"""
        if forecast_df.empty:
            return go.Figure()
        
        # Group by hour for animation
        hourly_data = []
        for i in range(0, len(forecast_df), 3):  # Every 3 hours
            hour_data = forecast_df.iloc[i:i+1]
            if not hour_data.empty:
                hourly_data.append({
                    'hour': i//3,
                    'datetime': hour_data.iloc[0]['datetime'],
                    'temperature': hour_data.iloc[0]['temperature']
                })
        
        # Create simple time series for now (can be enhanced to spatial animation)
        times = [d['datetime'] for d in hourly_data]
        temps = [d['temperature'] for d in hourly_data]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=temps,
            mode='lines+markers',
            name='Forecast Temperature',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Temperature Forecast - Charlotte Area',
            xaxis_title='Time',
            yaxis_title='Temperature (°F)',
            height=400,
            margin=dict(r=0, t=40, l=0, b=0)
        )
        
        return fig
    
    def train_temperature_model(self, historical_temp_data, weather_features):
        """Train ML model for temperature prediction using historical data"""
        if len(historical_temp_data) < 50:  # Need sufficient data
            return None
        
        # Prepare features
        features = ['humidity', 'pressure', 'wind_speed', 'clouds', 'hour', 'day_of_year']
        
        # Create time-based features
        weather_features['hour'] = weather_features['timestamp'].dt.hour
        weather_features['day_of_year'] = weather_features['timestamp'].dt.dayofyear
        
        X = weather_features[features].fillna(0)
        y = weather_features['temperature']
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model
    
    def predict_temperature_with_ml(self, model, forecast_features):
        """Use trained ML model to predict temperatures"""
        if model is None:
            return forecast_features['temperature']  # Fallback to API forecast
        
        # Prepare features
        features = ['humidity', 'pressure', 'wind_speed', 'clouds', 'hour', 'day_of_year']
        forecast_features['hour'] = forecast_features['datetime'].dt.hour
        forecast_features['day_of_year'] = forecast_features['datetime'].dt.dayofyear
        
        X_pred = forecast_features[features].fillna(0)
        predictions = model.predict(X_pred)
        
        return predictions

def create_forecasting_tab():
    """Create Streamlit tab for temperature forecasting"""
    st.subheader("🌡️ Temperature Forecasting")
    
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
        
        # Forecast visualization
        fig_forecast = forecaster.create_forecast_surface_animation(forecast_data)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature trend
            fig_trend = px.line(
                forecast_data,
                x='datetime',
                y='temperature',
                title='Temperature Trend',
                markers=True
            )
            fig_trend.update_layout(height=300)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
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
        
        # Forecast table
        with st.expander("Detailed Forecast Data"):
            display_forecast = forecast_data.copy()
            display_forecast['datetime'] = display_forecast['datetime'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(display_forecast.round(2))

if __name__ == "__main__":
    create_forecasting_tab() 