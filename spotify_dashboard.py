import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import glob
import os
from datetime import datetime, timedelta
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Spotify Listening History Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Spotify-like styling
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    
    /* Main content area */
    .main > div {
        padding-top: 2rem;
        background-color: #121212;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #000000;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #181818;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1db954;
        color: #ffffff;
    }
    
    /* Headers and text */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Streamlit components */
    .stSelectbox > div > div > div {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #535353;
    }
    
    .stMultiSelect > div > div > div {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #535353;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1db954;
        color: #ffffff;
        border: none;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #1ed760;
    }
    
    /* Sidebar elements */
    .stSidebar > div {
        background-color: #000000;
    }
    
    .stSidebar .stSelectbox > div > div > div {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #535353;
    }
    
    /* Date input */
    .stDateInput > div > div > input {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #535353;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #181818;
        color: #ffffff;
        border: 1px solid #535353;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background-color: #1db954;
        color: #ffffff;
    }
    
    .stInfo {
        background-color: #2a2a2a;
        color: #ffffff;
        border-left: 4px solid #1db954;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #181818;
        color: #ffffff;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_spotify_data():
    """Load and combine all Spotify JSON files"""
    data_dir = "raw_data"
    json_files = glob.glob(os.path.join(data_dir, "Streaming_History_Audio_*.json"))
    
    all_data = []
    
    # Check if JSON files exist
    if not json_files:
        st.error("No Spotify data files found in the 'raw_data' directory.")
        st.info("Please upload your Spotify Extended Streaming History JSON files (Streaming_History_Audio_*.json)")
        st.stop()
    
    with st.spinner("Loading Spotify data..."):
        for file in json_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
    
    df = pd.DataFrame(all_data)
    
    # Data preprocessing
    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = df['ts'].dt.date
    df['hour'] = df['ts'].dt.hour
    df['day_of_week'] = df['ts'].dt.day_name()
    df['month'] = df['ts'].dt.month_name()
    df['year'] = df['ts'].dt.year
    df['minutes_played'] = df['ms_played'] / 60000
    df['hours_played'] = df['minutes_played'] / 60
    
    # Clean up track names and artist names
    df['track_name'] = df['master_metadata_track_name'].fillna('Unknown Track')
    df['artist_name'] = df['master_metadata_album_artist_name'].fillna('Unknown Artist')
    df['album_name'] = df['master_metadata_album_album_name'].fillna('Unknown Album')
    
    # Filter out very short plays (less than 30 seconds)
    df = df[df['ms_played'] >= 30000]
    
    return df

def create_overview_metrics(df):
    """Create overview metrics cards"""
    total_hours = df['hours_played'].sum()
    total_tracks = len(df)
    unique_artists = df['artist_name'].nunique()
    unique_tracks = df['track_name'].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Listening Hours",
            value=f"{total_hours:,.0f}",
            delta=f"{total_hours/24:.0f} days"
        )
    
    with col2:
        st.metric(
            label="Total Plays",
            value=f"{total_tracks:,}"
        )
    
    with col3:
        st.metric(
            label="Unique Artists",
            value=f"{unique_artists:,}"
        )
    
    with col4:
        st.metric(
            label="Unique Tracks",
            value=f"{unique_tracks:,}"
        )

def create_listening_timeline(df):
    """Create listening activity timeline with segmentation options"""
    st.subheader("📈 Listening Activity Over Time")
    
    # Add time segmentation options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        time_granularity = st.selectbox(
            "Time Grouping",
            options=["Daily", "Weekly", "Monthly"],
            index=0,
            key="timeline_granularity"
        )
    
    with col1:
        # Process data based on selected granularity
        if time_granularity == "Daily":
            grouped_data = df.groupby('date')['hours_played'].sum().reset_index()
            grouped_data['date'] = pd.to_datetime(grouped_data['date'])
            title = "Daily Listening Hours"
            x_label = "Date"
        elif time_granularity == "Weekly":
            df['week'] = df['ts'].dt.to_period('W').dt.start_time
            grouped_data = df.groupby('week')['hours_played'].sum().reset_index()
            grouped_data.rename(columns={'week': 'date'}, inplace=True)
            title = "Weekly Listening Hours"
            x_label = "Week"
        else:  # Monthly
            df['month'] = df['ts'].dt.to_period('M').dt.start_time
            grouped_data = df.groupby('month')['hours_played'].sum().reset_index()
            grouped_data.rename(columns={'month': 'date'}, inplace=True)
            title = "Monthly Listening Hours"
            x_label = "Month"
        
        fig = px.line(
            grouped_data, 
            x='date', 
            y='hours_played',
            title=title,
            labels={'hours_played': 'Hours Played', 'date': x_label},
            color_discrete_sequence=['#1db954']
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title=x_label,
            yaxis_title="Hours Played",
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font_color='#ffffff',
            title_font_color='#ffffff'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_top_artists_tracks(df):
    """Create top artists and tracks visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎤 Top Artists")
        all_artists = df.groupby('artist_name')['hours_played'].sum().sort_values(ascending=False)
        top_artists = all_artists.head(15)
        
        fig = px.bar(
            x=top_artists.values,
            y=top_artists.index,
            orientation='h',
            title="Top 15 Artists by Listening Time",
            labels={'x': 'Hours Played', 'y': 'Artist'},
            color_discrete_sequence=['#1db954']
        )
        fig.update_layout(
            height=500, 
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font_color='#ffffff',
            title_font_color='#ffffff'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Collapsible scrollable list for top 500 artists
        top_500_artists = all_artists.head(500)
        show_count = min(len(all_artists), 500)
        
        with st.expander(f"📋 View Top {show_count} Artists (of {len(all_artists)} total)"):
            st.write(f"**Top {show_count} Artist Rankings** (Total artists in dataset: {len(all_artists)})")
            
            # Create a container with maximum height and scrolling
            with st.container():
                # Create the scrollable content
                artist_data = []
                for rank, (artist, hours) in enumerate(top_500_artists.items(), 1):
                    plays = df[df['artist_name'] == artist].shape[0]
                    avg_per_play = (hours * 60) / plays if plays > 0 else 0
                    artist_data.append({
                        'Rank': f"#{rank}",
                        'Artist': artist,
                        'Hours': f"{hours:.1f}",
                        'Plays': f"{plays:,}",
                        'Avg/Play': f"{avg_per_play:.1f} min"
                    })
                
                # Display as a dataframe with custom styling
                if len(artist_data) > 0:
                    import pandas as pd
                    artist_df = pd.DataFrame(artist_data)
                    
                    # Use st.dataframe with height parameter for scrolling
                    st.dataframe(
                        artist_df,
                        use_container_width=True,
                        height=400,
                        hide_index=True
                    )
    
    with col2:
        st.subheader("🎵 Top Tracks")
        df['track_artist'] = df['track_name'] + " - " + df['artist_name']
        all_tracks = df.groupby('track_artist')['hours_played'].sum().sort_values(ascending=False)
        top_tracks = all_tracks.head(15)
        
        fig = px.bar(
            x=top_tracks.values,
            y=top_tracks.index,
            orientation='h',
            title="Top 15 Tracks by Listening Time",
            labels={'x': 'Hours Played', 'y': 'Track - Artist'},
            color_discrete_sequence=['#1ed760']
        )
        fig.update_layout(
            height=500, 
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font_color='#ffffff',
            title_font_color='#ffffff'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Collapsible scrollable list for top 500 tracks
        top_500_tracks = all_tracks.head(500)
        show_count_tracks = min(len(all_tracks), 500)
        
        with st.expander(f"📋 View Top {show_count_tracks} Tracks (of {len(all_tracks)} total)"):
            st.write(f"**Top {show_count_tracks} Track Rankings** (Total tracks in dataset: {len(all_tracks)})")
            
            # Create a container with maximum height and scrolling
            with st.container():
                # Create the scrollable content
                track_data = []
                for rank, (track_artist, hours) in enumerate(top_500_tracks.items(), 1):
                    plays = df[df['track_artist'] == track_artist].shape[0]
                    avg_per_play = (hours * 60) / plays if plays > 0 else 0
                    
                    # Split track_artist back to track and artist
                    if " - " in track_artist:
                        track_name = track_artist.rsplit(" - ", 1)[0]
                        artist_name = track_artist.rsplit(" - ", 1)[1]
                    else:
                        track_name = track_artist
                        artist_name = "Unknown"
                    
                    track_data.append({
                        'Rank': f"#{rank}",
                        'Track': track_name,
                        'Artist': artist_name,
                        'Hours': f"{hours:.1f}",
                        'Plays': f"{plays:,}",
                        'Avg/Play': f"{avg_per_play:.1f} min"
                    })
                
                # Display as a dataframe with custom styling
                if len(track_data) > 0:
                    import pandas as pd
                    track_df = pd.DataFrame(track_data)
                    
                    # Use st.dataframe with height parameter for scrolling
                    st.dataframe(
                        track_df,
                        use_container_width=True,
                        height=400,
                        hide_index=True
                    )

def create_listening_patterns(df):
    """Create listening pattern visualizations with enhanced options"""
    st.subheader("⏰ Listening Patterns")
    
    # Pattern analysis options
    col_control, col_space = st.columns([2, 3])
    with col_control:
        pattern_view = st.selectbox(
            "Pattern View",
            options=["Hour & Day", "Heatmap", "Monthly Patterns"],
            index=0,
            key="pattern_view"
        )
    
    if pattern_view == "Hour & Day":
        col1, col2 = st.columns(2)
        
        with col1:
            # Hour of day pattern
            hourly_listening = df.groupby('hour')['hours_played'].sum()
            
            fig = px.bar(
                x=hourly_listening.index,
                y=hourly_listening.values,
                title="Listening Activity by Hour of Day",
                labels={'x': 'Hour of Day', 'y': 'Hours Played'},
                color_discrete_sequence=['#1db954']
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='#121212',
                paper_bgcolor='#121212',
                font_color='#ffffff',
                title_font_color='#ffffff'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week pattern
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_listening = df.groupby('day_of_week')['hours_played'].sum().reindex(day_order)
            
            fig = px.bar(
                x=daily_listening.index,
                y=daily_listening.values,
                title="Listening Activity by Day of Week",
                labels={'x': 'Day of Week', 'y': 'Hours Played'},
                color_discrete_sequence=['#1ed760']
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='#121212',
                paper_bgcolor='#121212',
                font_color='#ffffff',
                title_font_color='#ffffff'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif pattern_view == "Heatmap":
        # Create heatmap of hour vs day of week
        df['hour_day'] = df['hour'].astype(str) + ':00'
        heatmap_data = df.groupby(['day_of_week', 'hour'])['hours_played'].sum().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='hours_played').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(day_order)
        
        fig = px.imshow(
            heatmap_pivot,
            title="Listening Activity Heatmap (Day vs Hour)",
            labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Hours Played'},
            color_continuous_scale=[[0, '#121212'], [0.5, '#1db954'], [1, '#1ed760']]
        )
        fig.update_layout(
            height=500,
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font_color='#ffffff',
            title_font_color='#ffffff'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Monthly Patterns
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly listening
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_listening = df.groupby('month')['hours_played'].sum().reindex(month_order)
            
            fig = px.bar(
                x=monthly_listening.index,
                y=monthly_listening.values,
                title="Listening Activity by Month",
                labels={'x': 'Month', 'y': 'Hours Played'},
                color_discrete_sequence=['#1db954']
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='#121212',
                paper_bgcolor='#121212',
                font_color='#ffffff',
                title_font_color='#ffffff'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Year-over-year monthly comparison
            df['year_month'] = df['ts'].dt.strftime('%Y-%m')
            monthly_yearly = df.groupby(['year', 'month'])['hours_played'].sum().reset_index()
            
            fig = px.line(
                monthly_yearly,
                x='month',
                y='hours_played',
                color='year',
                title="Monthly Listening by Year",
                labels={'hours_played': 'Hours Played', 'month': 'Month'},
                color_discrete_sequence=['#1db954', '#1ed760', '#ff6b35', '#f7931e', '#c13584']
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='#121212',
                paper_bgcolor='#121212',
                font_color='#ffffff',
                title_font_color='#ffffff'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

def create_discovery_analysis(df):
    """Analyze music discovery patterns"""
    st.subheader("🔍 Music Discovery Analysis")
    
    # Calculate first play date for each track
    first_plays = df.groupby(['track_name', 'artist_name'])['ts'].min().reset_index()
    first_plays['year'] = first_plays['ts'].dt.year
    
    # Count new tracks discovered each year
    discoveries_by_year = first_plays.groupby('year').size()
    
    fig = px.bar(
        x=discoveries_by_year.index,
        y=discoveries_by_year.values,
        title="New Tracks Discovered by Year",
        labels={'x': 'Year', 'y': 'New Tracks Discovered'},
        color_discrete_sequence=['#1db954']
    )
    fig.update_layout(
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        font_color='#ffffff',
        title_font_color='#ffffff'
    )
    st.plotly_chart(fig, use_container_width=True)

def create_skip_analysis(df):
    """Analyze skipping behavior"""
    st.subheader("⏭️ Skip Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Skip rate by artist
        artist_stats = df.groupby('artist_name').agg({
            'skipped': ['count', 'sum']
        }).round(2)
        artist_stats.columns = ['total_plays', 'skips']
        artist_stats['skip_rate'] = (artist_stats['skips'] / artist_stats['total_plays'] * 100).round(1)
        artist_stats = artist_stats[artist_stats['total_plays'] >= 10]  # Only artists with 10+ plays
        
        top_skip_artists = artist_stats.sort_values('skip_rate', ascending=False).head(10)
        
        fig = px.bar(
            x=top_skip_artists['skip_rate'],
            y=top_skip_artists.index,
            orientation='h',
            title="Artists with Highest Skip Rates (10+ plays)",
            labels={'x': 'Skip Rate (%)', 'y': 'Artist'},
            color_discrete_sequence=['#ff6b35']
        )
        fig.update_layout(
            height=400, 
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font_color='#ffffff',
            title_font_color='#ffffff'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Skip rate over time with segmentation options
        time_period = st.selectbox(
            "Time Period",
            options=["Monthly", "Weekly", "Daily"],
            index=0,
            key="skip_time_period"
        )
        
        df['skip_rate'] = df['skipped'].astype(int)
        
        if time_period == "Monthly":
            df['time_period'] = df['ts'].dt.to_period('M')
            title = "Skip Rate Over Time (Monthly Average)"
        elif time_period == "Weekly":
            df['time_period'] = df['ts'].dt.to_period('W')
            title = "Skip Rate Over Time (Weekly Average)"
        else:  # Daily
            df['time_period'] = df['ts'].dt.to_period('D')
            title = "Skip Rate Over Time (Daily Average)"
        
        skip_data = df.groupby('time_period').agg({
            'skip_rate': 'mean'
        }).reset_index()
        skip_data['time_period'] = skip_data['time_period'].dt.to_timestamp()
        skip_data['skip_rate'] = skip_data['skip_rate'] * 100
        
        # Limit data points for readability
        if time_period == "Daily":
            skip_data = skip_data.tail(365)  # Last year
        elif time_period == "Weekly":
            skip_data = skip_data.tail(104)  # Last 2 years
        
        fig = px.line(
            skip_data,
            x='time_period',
            y='skip_rate',
            title=title,
            labels={'skip_rate': 'Skip Rate (%)', 'time_period': 'Date'},
            color_discrete_sequence=['#ff6b35']
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font_color='#ffffff',
            title_font_color='#ffffff'
        )
        st.plotly_chart(fig, use_container_width=True)

def create_stacked_area_charts(df):
    """Create stacked area charts for top artists and tracks over time"""
    st.subheader("📊 Listening Trends Over Time")
    
    # Controls for the charts
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            options=["Artists", "Tracks"],
            index=0,
            key="stacked_chart_type"
        )
    
    with col2:
        top_n = st.selectbox(
            f"Top {chart_type}",
            options=[5, 10, 15, 20],
            index=1,
            key="stacked_top_n"
        )
    
    with col3:
        time_period = st.selectbox(
            "Time Period",
            options=["Monthly", "Weekly"],
            index=0,
            key="stacked_time_period"
        )
    
    with col4:
        include_other = st.checkbox(
            "Include 'Other'",
            value=True,
            key="stacked_include_other",
            help="Group remaining items as 'Other'"
        )
    
    # Process data based on selections
    if chart_type == "Artists":
        group_col = 'artist_name'
        title_base = f"Top {top_n} Artists"
    else:
        group_col = 'track_name'
        title_base = f"Top {top_n} Tracks"
    
    # Get top artists/tracks by total listening time
    top_items = df.groupby(group_col)['hours_played'].sum().sort_values(ascending=False).head(top_n)
    top_item_names = top_items.index.tolist()
    
    # Prepare data for time series analysis
    df_copy = df.copy()
    
    # Group by time period first
    if time_period == "Monthly":
        df_copy['time_period'] = df_copy['ts'].dt.to_period('M')
        period_label = "Month"
    else:  # Weekly
        df_copy['time_period'] = df_copy['ts'].dt.to_period('W')
        period_label = "Week"
        # Limit to last 52 weeks for readability
        recent_periods = df_copy['time_period'].drop_duplicates().sort_values().tail(52)
        df_copy = df_copy[df_copy['time_period'].isin(recent_periods)]
    
    # Handle "Other" category
    if include_other:
        # Create "Other" category for items not in top N
        df_copy[group_col] = df_copy[group_col].apply(
            lambda x: x if x in top_item_names else "Other"
        )
        # Update title
        title = f"{title_base} + Other - 100% Stacked Listening Trends"
        # Get all unique items including "Other"
        all_items = top_item_names + ["Other"]
    else:
        # Filter to only top items
        df_copy = df_copy[df_copy[group_col].isin(top_item_names)]
        title = f"{title_base} - 100% Stacked Listening Trends"
        all_items = top_item_names
    
    # Create time series data
    time_series_data = df_copy.groupby(['time_period', group_col])['hours_played'].sum().reset_index()
    time_series_data['time_period'] = time_series_data['time_period'].dt.to_timestamp()
    
    # Create pivot table for stacked area chart
    pivot_data = time_series_data.pivot(index='time_period', columns=group_col, values='hours_played').fillna(0)
    
    # Calculate percentages for 100% stacked chart
    pivot_data_pct = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
    
    # Reorder columns by total listening time (descending) - "Other" goes last
    if include_other and "Other" in pivot_data_pct.columns:
        # Sort top items by total listening time, put "Other" last
        non_other_cols = [col for col in top_item_names if col in pivot_data_pct.columns]
        column_order = non_other_cols + ["Other"]
    else:
        # Just sort by total listening time
        column_order = [col for col in top_item_names if col in pivot_data_pct.columns]
    
    pivot_data_pct = pivot_data_pct[column_order]
    
    # Create 100% stacked area chart
    fig = go.Figure()
    
    # Define Spotify-inspired colors
    spotify_colors = [
        '#1db954', '#1ed760', '#ff6b35', '#f7931e', '#c13584',
        '#00d4ff', '#ff9500', '#8b5a3c', '#6a994e', '#bc4749',
        '#577590', '#f8961e', '#90e0ef', '#ffd166', '#06ffa5',
        '#ff006e', '#8338ec', '#3a86ff', '#06ffa5', '#fb8500'
    ]
    colors = spotify_colors * 10  # Repeat colors if needed
    
    # Add traces for each artist/track (in reverse order for proper stacking and legend display)
    for i, item in enumerate(reversed(column_order)):
        # Use gray for "Other" category
        color = "#535353" if item == "Other" else colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=pivot_data_pct.index,
            y=pivot_data_pct[item],
            mode='lines',
            stackgroup='one',
            name=item,
            line=dict(width=0.5),
            fillcolor=color,
            hovertemplate=f'<b>{item}</b><br>' +
                         f'{period_label}: %{{x}}<br>' +
                         'Percentage: %{y:.1f}%<br>' +
                         f'Hours: %{{customdata:.1f}}<br>' +
                         '<extra></extra>',
            customdata=pivot_data[item]  # Add actual hours for hover
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=period_label,
        yaxis_title="Percentage of Listening Time (%)",
        height=500,
        hovermode='x unified',
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        font_color='#ffffff',
        title_font_color='#ffffff',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            traceorder="normal",  # This will show legend in the order we added traces
            bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff'
        ),
        yaxis=dict(
            range=[0, 100],
            ticksuffix="%"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    with st.expander(f"📈 {chart_type} Trend Summary"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total listening time for top {top_n} {chart_type.lower()}:**")
            for item, hours in top_items.head(top_n).items():
                percentage = (hours / top_items.sum()) * 100
                st.write(f"• {item}: {hours:.1f} hours ({percentage:.1f}%)")
            
            if include_other:
                # Calculate "Other" total
                other_total = pivot_data["Other"].sum() if "Other" in pivot_data.columns else 0
                other_percentage = (other_total / pivot_data.sum().sum()) * 100
                st.write(f"• Other: {other_total:.1f} hours ({other_percentage:.1f}%)")
        
        with col2:
            # Calculate trend information
            latest_period = pivot_data.index.max()
            earliest_period = pivot_data.index.min()
            
            st.write(f"**Period analyzed:** {earliest_period.strftime('%Y-%m-%d')} to {latest_period.strftime('%Y-%m-%d')}")
            
            # Show most active period
            period_totals = pivot_data.sum(axis=1)
            most_active_period = period_totals.idxmax()
            st.write(f"**Most active {period_label.lower()}:** {most_active_period.strftime('%Y-%m-%d')} ({period_totals.max():.1f} hours)")
            
            # Show average percentages in latest period
            if len(pivot_data_pct) > 0:
                latest_data = pivot_data_pct.iloc[-1]
                top_in_latest = latest_data.nlargest(3)
                st.write("**Latest period top 3:**")
                for item, pct in top_in_latest.items():
                    st.write(f"• {item}: {pct:.1f}%")

def create_artist_wordcloud(df):
    """Create word cloud of artists"""
    st.subheader("☁️ Artist Word Cloud")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        wordcloud_metric = st.selectbox(
            "Size by",
            options=["Play Count", "Listening Hours"],
            index=0,
            key="wordcloud_metric"
        )
        
        max_artists = st.selectbox(
            "Max Artists",
            options=[50, 75, 100, 150],
            index=2,
            key="wordcloud_max_artists"
        )
    
    with col1:
        # Get artist data based on selected metric
        if wordcloud_metric == "Play Count":
            artist_data = df['artist_name'].value_counts().head(max_artists)
            subtitle = f"Top {max_artists} Artists by Play Count"
        else:
            artist_data = df.groupby('artist_name')['hours_played'].sum().sort_values(ascending=False).head(max_artists)
            subtitle = f"Top {max_artists} Artists by Listening Hours"
        
        # Prepare frequencies dictionary for WordCloud
        # Replace spaces with underscores to keep artist names intact
        frequencies = {}
        for artist, value in artist_data.items():
            # Clean artist name and replace spaces with underscores
            clean_artist = artist.replace(' ', '_').replace('-', '_').replace('.', '_')
            frequencies[clean_artist] = float(value)
        
        if frequencies:
            # Create word cloud using frequencies with Spotify colors
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='#121212',
                colormap='Greens',
                max_words=max_artists,
                relative_scaling=0.5,
                min_font_size=8,
                prefer_horizontal=0.8,
                color_func=lambda *args, **kwargs: '#1db954'
            ).generate_from_frequencies(frequencies)
            
            # Display using matplotlib with dark background
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='#121212')
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(subtitle, fontsize=14, pad=20, color='#ffffff')
            ax.set_facecolor('#121212')
            st.pyplot(fig)
            
            # Show top 10 in text format below
            with st.expander("📋 Top 10 Details"):
                st.write(f"**Top 10 by {wordcloud_metric}:**")
                for i, (artist, value) in enumerate(artist_data.head(10).items(), 1):
                    if wordcloud_metric == "Play Count":
                        st.write(f"{i}. {artist}: {value:,} plays")
                    else:
                        st.write(f"{i}. {artist}: {value:.1f} hours")
        else:
            st.warning("No artist data available for the selected time period.")

def create_time_based_analysis(df):
    """Create flexible time-based analysis with segmentation options"""
    st.subheader("📅 Time-Based Analysis")
    
    # Add segmentation options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            options=["Yearly", "Monthly", "Weekly"],
            index=0,
            key="time_analysis_type"
        )
    
    with col1:
        # Process data based on selected analysis type
        if analysis_type == "Yearly":
            time_stats = df.groupby('year').agg({
                'hours_played': 'sum',
                'track_name': 'count',
                'artist_name': 'nunique'
            }).round(2)
            time_stats.columns = ['Total Hours', 'Total Plays', 'Unique Artists']
            x_title = "Year"
            
        elif analysis_type == "Monthly":
            df['year_month'] = df['ts'].dt.to_period('M')
            time_stats = df.groupby('year_month').agg({
                'hours_played': 'sum',
                'track_name': 'count',
                'artist_name': 'nunique'
            }).round(2)
            time_stats.columns = ['Total Hours', 'Total Plays', 'Unique Artists']
            time_stats.index = time_stats.index.astype(str)
            x_title = "Month"
            
        else:  # Weekly
            df['year_week'] = df['ts'].dt.to_period('W')
            time_stats = df.groupby('year_week').agg({
                'hours_played': 'sum',
                'track_name': 'count',
                'artist_name': 'nunique'
            }).round(2)
            time_stats.columns = ['Total Hours', 'Total Plays', 'Unique Artists']
            time_stats.index = time_stats.index.astype(str)
            x_title = "Week"
            # Limit to last 52 weeks for readability
            time_stats = time_stats.tail(52)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Total Hours', 'Total Plays', 'Unique Artists')
        )
        
        fig.add_trace(
            go.Bar(x=time_stats.index, y=time_stats['Total Hours'], name='Total Hours', marker_color='#1db954'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=time_stats.index, y=time_stats['Total Plays'], name='Total Plays', marker_color='#1ed760'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=time_stats.index, y=time_stats['Unique Artists'], name='Unique Artists', marker_color='#ff6b35'),
            row=1, col=3
        )
        
        fig.update_layout(
            height=400, 
            showlegend=False,
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font_color='#ffffff',
            title_font_color='#ffffff'
        )
        fig.update_xaxes(title_text=x_title)
        
        # Rotate x-axis labels if needed
        if analysis_type in ["Monthly", "Weekly"]:
            fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.title("🎵 Spotify Listening History Dashboard")
    st.markdown("---")
    
    # Load data
    try:
        df = load_spotify_data()
        st.success(f"Successfully loaded {len(df):,} listening records from {df['ts'].min().strftime('%Y-%m-%d')} to {df['ts'].max().strftime('%Y-%m-%d')}")
        
        # Sidebar filters
        st.sidebar.header("🎛️ Filters")
        
        # Enhanced date range filter
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        st.sidebar.subheader("📅 Date Range")
        
        # Get available years from the data
        available_years = sorted(df['year'].unique(), reverse=True)
        year_options = [f"{year}" for year in available_years]
        
        # Quick date range presets
        preset_options = [
            "All Time",
            "Last Year", 
            "Last 6 Months",
            "Last 3 Months", 
            "Last Month",
            "This Year"
        ] + year_options + ["Custom Range"]
        
        preset_range = st.sidebar.selectbox(
            "Quick Select",
            options=preset_options,
            index=0
        )
        
        # Calculate preset date ranges
        today = pd.Timestamp.now().date()
        current_year_start = pd.Timestamp(today.year, 1, 1).date()
        
        if preset_range == "All Time":
            default_start, default_end = min_date, max_date
        elif preset_range == "Last Year":
            default_start = max(min_date, today - pd.Timedelta(days=365))
            default_end = max_date
        elif preset_range == "Last 6 Months":
            default_start = max(min_date, today - pd.Timedelta(days=180))
            default_end = max_date
        elif preset_range == "Last 3 Months":
            default_start = max(min_date, today - pd.Timedelta(days=90))
            default_end = max_date
        elif preset_range == "Last Month":
            default_start = max(min_date, today - pd.Timedelta(days=30))
            default_end = max_date
        elif preset_range == "This Year":
            default_start = max(min_date, current_year_start)
            default_end = max_date
        elif preset_range.isdigit():  # Individual year selection
            selected_year = int(preset_range)
            year_start = pd.Timestamp(selected_year, 1, 1).date()
            year_end = pd.Timestamp(selected_year, 12, 31).date()
            default_start = max(min_date, year_start)
            default_end = min(max_date, year_end)
        else:  # Custom Range
            default_start, default_end = min_date, max_date
        
        # Custom date range selector (always shown but populated based on preset)
        date_range = st.sidebar.date_input(
            "Select Custom Date Range" if preset_range == "Custom Range" else "Selected Range",
            value=(default_start, default_end),
            min_value=min_date,
            max_value=max_date,
            disabled=(preset_range != "Custom Range")
        )
        
        # Filter data based on date range
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        else:
            df_filtered = df
        
        # Show selected date range info
        if len(date_range) == 2:
            total_days = (date_range[1] - date_range[0]).days + 1
            st.sidebar.info(f"📊 Selected: {total_days} days\n({date_range[0]} to {date_range[1]})")
        
        # Artist filter
        top_artists = df['artist_name'].value_counts().head(20).index.tolist()
        selected_artists = st.sidebar.multiselect(
            "Filter by Top Artists (optional)",
            top_artists
        )
        
        if selected_artists:
            df_filtered = df_filtered[df_filtered['artist_name'].isin(selected_artists)]
        
        # Display overview metrics
        create_overview_metrics(df_filtered)
        st.markdown("---")
        
        # Create visualizations
        create_listening_timeline(df_filtered)
        st.markdown("---")
        
        create_top_artists_tracks(df_filtered)
        st.markdown("---")
        
        create_listening_patterns(df_filtered)
        st.markdown("---")
        
        create_time_based_analysis(df_filtered)
        st.markdown("---")
        
        create_discovery_analysis(df_filtered)
        st.markdown("---")
        
        create_skip_analysis(df_filtered)
        st.markdown("---")
        
        create_stacked_area_charts(df_filtered)
        st.markdown("---")
        
        create_artist_wordcloud(df_filtered)
        
        # Data table
        if st.checkbox("Show Raw Data"):
            st.subheader("📊 Raw Data")
            st.dataframe(df_filtered[['ts', 'track_name', 'artist_name', 'album_name', 'minutes_played', 'skipped']].head(1000))
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please make sure your JSON files are in the 'raw_data' directory.")

if __name__ == "__main__":
    main() 