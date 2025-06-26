# 🎵 Spotify Listening History Dashboard

An interactive dashboard built with Streamlit to visualize and analyze your Spotify listening history data.

## Features

### 📊 Overview Metrics
- Total listening hours
- Total plays
- Unique artists and tracks
- Listening activity timeline

### 📈 Visualizations
- **Listening Timeline**: Daily listening activity over time
- **Top Artists & Tracks**: Your most-played music ranked by listening time
- **Listening Patterns**: Activity by hour of day and day of week
- **Yearly Comparison**: Compare your listening habits across different years
- **Music Discovery**: Track when you discovered new music
- **Skip Analysis**: Analyze your skipping behavior and preferences
- **Artist Word Cloud**: Visual representation of your most-played artists

### 🎛️ Interactive Filters
- Date range selection
- Year filtering
- Top artists filtering
- Real-time data updates based on selections

## Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
# Create virtual environment
py -m venv spotify_venv

# Activate virtual environment (Windows)
spotify_venv\Scripts\Activate.ps1

# On macOS/Linux, use:
# source spotify_venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data
1. Request your Spotify data from [Spotify Privacy Settings](https://www.spotify.com/account/privacy/)
2. Wait for Spotify to send you your data (usually takes a few days)
3. Extract the ZIP file and place all `Streaming_History_Audio_*.json` files in the `raw_data/` folder

### 4. Run the Dashboard
```bash
streamlit run spotify_dashboard.py
```

The dashboard will automatically open in your web browser at `http://localhost:8501`

## Data Structure

The dashboard expects JSON files with the following structure:
```json
{
  "ts": "2025-02-24T17:55:17Z",
  "platform": "not_applicable",
  "ms_played": 142582,
  "conn_country": "US",
  "master_metadata_track_name": "Song Title",
  "master_metadata_album_artist_name": "Artist Name",
  "master_metadata_album_album_name": "Album Name",
  "spotify_track_uri": "spotify:track:...",
  "reason_start": "trackdone",
  "reason_end": "fwdbtn",
  "shuffle": false,
  "skipped": true,
  "offline": false,
  "incognito_mode": false
}
```

## Features Breakdown

### 📈 Listening Activity Timeline
- Interactive line chart showing daily listening hours
- Hover for detailed information
- Zoom and pan capabilities

### 🎤 Top Artists & Tracks
- Horizontal bar charts of your most-played content
- Ranked by total listening time (hours)
- Top 15 artists and tracks displayed

### ⏰ Listening Patterns
- **Hour of Day**: See when you listen to music most
- **Day of Week**: Discover your weekly listening patterns

### 🔍 Music Discovery Analysis
- Track when you first discovered new music
- Yearly breakdown of music discovery
- Helps identify periods of musical exploration

### ⏭️ Skip Analysis
- Artists with highest skip rates
- Skip rate trends over time
- Understand your listening preferences better

### ☁️ Artist Word Cloud
- Visual representation of your music taste
- Artist names sized by play count
- Beautiful, colorful display

## Tips for Best Results

1. **Large Datasets**: The dashboard uses caching to handle large datasets efficiently
2. **Date Filtering**: Use date range filters to focus on specific periods
3. **Performance**: For better performance with very large datasets, consider filtering by year first
4. **Data Quality**: The dashboard filters out plays shorter than 30 seconds for better accuracy

## Troubleshooting

### Common Issues
1. **File Not Found**: Ensure all JSON files are in the `raw_data/` directory
2. **Memory Issues**: If you have a very large dataset, try filtering by year or date range
3. **Slow Loading**: Large datasets may take a moment to load initially (cached after first load)

### Data Requirements
- JSON files must be named `Streaming_History_Audio_*.json`
- Files should contain valid JSON arrays
- Minimum data: timestamp, track name, artist name, and play duration

## Customization

The dashboard is highly customizable. You can:
- Modify color schemes in the Plotly configurations
- Add new visualization types
- Adjust filtering options
- Change the layout and styling

## Privacy

All data processing happens locally on your machine. No data is sent to external servers.

---

Happy exploring your music taste! 🎶 