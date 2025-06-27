# 🎵 Spotify Listening History Dashboard

An interactive dashboard built with Streamlit to visualize and analyze your Spotify listening history data.

## Features

### 🤖 AI-Powered Chat Analysis (NEW!)
- **Conversational Data Analysis**: Ask questions about your music data in natural language
- **Smart Visualizations**: AI generates custom charts and insights on demand
- **Context-Aware**: Understands your current filters and data scope
- **Real-time Code Execution**: Generates and runs Python/Plotly code to answer your questions
- **Example queries**: "What's my most played song?", "Show my listening patterns by hour", "How has my music taste changed?"

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
- **Stacked Area Charts**: Timeline view of your top artists/tracks over time

### 🎛️ Interactive Filters
- Date range selection with quick presets (Last Year, Last 6 Months, etc.)
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

### 4. Setup AI Chat (Optional)
To enable the AI-powered conversational analysis:

1. **Get a Gemini API Key** (free with generous limits):
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy your API key

2. **Configure the API Key**:
   
   **For Local Development:**
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Replace `your-gemini-api-key-here` with your actual API key
   
   **For Streamlit Cloud Deployment:**
   - Go to your app settings in Streamlit Cloud
   - Navigate to "Secrets"
   - Add: `GEMINI_API_KEY = "your-api-key-here"`

3. **Manual Setup (Testing):**
   - The dashboard also allows manual API key entry through the sidebar interface

### 5. Run the Dashboard
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

### 🤖 AI Chat Features
- **Natural Language Queries**: Ask questions like "What's my top artist this year?"
- **Dynamic Visualizations**: AI creates custom charts based on your questions
- **Data Context Awareness**: Understands your current filters and time periods
- **Conversation Memory**: Maintains context throughout your analysis session
- **Code Generation**: Shows the Python code it generates for transparency
- **Error Handling**: Graceful handling of complex queries and edge cases

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
4. **AI Chat Not Working**: 
   - Check that your Gemini API key is correctly configured
   - Verify you have internet connection for API calls
   - Try the manual API key entry option in the sidebar
5. **AI Code Errors**: The AI generates code dynamically; some complex queries may need refinement

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

## Privacy & Security

### Data Processing
- All core dashboard features process data locally on your machine
- No personal data is sent to external servers for the main dashboard functionality

### AI Chat Feature
- **Data Summaries Only**: The AI chat sends only statistical summaries and metadata (not raw listening data)
- **No Personal Details**: Track names, artist names, and listening patterns are aggregated before sending
- **API Security**: Uses Google's Gemini API with industry-standard security practices
- **Optional Feature**: AI chat can be completely disabled if preferred
- **Code Transparency**: All generated code is visible before execution

### What Data is Shared (AI Chat Only)
- Summary statistics (total hours, play counts, top artists)
- Dataframe structure and column information
- Aggregated insights (no individual track plays)
- **Never shared**: Raw timestamps, detailed listening history, personal identifiers

---

Happy exploring your music taste! 🎶 