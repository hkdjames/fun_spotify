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
import google.generativeai as genai
import re
import traceback
import io

# Set page config
st.set_page_config(
    page_title="Spotify Listening History Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default Streamlit styling (no custom CSS)

# Initialize session state for chat
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None
if "data_context_hash" not in st.session_state:
    st.session_state.data_context_hash = None

def setup_gemini_api():
    """Setup Gemini API with error handling"""
    try:
        # Try to get API key from Streamlit secrets
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        elif "GEMINI_API_KEY" in os.environ:
            api_key = os.environ["GEMINI_API_KEY"]
        else:
            return None, "API key not found in Streamlit secrets"
        
        if not api_key or api_key == "your-gemini-api-key-here":
            return None, "Please replace the placeholder API key with your actual Gemini API key"
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model, None
    except Exception as e:
        return None, f"Error setting up Gemini API: {str(e)}"

def get_data_context(df, df_filtered, is_filtered):
    """Generate data context for the AI"""
    
    # Calculate top tracks using the exact same method as dashboard
    top_tracks_by_time = df.groupby('track_artist')['hours_played'].sum().sort_values(ascending=False).head(10)
    top_tracks_dict = {}
    for track_artist, hours in top_tracks_by_time.items():
        plays = len(df[df['track_artist'] == track_artist])
        top_tracks_dict[track_artist] = {'hours': float(hours), 'plays': plays}
    
    context = {
        "dataset_info": {
            "total_records": len(df),
            "filtered_records": len(df_filtered) if is_filtered else len(df),
            "date_range": f"{df['ts'].min().strftime('%Y-%m-%d')} to {df['ts'].max().strftime('%Y-%m-%d')}",
            "columns": list(df.columns),
            "is_filtered": is_filtered
        },
        "summary_stats": {
            "total_hours": float(df_filtered['hours_played'].sum()),
            "total_plays": len(df_filtered),
            "unique_artists": df_filtered['artist_name'].nunique(),
            "unique_tracks": df_filtered['track_name'].nunique(),
            "date_range_filtered": f"{df_filtered['ts'].min().strftime('%Y-%m-%d')} to {df_filtered['ts'].max().strftime('%Y-%m-%d')}" if len(df_filtered) > 0 else "No data",
            "top_artists": df_filtered['artist_name'].value_counts().head(10).to_dict(),
            "top_tracks_correct": top_tracks_dict  # EXACT results using track_artist method
        }
    }
    return context

def create_system_prompt(data_context):
    """Create system prompt for Gemini"""
    return f"""You are a Spotify listening data analyst with FULL ACCESS to powerful data analysis tools. You have complete access to a Spotify streaming history dataset and can perform ANY data analysis requested.

DATASET ACCESS:
- **df_filtered**: Currently filtered dataset ({data_context['dataset_info']['filtered_records']:,} records)
- **df_full**: Complete unfiltered dataset ({data_context['dataset_info']['total_records']:,} records)
- **Full date range**: {data_context['dataset_info']['date_range']}
- **Current filter**: {'Applied' if data_context['dataset_info']['is_filtered'] else 'No filters applied'}

CURRENT DATA SUMMARY (FILTERED):
- Total listening hours: {data_context['summary_stats']['total_hours']:.1f}
- Total plays: {data_context['summary_stats']['total_plays']:,}
- Unique artists: {data_context['summary_stats']['unique_artists']:,}
- Unique tracks: {data_context['summary_stats']['unique_tracks']:,}
- Time period: {data_context['summary_stats']['date_range_filtered']}

TOP ARTISTS (by play count): {', '.join([f"{artist} ({count})" for artist, count in list(data_context['summary_stats']['top_artists'].items())[:5]])}

EXACT TOP TRACKS (what you MUST match): {', '.join([f"{track} ({data['hours']:.1f}h, {data['plays']} plays)" for track, data in list(data_context['summary_stats']['top_tracks_correct'].items())[:5]])}

⚠️ VALIDATION REQUIREMENT: When asked about top tracks, your answer MUST match the above numbers exactly!

AVAILABLE TOOLS & LIBRARIES:
- **pandas (pd)**: Full pandas functionality for complex data manipulation
- **plotly (px, go)**: Advanced plotting and visualization
- **numpy (np)**: Numerical operations
- **streamlit (st)**: Display results
- **ALL pandas operations**: groupby, merge, pivot, time series analysis, filtering, etc.

PRE-COMPUTED HELPERS (ready to use):
- **top_tracks_by_hours**: df_full.groupby('track_artist')['hours_played'].sum().sort_values(ascending=False)
- **top_artists_by_hours**: df_full.groupby('artist_name')['hours_played'].sum().sort_values(ascending=False)

💡 Use these pre-computed variables for instant access to common results!

AVAILABLE COLUMNS:
{', '.join(data_context['dataset_info']['columns'])}

KEY COLUMNS EXPLAINED:
- ts: timestamp (datetime) - use for time-based analysis
- ms_played: milliseconds played
- hours_played, minutes_played: derived listening time
- track_name, artist_name, album_name: music metadata (cleaned and normalized)
- track_name_raw, artist_name_raw: original uncleaned names from Spotify
- track_artist: combined field "Track Name - Artist Name" (USE THIS FOR TRACK ANALYSIS)
- skipped: boolean if track was skipped
- date, hour, day_of_week, month, year: derived time fields

⚠️ CRITICAL FOR TRACK ANALYSIS:
- **ALWAYS use 'track_artist' field for track analysis** - it properly combines track and artist
- **NEVER group by 'track_name' alone** - this can miss duplicates and variations
- The data has been cleaned to normalize track name variations (e.g., "Reelin' In The Years" vs "Reeling in the Years")
- For accurate results: df.groupby('track_artist')['hours_played'].sum()

⚠️ EXACT DASHBOARD METHODOLOGY - USE THIS EXACTLY:
```python
# EXACT method the dashboard uses for top tracks:
all_tracks = df_full.groupby('track_artist')['hours_played'].sum().sort_values(ascending=False)
top_track = all_tracks.iloc[0]  # Most played track
print(f"Top track: {all_tracks.index[0]} with {top_track:.1f} hours")

# To validate your result, always use:
track_artist_to_check = "Track Name - Artist Name"  # Replace with actual track
result = df_full[df_full['track_artist'] == track_artist_to_check]['hours_played'].sum()
plays = len(df_full[df_full['track_artist'] == track_artist_to_check])
print(f"Validation: {track_artist_to_check} = {result:.1f} hours, {plays} plays")
```

EXAMPLE TRACK ANALYSIS CODE (when showing work):
```python
# SIMPLE method using pre-computed helpers:
most_played_track = top_tracks_by_hours.index[0]
most_played_hours = top_tracks_by_hours.iloc[0]
plays_count = len(df_full[df_full['track_artist'] == most_played_track])
print(f"Most listened track: {most_played_track} with {most_played_hours:.1f} hours and {plays_count} plays")

# Alternative - define your own if needed:
my_top_tracks = df_full.groupby('track_artist')['hours_played'].sum().sort_values(ascending=False)
print(f"Top track: {my_top_tracks.index[0]} with {my_top_tracks.iloc[0]:.1f} hours")
```

⚠️ CRITICAL: Always define your variables before using them! Never reference undefined variables like 'all_tracks'.

YOUR CAPABILITIES:
✅ **Complex Data Analysis**: You CAN perform advanced filtering, grouping, time-series analysis
✅ **Multi-dataset Comparison**: You CAN compare different time periods using df_full
✅ **Statistical Analysis**: You CAN calculate trends, patterns, correlations
✅ **Time-based Filtering**: You CAN filter by years, months, date ranges
✅ **Advanced Queries**: You CAN answer complex questions about listening habits

CRITICAL INSTRUCTIONS:
1. **DO NOT SHOW WORK UNLESS EXPLICITLY ASKED**: Don't mention filtering, grouping, datasets, or methodology unless explicitly asked to do so.
2. **DIRECT ANSWERS ONLY**: Jump straight to the insights and results
3. **PLAYFUL TONE**: Be conversational, fun, and engaging
4. **NO TECHNICAL LANGUAGE**: Avoid words like "requires", "accessing", "determining", "data analysis"
5. **JUST THE FACTS**: Give specific numbers and insights without explaining how you got them
6. **USE df_full BY DEFAULT**, switch to df_filtered for questions where it's important to use the filtered dataset
7. **NO CODE OR CHARTS**: Never generate visualizations or show code
8. **DEFAULT TO LISTENING TIME RATHER THAN PLAYS** when asked for top artists, tracks or trends default to listening time.
9. **USE TRACK_ARTIST FIELD**: Always use 'track_artist' for track analysis, never 'track_name' alone

FORBIDDEN PHRASES:
❌ "requires accessing", "needs to be filtered", "after filtering", "determining", "unfortunately"
❌ "based on analysis", "the data shows", "by examining", "requires grouping"
❌ Any mention of datasets, filtering, or technical processes

RESPONSE STYLE:
✅ Direct, enthusiastic, conversational
✅ Start with the answer immediately
✅ Include specific numbers and fun insights
✅ Use casual language and personality

CORE RULE: Jump straight to the fun insights with real numbers. Be enthusiastic and conversational. NO methodology talk unless explicitly asked to show work!"""

def execute_ai_code(code, df_filtered, df_full=None):
    """Safely execute AI-generated code"""
    try:
        # Pre-compute common analysis results to prevent undefined variable errors
        top_tracks_by_hours = df_full.groupby('track_artist')['hours_played'].sum().sort_values(ascending=False) if df_full is not None else None
        top_artists_by_hours = df_full.groupby('artist_name')['hours_played'].sum().sort_values(ascending=False) if df_full is not None else None
        
        # Create a restricted namespace with pre-computed helpers
        namespace = {
            'df_filtered': df_filtered,
            'df_full': df_full,
            'pd': pd,
            'px': px,
            'go': go,
            'plt': plt,
            'np': np,
            'st': st,
            'make_subplots': make_subplots,
            # Pre-computed helpers to prevent undefined variable errors
            'top_tracks_by_hours': top_tracks_by_hours,
            'top_artists_by_hours': top_artists_by_hours,
        }
        
        # Execute the code
        exec(code, namespace)
        return True, None
    except Exception as e:
        return False, f"Error executing code: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

def clean_track_name(track_name):
    """Clean and normalize track names for better matching"""
    if pd.isna(track_name) or track_name == 'Unknown Track':
        return track_name
    
    # Remove extra whitespace and normalize
    cleaned = str(track_name).strip()
    
    # Common normalizations for track name variations
    cleaned = re.sub(r"[''']", "'", cleaned)  # Normalize apostrophes
    cleaned = re.sub(r"\s+", " ", cleaned)    # Multiple spaces to single space
    cleaned = re.sub(r"[\u00A0\u2000-\u200B\u2028\u2029\u202F\u205F\u3000]", " ", cleaned)  # Various unicode spaces
    
    return cleaned

def clean_artist_name(artist_name):
    """Clean and normalize artist names for better matching"""
    if pd.isna(artist_name) or artist_name == 'Unknown Artist':
        return artist_name
    
    # Remove extra whitespace and normalize
    cleaned = str(artist_name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)    # Multiple spaces to single space
    
    return cleaned

def create_chat_interface(df, df_filtered):
    """Create the chat interface in sidebar"""
    # Check if data context has changed
    current_hash = hash(str(df_filtered.shape) + str(df_filtered['ts'].min()) + str(df_filtered['ts'].max()))
    context_changed = st.session_state.data_context_hash != current_hash
    
    if context_changed:
        st.session_state.data_context_hash = current_hash
        # Add a system message about context change
        if len(st.session_state.chat_messages) > 0:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "📊 **Data context updated** - I'm now analyzing your filtered dataset with " + 
                          f"{len(df_filtered):,} records from {df_filtered['ts'].min().strftime('%Y-%m-%d')} to " +
                          f"{df_filtered['ts'].max().strftime('%Y-%m-%d')}" if len(df_filtered) > 0 else "No data in current filter."
            })
    
    st.markdown("### 🤖 Ask About Your Music Data")
    
    # Setup Gemini API
    if st.session_state.gemini_model is None:
        model, error = setup_gemini_api()
        if error:
            st.error("**Gemini API Setup Required**")
            
            st.markdown("""
            <div class="api-key-info">
            <h4>🔑 How to set up Gemini API:</h4>
            <ol>
                <li>Go to <a href="https://aistudio.google.com/app/apikey" target="_blank">Google AI Studio</a></li>
                <li>Sign in with your Google account</li>
                <li>Click "Create API Key"</li>
                <li>Copy your API key</li>
                <li>In Streamlit Cloud: Go to your app settings → Secrets → Add:</li>
                <pre><code>GEMINI_API_KEY = "your-api-key-here"</code></pre>
                <li>For local development, create <code>.streamlit/secrets.toml</code>:</li>
                <pre><code>[secrets]
GEMINI_API_KEY = "your-api-key-here"</code></pre>
            </ol>
            <p><strong>Note:</strong> The API is free with generous usage limits!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Offer manual API key input for testing
            with st.expander("🔧 Enter API Key Manually (for testing)"):
                manual_key = st.text_input("Gemini API Key", type="password", key="manual_gemini_key")
                if manual_key and st.button("Setup API"):
                    try:
                        genai.configure(api_key=manual_key)
                        st.session_state.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                        st.success("✅ API configured successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ API setup failed: {str(e)}")
            return
        else:
            st.session_state.gemini_model = model
            st.success("✅ Gemini AI ready!")
    
    # Show data context info
    is_filtered = len(df_filtered) != len(df)
    data_info = f"📊 **Current dataset:** {len(df_filtered):,} records"
    if is_filtered:
        data_info += f" (filtered from {len(df):,} total)"
    st.caption(data_info)
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Suggested questions for new users
    if len(st.session_state.chat_messages) == 0:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "What's my most played song?",
            "Show my listening patterns by hour",
            "Which artist do I listen to most?",
            "Create a chart of my monthly listening",
            "What's my skip rate?"
        ]
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{suggestion}"):
                # Add user message
                st.session_state.chat_messages.append({"role": "user", "content": suggestion})
                st.rerun()
    
    # Chat input
    if user_input := st.chat_input("Ask me anything about your music data...", key="main_chat_input"):
        # Add user message to session state
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        # Set flag to generate response
        st.session_state.needs_response = True
    
    # Check if we need to generate a response
    if (len(st.session_state.chat_messages) > 0 and 
        st.session_state.chat_messages[-1]["role"] == "user" and
        (len(st.session_state.chat_messages) % 2 == 1 or st.session_state.get("needs_response", False))):  # Odd number means last is user message without response
        
        user_question = st.session_state.chat_messages[-1]["content"]
        
        # Show spinner while generating response
        with st.spinner("🤖 Analyzing your data..."):
            try:
                # Get data context
                data_context = get_data_context(df, df_filtered, is_filtered)
                system_prompt = create_system_prompt(data_context)
                
                # Create conversation history for context
                conversation_history = ""
                for msg in st.session_state.chat_messages[-5:]:  # Last 5 messages for context
                    conversation_history += f"{msg['role']}: {msg['content']}\n"
                
                # Generate response
                full_prompt = f"{system_prompt}\n\nCONVERSATION HISTORY:\n{conversation_history}\n\nUSER QUESTION: {user_question}\n\nPlease provide analysis with code if needed:"
                
                response = st.session_state.gemini_model.generate_content(full_prompt)
                ai_response = response.text
                
                # Add AI response to chat history and clear response flag
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
                st.session_state.needs_response = False
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                st.session_state.needs_response = False
                st.rerun()
    
    # Clear chat button
    if len(st.session_state.chat_messages) > 0:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()

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
    
    # Clean up track names and artist names with normalization
    df['track_name_raw'] = df['master_metadata_track_name'].fillna('Unknown Track')
    df['artist_name_raw'] = df['master_metadata_album_artist_name'].fillna('Unknown Artist')
    df['album_name'] = df['master_metadata_album_album_name'].fillna('Unknown Album')
    
    # Apply cleaning functions
    df['track_name'] = df['track_name_raw'].apply(clean_track_name)
    df['artist_name'] = df['artist_name_raw'].apply(clean_artist_name)
    
    # Create the combined track_artist field using cleaned names
    df['track_artist'] = df['track_name'] + " - " + df['artist_name']
    
    # Filter out very short plays (less than 30 seconds)
    df = df[df['ms_played'] >= 30000]
    
    # Add data quality insights
    duplicates_detected = []
    track_variations = df.groupby(['track_name', 'artist_name'])['track_name_raw'].unique()
    for (track, artist), raw_names in track_variations.items():
        if len(raw_names) > 1:
            duplicates_detected.append({
                'cleaned_name': f"{track} - {artist}",
                'variations': list(raw_names)
            })
    
    # Store data quality info in session state for optional display
    if duplicates_detected:
        st.session_state.data_quality_info = {
            'duplicates_detected': len(duplicates_detected),
            'examples': duplicates_detected[:5]  # Show first 5 examples
        }
    
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
            labels={'hours_played': 'Hours Played', 'date': x_label}
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title=x_label,
            yaxis_title="Hours Played"
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
            labels={'x': 'Hours Played', 'y': 'Artist'}
        )
        fig.update_layout(
            height=500, 
            yaxis={'categoryorder': 'total ascending'}
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
        # track_artist field is now created in load_spotify_data function
        all_tracks = df.groupby('track_artist')['hours_played'].sum().sort_values(ascending=False)
        top_tracks = all_tracks.head(15)
        
        fig = px.bar(
            x=top_tracks.values,
            y=top_tracks.index,
            orientation='h',
            title="Top 15 Tracks by Listening Time",
            labels={'x': 'Hours Played', 'y': 'Track - Artist'}
        )
        fig.update_layout(
            height=500, 
            yaxis={'categoryorder': 'total ascending'}
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
                labels={'x': 'Hour of Day', 'y': 'Hours Played'}
            )
            fig.update_layout(
                height=400
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
                labels={'x': 'Day of Week', 'y': 'Hours Played'}
            )
            fig.update_layout(
                height=400
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
            labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Hours Played'}
        )
        fig.update_layout(
            height=500
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
                labels={'x': 'Month', 'y': 'Hours Played'}
            )
            fig.update_layout(
                height=400)
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
                labels={'hours_played': 'Hours Played', 'month': 'Month'}
            )
            fig.update_layout(
                height=400
            )
            fig.update_xaxes(tickangle=45)
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
            labels={'x': 'Skip Rate (%)', 'y': 'Artist'}
        )
        fig.update_layout(
            height=400, 
            yaxis={'categoryorder': 'total ascending'})
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
            labels={'skip_rate': 'Skip Rate (%)', 'time_period': 'Date'}
        )
        fig.update_layout(
            height=400)
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
    
    # Add traces for each artist/track (in reverse order for proper stacking and legend display)
    for i, item in enumerate(reversed(column_order)):        
        fig.add_trace(go.Scatter(
            x=pivot_data_pct.index,
            y=pivot_data_pct[item],
            mode='lines',
            stackgroup='one',
            name=item,
            line=dict(width=0.5),
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
        hovermode='x unified'
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
            # Create word cloud using frequencies
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=max_artists,
                relative_scaling=0.5,
                min_font_size=8,
                prefer_horizontal=0.8
            ).generate_from_frequencies(frequencies)
            
            # Display using matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(subtitle, fontsize=14, pad=20)
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
            go.Bar(x=time_stats.index, y=time_stats['Total Hours'], name='Total Hours'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=time_stats.index, y=time_stats['Total Plays'], name='Total Plays'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=time_stats.index, y=time_stats['Unique Artists'], name='Unique Artists'),
            row=1, col=3
        )
        
        fig.update_layout(
            height=400, 
            showlegend=False)
        fig.update_xaxes(title_text=x_title)
        
        # Rotate x-axis labels if needed
        if analysis_type in ["Monthly", "Weekly"]:
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
        labels={'x': 'Year', 'y': 'New Tracks Discovered'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Create sidebar with chat interface
    with st.sidebar:
        # Add toggle for chat interface
        chat_enabled = st.checkbox("🤖 Enable AI Chat", value=True, key="chat_toggle")
        
        if chat_enabled:
            st.markdown("---")
    
    # Header
    st.title("🎵 Spotify Listening History Dashboard")
    
    # Load data
    try:
        df = load_spotify_data()
        st.success(f"Successfully loaded {len(df):,} listening records from {df['ts'].min().strftime('%Y-%m-%d')} to {df['ts'].max().strftime('%Y-%m-%d')}")
        
        # Show data quality information if available
        if hasattr(st.session_state, 'data_quality_info'):
            quality_info = st.session_state.data_quality_info
            with st.expander(f"🔧 Data Quality: {quality_info['duplicates_detected']} track name variations detected and cleaned"):
                st.write("**Track name variations that were normalized:**")
                for example in quality_info['examples']:
                    st.write(f"**{example['cleaned_name']}**")
                    variations_text = ', '.join([f'"{name}"' for name in example['variations']])
                    st.write(f"   Original variations: {variations_text}")
                    st.write("")
                if quality_info['duplicates_detected'] > 5:
                    st.write(f"... and {quality_info['duplicates_detected'] - 5} more variations cleaned")
                st.info("💡 This normalization ensures accurate track statistics in both the dashboard and AI chat.")
        
        # Add Dashboard vs AI Results Comparison
        with st.expander("🔍 Dashboard vs AI Results Comparison"):
            st.write("**Verify that dashboard and AI will get the same results:**")
            
            # Calculate top tracks using the exact same method as the dashboard
            dashboard_top_tracks = df.groupby('track_artist')['hours_played'].sum().sort_values(ascending=False).head(10)
            
            st.write("**Top 10 tracks by listening time (what both dashboard and AI should show):**")
            for i, (track_artist, hours) in enumerate(dashboard_top_tracks.items(), 1):
                plays = len(df[df['track_artist'] == track_artist])
                st.write(f"{i}. **{track_artist}** - {hours:.1f} hours ({plays} plays)")
            
            st.markdown("---")
            st.write("**AI Validation Code:**")
            st.code("""
# This is the EXACT code Gemini should use:
top_tracks = df_full.groupby('track_artist')['hours_played'].sum().sort_values(ascending=False)
print(f"#1 Most played track: {top_tracks.index[0]} with {top_tracks.iloc[0]:.1f} hours")

# To check specific track:
track = "Reelin' In The Years - Steely Dan"  # Replace with actual track
hours = df_full[df_full['track_artist'] == track]['hours_played'].sum()
plays = len(df_full[df_full['track_artist'] == track])
print(f"{track}: {hours:.1f} hours, {plays} plays")
            """, language="python")
        
        # Filters section
        st.markdown("### 🎛️ Filters")
        
        # Create filter container
        with st.container():
            # Enhanced date range filter
            min_date = df['date'].min()
            max_date = df['date'].max()
            
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
            
            # Date range controls in columns
            col1, col2, col3 = st.columns([2, 3, 2])
            
            with col1:
                preset_range = st.selectbox(
                    "📅 Quick Select",
                    options=preset_options,
                    index=0,
                    key="main_preset_range"
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
            
            with col2:
                # Custom date range selector (always shown but populated based on preset)
                date_range = st.date_input(
                    "📆 Custom Date Range" if preset_range == "Custom Range" else "📆 Selected Range",
                    value=(default_start, default_end),
                    min_value=min_date,
                    max_value=max_date,
                    disabled=(preset_range != "Custom Range"),
                    key="main_date_range"
                )
            
            with col3:
                # Show selected date range info
                if len(date_range) == 2:
                    total_days = (date_range[1] - date_range[0]).days + 1
                    st.info(f"📊 **{total_days} days** selected\n\n{date_range[0]} to {date_range[1]}")
            
            # Artist filter in a separate row
            st.markdown("##### 🎤 Artist Filter")
            top_artists = df['artist_name'].value_counts().head(20).index.tolist()
            selected_artists = st.multiselect(
                "Filter by Top Artists (optional)",
                top_artists,
                key="main_artist_filter",
                help="Select one or more artists to focus your analysis"
            )
        
        # Filter data based on date range
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        else:
            df_filtered = df
        
        # Apply artist filter
        if selected_artists:
            df_filtered = df_filtered[df_filtered['artist_name'].isin(selected_artists)]
        

        
        # Add chat interface to sidebar if enabled
        if chat_enabled:
            with st.sidebar:
                create_chat_interface(df, df_filtered)
        
        st.markdown("---")
        
        # Display overview metrics
        create_overview_metrics(df_filtered)
        st.markdown("---")
        
        # Add track analysis debugging tool
        with st.expander("🔍 Track Analysis Debug Tool"):
            st.write("**Search for track variations and duplicates:**")
            search_term = st.text_input("Enter part of a track or artist name:", key="track_search")
            if search_term:
                # Search in both original and cleaned names
                matches = df[
                    df['track_name'].str.contains(search_term, case=False, na=False) |
                    df['artist_name'].str.contains(search_term, case=False, na=False) |
                    df['track_name_raw'].str.contains(search_term, case=False, na=False) |
                    df['artist_name_raw'].str.contains(search_term, case=False, na=False)
                ]
                
                if len(matches) > 0:
                    # Show track statistics
                    track_stats = matches.groupby('track_artist').agg({
                        'hours_played': 'sum',
                        'track_name_raw': 'unique',
                        'artist_name_raw': 'unique'
                    }).sort_values('hours_played', ascending=False)
                    
                    st.write(f"**Found {len(track_stats)} matching tracks:**")
                    for track_artist, row in track_stats.head(10).iterrows():
                        st.write(f"**{track_artist}** - {row['hours_played']:.1f} hours")
                        if len(row['track_name_raw']) > 1 or len(row['artist_name_raw']) > 1:
                            track_vars = list(row['track_name_raw'])
                            artist_vars = list(row['artist_name_raw'])
                            st.write(f"   Original variations: Track: {track_vars}, Artist: {artist_vars}")
                else:
                    st.write("No matches found.")
            
            # Add detailed comparison tool
            st.markdown("---")
            st.write("**🔍 Detailed Track Analysis:**")
            st.write("Search for a specific track to see ALL data entries and why results might differ:")
            
            specific_search = st.text_input("Enter exact track name (e.g., 'Reelin'):", key="specific_search")
            if specific_search:
                # Find all entries for this track
                specific_matches = df[
                    df['track_name'].str.contains(specific_search, case=False, na=False) |
                    df['track_name_raw'].str.contains(specific_search, case=False, na=False)
                ]
                
                if len(specific_matches) > 0:
                    st.write(f"**Found {len(specific_matches)} individual plays matching '{specific_search}':**")
                    
                    # Show raw data breakdown
                    raw_breakdown = specific_matches.groupby(['track_name_raw', 'artist_name_raw']).agg({
                        'hours_played': ['sum', 'count'],
                        'track_name': 'first',
                        'artist_name': 'first',
                        'track_artist': 'first'
                    }).round(2)
                    
                    st.write("**Raw Data Breakdown (before cleaning):**")
                    for (track_raw, artist_raw), row in raw_breakdown.iterrows():
                        hours = row[('hours_played', 'sum')]
                        plays = row[('hours_played', 'count')]
                        cleaned_track = row[('track_name', 'first')]
                        cleaned_artist = row[('artist_name', 'first')]
                        track_artist = row[('track_artist', 'first')]
                        
                        st.write(f"**Original**: \"{track_raw}\" by \"{artist_raw}\"")
                        st.write(f"   → **Cleaned to**: {track_artist}")
                        st.write(f"   → **Stats**: {hours:.1f} hours, {plays} plays")
                        st.write("")
                    
                    # Show final consolidated stats
                    consolidated = specific_matches.groupby('track_artist').agg({
                        'hours_played': 'sum',
                        'ms_played': 'count'
                    }).round(2)
                    
                    st.write("**Final Consolidated Results (what both dashboard and Gemini should show):**")
                    for track_artist, row in consolidated.iterrows():
                        hours = row['hours_played']
                        plays = row['ms_played']
                        st.write(f"🎵 **{track_artist}**: {hours:.1f} hours, {plays} plays")
                    
                    # Show what Gemini should use
                    st.markdown("---")
                    st.code(f"""
# What Gemini should use for accurate results:
top_tracks = df_full.groupby('track_artist')['hours_played'].sum().sort_values(ascending=False)
# For '{specific_search}', this should return: {consolidated.iloc[0]['hours_played']:.1f} hours
                    """, language="python")
                    
                else:
                    st.write("No matches found.")
        
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
        
        # Data Export Section
        st.markdown("---")
        st.subheader("💾 Export Data")
        
        # Create export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Full Dataset**")
            st.write(f"Total records: {len(df):,}")
            st.write(f"Date range: {df['ts'].min().strftime('%Y-%m-%d')} to {df['ts'].max().strftime('%Y-%m-%d')}")
            
            # Convert full dataframe to CSV
            csv_full = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset (CSV)",
                data=csv_full,
                file_name=f"spotify_full_data_{df['ts'].min().strftime('%Y%m%d')}_{df['ts'].max().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download complete Spotify listening history as CSV file"
            )
            
            # Convert to Excel (requires openpyxl)
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Full_Data', index=False)
                    
                excel_full = buffer.getvalue()
                st.download_button(
                    label="📥 Download Full Dataset (Excel)",
                    data=excel_full,
                    file_name=f"spotify_full_data_{df['ts'].min().strftime('%Y%m%d')}_{df['ts'].max().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download complete Spotify listening history as Excel file"
                )
            except ImportError:
                st.info("📝 Install openpyxl to enable Excel download: `pip install openpyxl`")
        
        with col2:
            st.markdown("**🎯 Filtered Dataset**")
            st.write(f"Filtered records: {len(df_filtered):,}")
            if len(df_filtered) > 0:
                st.write(f"Date range: {df_filtered['ts'].min().strftime('%Y-%m-%d')} to {df_filtered['ts'].max().strftime('%Y-%m-%d')}")
            
            # Convert filtered dataframe to CSV
            csv_filtered = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv_filtered,
                file_name=f"spotify_filtered_data_{df_filtered['ts'].min().strftime('%Y%m%d') if len(df_filtered) > 0 else 'empty'}_{df_filtered['ts'].max().strftime('%Y%m%d') if len(df_filtered) > 0 else 'empty'}.csv",
                mime="text/csv",
                help="Download currently filtered data as CSV file",
                disabled=(len(df_filtered) == 0)
            )
            
            # Convert filtered data to Excel
            try:
                if len(df_filtered) > 0:
                    buffer_filtered = io.BytesIO()
                    with pd.ExcelWriter(buffer_filtered, engine='openpyxl') as writer:
                        df_filtered.to_excel(writer, sheet_name='Filtered_Data', index=False)
                        
                    excel_filtered = buffer_filtered.getvalue()
                    st.download_button(
                        label="📥 Download Filtered Data (Excel)",
                        data=excel_filtered,
                        file_name=f"spotify_filtered_data_{df_filtered['ts'].min().strftime('%Y%m%d')}_{df_filtered['ts'].max().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download currently filtered data as Excel file"
                    )
                else:
                    st.button("📥 Download Filtered Data (Excel)", disabled=True, help="No data available with current filters")
            except ImportError:
                st.info("📝 Install openpyxl to enable Excel download: `pip install openpyxl`")
        
        # Additional export options
        st.markdown("**📋 Data Columns Included:**")
        st.write(", ".join(df.columns.tolist()))
        
        # Show a sample of the data structure
        with st.expander("🔍 Preview Data Structure"):
            st.write("**First 5 rows of dataset:**")
            st.dataframe(df.head(), use_container_width=True)
        
        st.markdown("---")
        
        # Data table
        if st.checkbox("Show Raw Data"):
            st.subheader("📊 Raw Data")
            st.dataframe(df_filtered[['ts', 'track_name', 'artist_name', 'album_name', 'minutes_played', 'skipped']].head(1000))
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please make sure your JSON files are in the 'raw_data' directory.")

if __name__ == "__main__":
    main() 