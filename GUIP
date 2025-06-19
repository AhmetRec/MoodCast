import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import random
import webbrowser
import urllib.parse
import http.server
import socketserver
from transformers import pipeline
from camera_mood_detector import get_camera_mood
from voice_mood_detector import get_voice_mood

# --- Streamlit Config ---
st.set_page_config(page_title="🎶 MoodCast Spotify", layout="centered")

# --- Spotify Secrets ---
CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
REDIRECT_URI = st.secrets["SPOTIFY_REDIRECT_URI"]
SCOPE = "playlist-modify-public"

# --- Spotify Auth ---
sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    show_dialog=True,
    cache_path=None
)

# --- Mood Keywords ---
MOOD_TERMS = {
    "happy": ["happy pop", "joyful beats", "dance songs"],
    "sad": ["sad ballads", "cry songs", "melancholy music"],
    "angry": ["rage rock", "hard trap", "angry rap"],
    "calm": ["chill lofi", "relaxing music", "ambient jazz"],
    "neutral": ["top 50 hits", "indie mix", "random playlist"],
    "anxious": ["dark ambient", "eerie instrumental", "mystic soundtrack"],
    # Fusion moods
    "love": ["romantic acoustic", "tender love songs", "sweet pop"],
    "heartbreak": ["breakup songs", "emotional indie", "painful ballads"],
    "betrayal": ["revenge pop", "dark alternative", "emotional rock"],
    "excitement": ["future bass", "festival hits", "epic pop"],
    "nostalgia": ["vintage lofi", "retro pop", "emotional piano"]
}

FUSION_MOODS = {
    frozenset(["happy", "calm"]): "love",
    frozenset(["sad", "anxious"]): "heartbreak",
    frozenset(["angry", "sad"]): "betrayal",
    frozenset(["happy", "anxious"]): "excitement",
    frozenset(["calm", "sad"]): "nostalgia"
}

# --- Emotion Model (Hugging Face) ---
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)

# --- Spotify Auth Callback Server ---
class AuthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        if 'code' in params:
            self.server.auth_code = params['code'][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write("""
                <html>
                    <head><script>window.close();</script></head>
                    <body><h1>Spotify authorization successful. This tab will close automatically.</h1></body>
                </html>
            """.encode("utf-8"))

def start_local_server():
    PORT = 8888
    with socketserver.TCPServer(("localhost", PORT), AuthHandler) as httpd:
        httpd.handle_request()
        return httpd.auth_code

# --- Session States ---
if "spotify_token_info" not in st.session_state:
    st.session_state.spotify_token_info = None
if "detected_mood" not in st.session_state:
    st.session_state.detected_mood = {}

# --- App Title ---
st.title("🎧 MoodCast - AI-Based Spotify Playlist")

# --- Spotify Auth UI ---
if st.session_state.spotify_token_info:
    token_info = st.session_state.spotify_token_info
    if token_info["expires_at"] - int(time.time()) < 60:
        token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
        st.session_state.spotify_token_info = token_info

    sp = spotipy.Spotify(auth=token_info["access_token"])
    user = sp.current_user()
    st.success(f"✅ Connected to Spotify as: {user['display_name']}")

    if st.button("🔌 Disconnect"):
        st.session_state.spotify_token_info = None
        st.rerun()
else:
    if st.button("🎧 Connect to Spotify"):
        auth_url = sp_oauth.get_authorize_url()
        webbrowser.open(auth_url)
        auth_code = start_local_server()
        token_info = sp_oauth.get_access_token(auth_code, as_dict=True)
        st.session_state.spotify_token_info = token_info
        st.rerun()

# --- Mood Detection ---
mood_source = st.radio("Select mood detection method:", ["Detect via Camera", "Detect via Voice", "Detect via Text"])
moods = []

if mood_source == "Detect via Camera":
    if st.button("📷 Detect Mood"):
        mood = get_camera_mood()
        if mood in MOOD_TERMS:
            st.session_state.detected_mood = {mood: 10}
            st.success(f"Detected mood: {mood}")
        else:
            st.session_state.detected_mood = {"neutral": 10}
            st.warning("Mood not recognized. Defaulting to neutral.")
    moods = list(st.session_state.detected_mood.keys())

elif mood_source == "Detect via Voice":
    if st.button("🎤 Detect Mood"):
        mood = get_voice_mood()
        if mood in MOOD_TERMS:
            st.session_state.detected_mood = {mood: 10}
            st.success(f"Detected mood: {mood}")
        else:
            st.session_state.detected_mood = {"neutral": 10}
            st.warning("Mood not recognized. Defaulting to neutral.")
    moods = list(st.session_state.detected_mood.keys())

elif mood_source == "Detect via Text":
    user_text = st.text_area("📝 Describe how you feel:", placeholder="I'm feeling both anxious and hopeful.")

    if user_text and st.button("💬 Detect Mood from Text"):
        try:
            result = emotion_model(user_text)
            mood_map = {
                "joy": "happy",
                "sadness": "sad",
                "anger": "angry",
                "fear": "anxious",
                "love": "calm"
            }

            total_score = sum(e['score'] for e in result[0])
            mood_weights = {}
            for e in result[0]:
                label = e['label'].lower()
                mapped = mood_map.get(label)
                if mapped:
                    mood_weights[mapped] = round((e['score'] / total_score) * 10)

            if not mood_weights:
                mood_weights = {"neutral": 10}

            st.session_state.detected_mood = mood_weights
            mood_str = ", ".join(f"{m} ({w})" for m, w in mood_weights.items())
            st.success(f"🧠 Detected mood blend: {mood_str}")
        except Exception as e:
            st.error(f"Failed to detect mood: {e}")

    moods = list(st.session_state.get("detected_mood", {}).keys())

# --- Fusion Mood Detection ---
fusion_mood = None
if len(moods) >= 2:
    for fusion_set, fusion_name in FUSION_MOODS.items():
        if fusion_set.issubset(set(moods)):
            fusion_mood = fusion_name
            st.success(f"💞 Your emotions were interpreted as: **{fusion_name.upper()}**")
            moods = [fusion_mood]
            break

# --- Playlist Setup ---
playlist_name = st.text_input("🎵 Playlist name:", value="MoodCast Playlist")
shuffle = st.checkbox("🔀 Shuffle playlist")

# --- Create Playlist ---
if st.session_state.spotify_token_info and st.button("🎼 Create Playlist"):
    if not moods:
        st.warning("Please detect a mood first.")
    else:
        sp = spotipy.Spotify(auth=st.session_state.spotify_token_info["access_token"])
        user_id = sp.current_user()["id"]
        search_terms = []

        if fusion_mood:
            search_terms.extend(random.sample(MOOD_TERMS[fusion_mood], 5))
        else:
            mood_weights = st.session_state.get("detected_mood", {})
            for mood, count in mood_weights.items():
                terms = random.sample(MOOD_TERMS[mood], k=min(count, len(MOOD_TERMS[mood])))
                search_terms.extend(terms)

        random.shuffle(search_terms)

        track_uris = []
        for term in search_terms:
            results = sp.search(q=term, type="track", limit=2)
            items = results["tracks"]["items"]
            for item in items:
                if len(track_uris) < 10:
                    track_uris.append(item["uri"])
            if len(track_uris) >= 10:
                break

        if shuffle:
            random.shuffle(track_uris)

        playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True)
        sp.playlist_add_items(playlist_id=playlist["id"], items=track_uris)

        st.success("✅ Playlist created successfully!")
        st.markdown(f"[🎧 Open on Spotify]({playlist['external_urls']['spotify']})")
        st.components.v1.iframe(f"https://open.spotify.com/embed/playlist/{playlist['id']}", height=400)
