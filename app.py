import streamlit as st
import pandas as pd
import os
import base64
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import re
from wordcloud import WordCloud, STOPWORDS
from dotenv import load_dotenv
from model import Config, get_hybrid_recommendations, format_recommendation_results, select_top_artists_by_connectivity, plot_artist_graph
load_dotenv()

def get_playlist_id(uri):
    if uri:
        match = re.search(r'/playlist/(\w+)', uri)
        if match:
            return match.group(1)
        else:
            print("No playlist ID found in the URI:", uri)
            return None
    else:
        return None
    
def get_playlist_data(uri):
    playlist_id = get_playlist_id(uri)
    # Get the tracks from the playlist
    playlist_tracks = Config.sp.playlist_tracks(playlist_id, fields='items(track(id, name, artists, album(id, name)))')
    # Extract relevant information and store in a list of dictionaries
    music_data = []
    for track_info in playlist_tracks['items']:
        track = track_info['track']
        track_name = track['name']
        artists = ', '.join([artist['name'] for artist in track['artists']])
        album_name = track['album']['name']
        album_id = track['album']['id']
        track_id = track['id']

        # Get audio features for the track
        audio_features = Config.sp.audio_features(track_id)[0] if track_id != 'Not available' else None
        
        # Get release date of the album
        try:
            album_info = Config.sp.album(album_id) if album_id != 'Not available' else None
            release_date = album_info['release_date'] if album_info else None
        except:
            release_date = None
            
        # Get popularity of the track
        try:
            track_info = Config.sp.track(track_id) if track_id != 'Not available' else None
            popularity = track_info['popularity'] if track_info else None
        except:
            popularity = None

        # Get genre of the artist
        try:
            artist_id = track['artists'][0]['id']  # Assuming the first artist in the list
            artist_info = Config.sp.artist(artist_id)
            genres = ', '.join(artist_info['genres'])
        except:
            genres = None

        # Add additional track information to the track data
        track_data = {
            'Track Name': track_name,
            'Artists': artists,
            'Artist Genres': genres,
            'Album Name': album_name,
            'Album ID': album_id,
            'Track ID': track_id,
            'Popularity': popularity,
            'Release Date': release_date,
            'Year': release_date.split('-')[0],
            'Explicit': track_info.get('explicit', None),
            'External URLs': track_info.get('external_urls', {}).get('spotify', None),
            'Danceability': audio_features['danceability'] if audio_features else None,
            'Energy': audio_features['energy'] if audio_features else None,
            'Loudness': audio_features['loudness'] if audio_features else None,
            'Mode': audio_features['mode'] if audio_features else None,
            'Speechiness': audio_features['speechiness'] if audio_features else None,
            'Acousticness': audio_features['acousticness'] if audio_features else None,
            'Instrumentalness': audio_features['instrumentalness'] if audio_features else None,
            'Liveness': audio_features['liveness'] if audio_features else None,
            'Valence': audio_features['valence'] if audio_features else None,
            'Tempo': audio_features['tempo'] if audio_features else None,
        }

        music_data.append(track_data)

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(music_data)

    return df

def select_playlist_columns(playlist):
    # Select and return the desired columns
    display_features = ['Track Name', 'Artists', 'Artist Genres', 'Album Name', 'Year', 'Popularity']
    result_df = playlist[display_features]
    
    return result_df


def combine_playlists(*args, **kwargs):
    if len(args) == 2:
        combined = pd.concat([args[0], args[1]], ignore_index=True)
    if len(args) == 3:
        combined = pd.concat([args[0], args[1], args[2]], ignore_index=True)
    
    duplicates = combined['Track ID'].duplicated()
    duplicate_rows = combined[duplicates]
    combined = combined.drop_duplicates()

    if duplicate_rows.shape[0] >= 1:
        st.write('## Tracks in Common')
        st.dataframe(select_playlist_columns(duplicate_rows), hide_index=True)

    return combined

def generate_wordcloud(data, column_name, stopwords=None, max_words=70, min_font_size=10, width=800, height=800, background_color='white'):
    # Extract text data from the specified column
    comment_words = " ".join(data[column_name]) + " "
    
    # Set stopwords if provided, otherwise use default
    if stopwords is None:
        stopwords = set(STOPWORDS)
    
    # Generate word cloud
    wordcloud = WordCloud(width=width, height=height,
                          background_color=background_color,
                          stopwords=stopwords,
                          max_words=max_words,
                          min_font_size=min_font_size).generate(comment_words)
    return wordcloud.to_image()

    
def get_metrics(df):
    st.write('## Playlist Feature Metrics')
    
    averages = df.mean()

    energy = averages['Energy']
    danceability = averages['Danceability']
    valence = averages['Valence']

    col1, col2, col3 = st.columns(3)
    col1.metric('Energy', int(energy*100))
    col2.metric('Danceability', int(danceability*100))
    col3.metric('Valence', int(valence*100))
    st.write("\n")
    # # Dropdown for info about features
    # with st.expander("Feature Description and Artists Wordcloud"):
    #     st.write("- Energy: Represents intensity and activity. High energy tracks feel fast, loud, and noisy, while low energy tracks are more subdued.")
    #     st.write("- Danceability: Indicates how suitable a track is for dancing, based on tempo, rhythm stability, beat strength, and overall feel.")
    #     st.write("- Valence: Conveys positiveness. Tracks with high valence sound positive (e.g., happy), while low valence tracks sound negative (e.g., sad).")
    #     wordcloud = generate_wordcloud(df, 'Artists')
    #     st.image(wordcloud, caption='Artists Wordcloud', width=600)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Spotify Recommender System 🎵", layout='wide')

    st.markdown("<h1 style='text-align: center; color: white; font-size: 66px;'>✨ Spotify Playlist Recommender ✨</h1>", unsafe_allow_html=True)
    add_bg_from_local('backrground_images/wallpaperflare.com_wallpaper.jpg')

    st.sidebar.markdown("You can search for up to 3 Spotify playlists to get personalized recommendations. Enjoy!")

    playlist_id = []
    playlist_links = []

    with st.sidebar.form(key='Form1'):
        # User inputs name and link to playlists
        for i in range(3):
            playlist_link = st.text_input('Enter Link of Playlist '+str(i+1), key=i,
                              placeholder='https://open.spotify.com/playlist/your-playlist-id')

            if playlist_link:
                playlist_id.append(get_playlist_id(playlist_link)) 
                playlist_links.append(playlist_link) 

            playlists = {key:value for key, value in zip(playlist_id, playlist_links)}
            
        num_recommendations = st.sidebar.slider("Number of Recommendations", 
                                                min_value=1, max_value=20, value=10)
        submitted_playlist = st.form_submit_button(label = 'Find Playlists 🔎')

    if submitted_playlist:
        playlist_keys = list(playlists.keys())
        st.toast('Searching Spotify for Playlists', icon='🔎')

        with st.expander('Expand to see information for each song in playlist'):
            if len(playlists) == 1:
                # Playlist 1 Dataframe
                playlist_1 = get_playlist_data(playlists[playlist_keys[0]])
                st.dataframe(select_playlist_columns(playlist_1), hide_index=True,  width=None)
            elif len(playlists) == 2:
                tab1, tab2 = st.tabs(['Playlist ' + str(i+1) for i in range(len(playlists.keys()))])
                # Playlist 1 Dataframe
                playlist_1 = get_playlist_data(playlists[playlist_keys[0]])
                tab1.dataframe(select_playlist_columns(playlist_1), hide_index=True, width=None)

                # Playlist 2 Dataframe
                playlist_2 = get_playlist_data(playlists[playlist_keys[1]])
                tab2.dataframe(select_playlist_columns(playlist_2), hide_index=True, width=None)          
            else:
                tab1, tab2, tab3 = st.tabs(['Playlist ' + str(i+1) for i in range(len(playlists.keys()))])
                # Playlist 1 Dataframe
                playlist_1 = get_playlist_data(playlists[playlist_keys[0]])
                tab1.dataframe(select_playlist_columns(playlist_1), width=None)

                # Playlist 2 Dataframe
                playlist_2 = get_playlist_data(playlists[playlist_keys[1]])
                tab2.dataframe(select_playlist_columns(playlist_2), width=None)

                # Playlist 3 Dataframe
                playlist_3 = get_playlist_data(playlists[playlist_keys[2]])
                tab3.dataframe(select_playlist_columns(playlist_3), width=None)

        if len(playlists) == 1:
            combined = playlist_1
        elif len(playlists) == 2:
            combined = combine_playlists(playlist_1, playlist_2)
        else:
            combined = combine_playlists(playlist_1, playlist_2, playlist_3)

        get_metrics(combined)
        
        # Get artist recommendations
        artist_recommendations, artist_graph = select_top_artists_by_connectivity(combined['Artists'].unique(),
                                                        num_recommendations=num_recommendations - int(num_recommendations * 0.2) + 1)
    
        hybrid_recommendations = get_hybrid_recommendations(combined, artist_recommendations, num_recommendations=num_recommendations)
        format_recommendation_results(hybrid_recommendations)

        st.write('## Related Artist Network')
        plot_artist_graph(artist_graph)

if __name__ == '__main__':
    main()

