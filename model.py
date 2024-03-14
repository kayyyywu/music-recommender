import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import math
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
load_dotenv()

def preprocess_spotify_songs(songs):
    songs_processed = pd.get_dummies(songs, columns=['Artist Main Genres'], prefix='', prefix_sep='')
    # Concatenate with the original 'Artist Main Genres' column
    songs_processed = pd.concat([songs['Artist Main Genres'], songs_processed], axis=1)

    songs_processed['year_1950_1959'] = songs_processed['Year'].apply(lambda year: 1 if year >= 1950 and year < 1960 else 0)
    songs_processed['year_2005_2009'] = songs_processed['Year'].apply(lambda year: 1 if year >= 2005 and year < 2010 else 0)
    songs_processed['year_2010_2014'] = songs_processed['Year'].apply(lambda year: 1 if year >= 2010 and year < 2015 else 0)
    songs_processed['year_2015_2019'] = songs_processed['Year'].apply(lambda year: 1 if year >= 2015 and year < 2020 else 0)
    songs_processed['year_2020_2023'] = songs_processed['Year'].apply(lambda year: 1 if year >= 2020 and year < 2024 else 0)

    # Drop the 'Year' column as it's not needed anymore
    songs_processed.drop('Year', axis=1, inplace=True)

    min_row = {'Popularity': '0', 'Loudness': '-60', 'Tempo': '0'}
    max_row = {'Popularity': '100', 'Loudness': '0', 'Tempo': '250'}

    min_row_df = pd.DataFrame([min_row])
    max_row_df = pd.DataFrame([max_row])

    songs_processed = pd.concat([songs_processed, min_row_df], ignore_index=True)
    songs_processed = pd.concat([songs_processed, max_row_df], ignore_index=True)

    # scale popularity, loudness, and tempo features to 0-1
    scale = ['Popularity', 'Loudness', 'Tempo']
    scaler = MinMaxScaler()

    # Fit the scaler to the same scale used in songs_processed
    scaler.fit(songs_processed[scale])

    songs_processed[scale] = scaler.fit_transform(songs_processed[scale])

    # drop min and max values
    songs_processed = songs_processed.iloc[:-2]

    # Perform one-hot encoding
    songs_processed['Mode'] = songs_processed['Mode'].astype(int)
    songs_processed = pd.get_dummies(songs_processed, columns=['Mode'])

    # Select numerical columns
    std_numeric_columns = songs_processed.select_dtypes(include='number')

    # Set the number of components
    n_components = 14
    # Perform PCA and transform the data
    pca = PCA(n_components=n_components)
    std_numeric_columns_pca = pd.DataFrame(pca.fit_transform(std_numeric_columns), columns=['PC'+str(i+1) for i in range(n_components)])

    kmeans = KMeans(n_clusters=6, random_state=42)
    # Fit the model to the data
    kmeans.fit(std_numeric_columns_pca)  
    # Generate cluster assignments for each data point
    songs_processed['Cluster'] = kmeans.predict(std_numeric_columns_pca)

    return songs_processed, scaler, pca, kmeans

def get_spotify_client(client_id, client_secret):
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

class Config:
    # List of main genres and their subgenres for reference
    main_genres = [
        'Country', 'R&B', 'Folk', 'Jazz', 
        'Metal', 'Soul','Reggae','Disco',
        'Classical', 'Gospel', 'Grime', 
        'Blues', ['Afro', 'Soca'],
        ['Pop', 'neo mellow', 'singer-songwriter', 
        'Stomp And Holler', 'Idol', 'Boy Band'], 
        ['Rock', 'British Invasion'],
        ['Hip Hop', 'drill', 'rap'],
        ['Electro', 'EDM', 'House','Techno', 
        'Ambient', 'Dubstep', 'Trance', 'New French Touch',
        'Eurodance', 'DNB', 'Drum and Bass', 'UK Garage'],
        ['Latin', 'salsa', 'bachata', 'sertanejo',
        'reggaeton', 'cumbia', 'urbano'],
        ['Alternative', 'Punk', 'Funk', 
        'Emo', 'Alt', 'Indie', 'Grunge'],
        ['Instrumental', 'Schlager', 'Instru'],
        ['Soundtrack', 'movie', 'show', 'hollywood', 'Film'],
        ['Musical', 'Broadway'],
    ]
    songs = pd.read_pickle('data/cleaned_songs.pkl')
    songs_processed, scaler, pca, kmeans = preprocess_spotify_songs(songs)
    songs_numeric_columns = songs_processed.drop(columns=['Cluster']).select_dtypes(include='number')    
    CLIENT_ID = os.getenv("CLIENT_ID")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")

    # CLIENT_ID = st.secrets["CLIENT_ID"]
    # CLIENT_SECRET = st.secrets["CLIENT_SECRET"]

    # Initialize the Spotify client
    sp = get_spotify_client(CLIENT_ID, CLIENT_SECRET)

def calculate_playlist_numeric_column_averages(playlist_data):
    numeric_data = playlist_data.select_dtypes(include='number')
    column_averages = numeric_data.mean()
    playlist_averages = pd.DataFrame([column_averages], index=['Average'])
    playlist_averages['Cluster'] = playlist_data['Cluster'].mode().iloc[0]
    
    return playlist_averages
    
def extract_main_genre(string):
    if pd.isna(string):
        return string
        
    dominant_genre, max_count = '', 0
    for genre in Config.main_genres:
        if isinstance(genre, list):
            instances = sum(string.count(sub.lower()) for sub in genre)
        else:
            instances = string.count(genre.lower())
        
        if instances > max_count:
            dominant_genre = genre[0] if isinstance(genre, list) else genre
            max_count = instances
    
    if not dominant_genre:
        dominant_genre = 'Other'

    return dominant_genre

def format_spotify_uris(df, columns):
    for column in columns:
        df[column.split(' ')[0] + ' ' + 'URI'] = "https://open.spotify.com/" + column.split(' ')[0].lower() + "/" + df[column].astype(str)
    return df

def process_playlist_data(playlist_data, scaler, pca, kmeans):
    column_order = Config.songs_numeric_columns.columns
    playlist_data = format_spotify_uris(playlist_data, ['Track ID'])
    
    # Extract the year from the 'Album Release Date' column
    playlist_data['Year'] = playlist_data['Release Date'].apply(lambda x: x.split('-')[0])
    playlist_data['Year'] = playlist_data['Year'].astype(int)

    playlist_data['year_1950_1959'] = playlist_data['Year'].apply(lambda year: 1 if year >= 1950 and year < 1960 else 0)
    playlist_data['year_2005_2009'] = playlist_data['Year'].apply(lambda year: 1 if year >= 2005 and year < 2010 else 0)
    playlist_data['year_2010_2014'] = playlist_data['Year'].apply(lambda year: 1 if year >= 2010 and year < 2015 else 0)
    playlist_data['year_2015_2019'] = playlist_data['Year'].apply(lambda year: 1 if year >= 2015 and year < 2020 else 0)
    playlist_data['year_2020_2023'] = playlist_data['Year'].apply(lambda year: 1 if year >= 2020 and year < 2024 else 0)

    # Drop the 'Year' column as it's not needed anymore
    playlist_data.drop('Year', axis=1, inplace=True)

    # Perform one-hot encoding
    playlist_data = pd.get_dummies(playlist_data, columns=['Mode'])

    playlist_data['Artist Main Genres'] = playlist_data['Artist Genres'].apply(lambda x: extract_main_genre(x))

    indices_to_drop = []

    genres = ['Afro', 'Alternative', 'Blues', 'Country', 'Disco', 'Electro', 'Folk', 'Gospel', 'Grime', 'Hip Hop', 'Instrumental', 'Jazz', 'Latin', 'Metal', 'Musical', 'Other', 'Pop', 'R&B', 'Reggae', 'Rock', 'Soul', 'Soundtrack']
    for genre in genres:
        if genre not in playlist_data['Artist Main Genres'].unique():
            # Append the row and keep track of the index of the added row
            new_index = len(playlist_data)
            playlist_data = playlist_data.append({'Artist Main Genres': genre}, ignore_index=True)
            indices_to_drop.append(new_index)
    
    # Convert 'Artist Main Genres' into dummy variables
    playlist_data = pd.get_dummies(playlist_data, columns=['Artist Main Genres'], prefix='', prefix_sep='')
    
    # Drop the added rows using the indices we tracked
    playlist_data = playlist_data.drop(indices_to_drop)
    
    # Reset index if necessary
    playlist_data.reset_index(drop=True, inplace=True)

    # scale popularity, loudness, and tempo features to 0-1
    scale = ['Popularity', 'Loudness', 'Tempo']
    playlist_data[scale] = scaler.fit_transform(playlist_data[scale])
    
    std_numeric_columns = playlist_data.select_dtypes(include='number')
    std_numeric_columns = std_numeric_columns.loc[:, column_order]  
    std_numeric_columns_pca = pd.DataFrame(pca.transform(std_numeric_columns), columns=['PC'+str(i+1) for i in range(14)])
    playlist_data['Cluster'] = kmeans.predict(std_numeric_columns_pca)
    
    return playlist_data

def extract_artist_details(spotify_search_result):
    result = {
        'artist_name': spotify_search_result.get('name', 'artist_name_not_available'),
        'artist_id': spotify_search_result.get('id', 'artist_id_not_available'),
        'artist_popularity': spotify_search_result.get('popularity', 0),
        'artist_first_genre': (spotify_search_result.get('genres', ['genre_not_available']) + ['genre_not_available'])[0],
        'artist_n_followers': spotify_search_result.get('followers', {}).get('total', 0)
    }
    return result

def expand_artist_graph(artists_name_list, max_artists=10, expand_factor=15):
    G = nx.Graph()  # Create an empty graph
    
    for name in artists_name_list:
        if len(G) >= max_artists * expand_factor:
            break
        
        search_result = Config.sp.search(name, type='artist')['artists']['items']
        if search_result:
            artist_details = extract_artist_details(search_result[0])
            G.add_node(artist_details['artist_name'], **artist_details, related_found=False)
    
    while True:
        current_size = len(G)
        for artist_name in list(G):
            if G.nodes[artist_name]['related_found'] or len(G) >= max_artists * expand_factor:
                continue
            
            # Assume Config.sp.artist_related_artists(...) is a placeholder for actual Spotify API call
            related_artists = Config.sp.artist_related_artists(G.nodes[artist_name]['artist_id'])['artists']
            sorted_related_artists = sorted(related_artists, key=lambda x: x['popularity'], reverse=True)

            for related_artist in sorted_related_artists[:10]:
                related_artist_details = extract_artist_details(related_artist)
                if related_artist_details['artist_name'] not in G:
                    G.add_node(related_artist_details['artist_name'], **related_artist_details, related_found=False)
                    if len(G) >= max_artists * expand_factor:
                        break
                G.add_edge(artist_name, related_artist_details['artist_name'])
            G.nodes[artist_name]['related_found'] = True
        
        if current_size == len(G) or len(G) >= max_artists * expand_factor:
            break

    return G

def plot_artist_graph(G):
    fig, ax = plt.subplots(figsize=(20, 20))
    nx.draw_networkx(G, with_labels=True, node_color=(.7, .8, .8), font_size=8, ax=ax)
    st.pyplot(fig)

def select_top_artists_by_connectivity(artists_name_list, num_recommendations=10, expand_factor=15):
    # Create and expand the graph based on the initial list of artists
    G = expand_artist_graph(artists_name_list, num_recommendations, expand_factor)
    
    # Exclude initial artists, then sort the remaining by their connectivity (degree)
    remaining_nodes = [(node, G.degree(node)) for node in G if node not in artists_name_list]
    sorted_by_connectivity = sorted(remaining_nodes, key=lambda x: x[1], reverse=True)
    
    # Select the top artists based on their connectivity, up to max_artists
    selected_artists = [node[0] for node in sorted_by_connectivity[:num_recommendations]]

    return selected_artists, G

def playlist_content_based_recommendations(playlist_averages, playlist_data, num_recommendations=5):
    # Select numerical columns for cosine similarity
    songs_numeric = Config.songs_processed.select_dtypes(include='number').copy()
    all_songs = Config.songs_processed.copy()
    column_order = Config.songs_numeric_columns.columns
    # Compute cosine similarity
    similarity_scores = cosine_similarity(songs_numeric[column_order], playlist_averages[column_order])

    # Add similarity scores to song DataFrame
    all_songs['Similarity Score'] = similarity_scores
    original_max = all_songs['Similarity Score'].max()
    # Calculate cluster distribution in the playlist and prioritize clusters with more songs
    cluster_distribution = playlist_data['Cluster'].value_counts(normalize=True)
    all_songs['Cluster Priority'] = all_songs['Cluster'].apply(lambda x: cluster_distribution.get(x, 0))
    
    # Prioritize recommendations based on similarity score and cluster priority
    all_songs['Similarity Score'] = all_songs['Similarity Score'] * (1 + all_songs['Cluster Priority'])

    # Normalize similarity scores to 0-1
    min_value = all_songs['Similarity Score'].min()
    max_value = all_songs['Similarity Score'].max()
    all_songs['Similarity Score'] = (all_songs['Similarity Score'] - min_value) / (max_value - min_value) * original_max

    # Sort songs by similarity score
    top_similarities = all_songs.sort_values(by='Similarity Score', ascending=False)

    # Remove songs already in playlist from recommendations
    top_similarities = top_similarities[~top_similarities['Track URI'].isin(playlist_data['Track URI'])]

    # Select relevant columns for recommendations
    recommendations_columns = ['Track Name', 'Artist Name(s)', 'Album Name', 'Album Release Date', 'Popularity', 
                               'Track URI', 'Artist Main Genres', 'Cluster', 'Similarity Score',
                               'Artist Genres']
    content_based_recommendations = top_similarities[recommendations_columns]

    if num_recommendations != -1:
        # Select top N recommendations
        content_based_recommendations_top_n = content_based_recommendations.head(num_recommendations)
    else:
        content_based_recommendations_top_n = content_based_recommendations
        
    return content_based_recommendations_top_n

def calculate_adjusted_popularity(release_date, original_popularity, max_year='2023', decay_rate=0.96):
    # Extract the release year from the release_date
    release_year = int(release_date.split('-')[0])

    # Convert max_year from string to integer
    max_year = int(max_year)
    
    # Calculate the time span between the release year and the max year
    time_span = max_year - release_year
    
    # Calculate the adjusted popularity
    adjusted_popularity = original_popularity * math.pow(decay_rate, time_span)
    
    return adjusted_popularity

def format_recommendation_results(playlist_recs):
    #show only specific columns    
    display_features = ['Track URI', 'Track Name', 'Artist Name(s)', 'Album Name', 'Artist Genres', 'Album Release Date', 'Popularity', 
                        'Similarity Score', 'Hybrid Score', 'Recommendation Type']
    playlist_recs = playlist_recs[display_features]

    playlist_recs[['Popularity', 'Similarity Score', 'Hybrid Score']] = \
        (playlist_recs[['Popularity', 'Similarity Score', 'Hybrid Score']]*100).round(2)

    column_configuration = {
        "Track URI": st.column_config.LinkColumn("Audio Preview", help="Link to track preview"),
        "Track Name": st.column_config.TextColumn("Track Name", help="Title of Track"),
        "Artist Name(s)": st.column_config.TextColumn("Artist Name(s)", help="Artist of Track"),
        "Album Name": st.column_config.TextColumn("Album Name", help="Title of Album"),
        "Artist Genres": st.column_config.TextColumn("Artist Genres", help="Artist Genres of Track"),
        "Album Release Date": st.column_config.TextColumn("Album Release Date", help="Release Date of Album"),       
        "Popularity": st.column_config.NumberColumn("Popularity", help="Popularity score for the song"),
        "Similarity Score": st.column_config.NumberColumn("Similarity Score", help="Similarity score to the playlist"),
        "Hybrid Score": st.column_config.NumberColumn("Hybrid Score", help="Hybrid score to the playlist"),
        "Recommendation Type": st.column_config.TextColumn("Recommendation Type", help="Recommendation type for the song"),
    }
    
    st.write('## Hybrid Song Recommendations')
    st.data_editor(playlist_recs, column_config=column_configuration, hide_index=True, num_rows='fixed', disabled=True)

def get_hybrid_recommendations(playlist_data, artist_recommendations, num_recommendations=5):
    st.toast('Computing Song Recommendations', icon='🎵')
        # Ensure a minimum number of recommendations
    num_recommendations = max(num_recommendations, 5)

    processed_playlist_data = process_playlist_data(playlist_data, Config.scaler, Config.pca, Config.kmeans)
    playlist_averages = calculate_playlist_numeric_column_averages(processed_playlist_data)
    # Generate content-based recommendations
    content_based_recommendations = playlist_content_based_recommendations(playlist_averages, processed_playlist_data, 
                                                                           num_recommendations=-1)
    
    # Calculate adjusted popularity based on release date and original popularity
    content_based_recommendations['Adjusted Popularity'] = content_based_recommendations.apply(
        lambda x: calculate_adjusted_popularity(x['Album Release Date'], x['Popularity']), axis=1)

    # Calculate hybrid score
    content_based_recommendations['Hybrid Score'] = content_based_recommendations['Adjusted Popularity'] * 0.1 \
                                                        + content_based_recommendations['Similarity Score'] * 0.9
    hybrid_based_recommendations = content_based_recommendations.sort_values(by='Hybrid Score', ascending=False)

    # Split recommendations into ad hoc and regular
    num_ad_hoc_recommendations = int(num_recommendations * 0.2)
    num_regular_recommendations = num_recommendations - num_ad_hoc_recommendations
    
    hybrid_recommendations_without_unlistened_similar_artists = hybrid_based_recommendations[~hybrid_based_recommendations['Artist Name(s)'].isin(artist_recommendations)] 
    hybrid_recommendations_without_unlistened_similar_artists['Recommendation Type'] = 'Familiar Artist'
    hybrid_recommendations_with_unlistened_similar_artists = hybrid_based_recommendations[hybrid_based_recommendations['Artist Name(s)'].isin(artist_recommendations)]
    hybrid_recommendations_with_unlistened_similar_artists['Recommendation Type'] = 'Unfamiliar Artist'

    # Sort recommendations
    hybrid_recommendations_without_unlistened_similar_artists = hybrid_recommendations_without_unlistened_similar_artists.sort_values(by='Similarity Score', ascending=False)
    hybrid_recommendations_with_unlistened_similar_artists = hybrid_recommendations_with_unlistened_similar_artists.sort_values(by='Similarity Score', ascending=False)

    # Concatenate final recommendations
    final_recommendations = pd.concat([hybrid_recommendations_without_unlistened_similar_artists.head(num_regular_recommendations), 
                                       hybrid_recommendations_with_unlistened_similar_artists.head(num_ad_hoc_recommendations)])
    final_recommendations = final_recommendations.sort_values(by='Hybrid Score', ascending=False)

    # Select relevant columns for recommendations
    recommendations_columns = ['Track Name', 'Artist Name(s)', 'Album Name', 'Album Release Date', 'Artist Genres', 'Track URI', 
                               'Popularity', 'Adjusted Popularity', 'Similarity Score', 'Artist Main Genres', 
                               'Cluster', 'Hybrid Score', 'Recommendation Type']
    
    final_recommendations = final_recommendations[recommendations_columns]

    return final_recommendations