from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
from pymongo import MongoClient, errors
from bson import ObjectId
import re

API_KEY = os.getenv('YOUTUBE_API_KEY')

# Configure your MongoDB client and database
client = MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")  # Update connection string if needed
db = client['ChatbotDB-DEV']  # DB name  

collection = db['videos']  # Collection name


def extract_playlist_id_from_url(playlist_url):
    """
    Extract the playlist ID from the URL and remove any unnecessary query parameters.
    """
    match = re.match(r'https://www\.youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)', playlist_url)
    if match:
        return match.group(1)  # Returns only the playlist ID
    else:
        raise ValueError("Invalid playlist URL")

def extract_and_store_descriptions(playlist_url, chatbot_id, version_id,inserted_count=0):
   youtube = build('youtube', 'v3', developerKey=API_KEY, cache_discovery=False)


    # Get the playlist ID directly from the URL
   playlist_id = playlist_url.split("list=")[1].split("&")[0]  # Extract playlist ID from the URL

   print("playlist")

   next_page_token = None

   while True:
        print("inside while")
        # Fetch playlist items
        pl_request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        
        pl_response = pl_request.execute()

        # Extract video IDs from the playlist
        for item in pl_response['items']:
            video_id = item['contentDetails']['videoId']
            video_url = f'https://www.youtube.com/watch?v={video_id}'

            # Get video details (description, keywords, etc.)
            video_info = youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()

            video_snippet = video_info['items'][0]['snippet']
            title = video_snippet.get('title', 'No title available')
            description = video_snippet.get('description', 'No description available')
            keywords = video_snippet.get('keywords', 'No keywords available')
            tags=video_snippet.get('tags', 'No keywords available')

            # Prepare video data to insert into MongoDB
            video_data = {
                'title': title,
                'video_url': video_url,
                'description': description,
                'chatbot_id': ObjectId(chatbot_id),
                'version_id': ObjectId(version_id),
                'keywords': keywords,
                'tags' : tags
            }

            # Insert video data into MongoDB collection
            try:
                collection.insert_one(video_data)
                inserted_count += 1
            except errors.PyMongoError as e:
                # Log the error and continue with the next video
                print(f"MongoDB insert error for video '{video_url}': {e}")

        # Check if there is another page of results
        next_page_token = pl_response.get('nextPageToken')

        if not next_page_token:
            break

   print(f"Successfully inserted {inserted_count} video(s) into the database.")
   return inserted_count
  