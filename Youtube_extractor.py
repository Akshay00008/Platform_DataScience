import os
import re
from dotenv import load_dotenv
from bson import ObjectId
from googleapiclient.discovery import build
from pymongo import errors
from mongo import get_mongo_client  # Assuming get_mongo_client() returns connected MongoClient instance


# Load environment variables from .env file
load_dotenv()

# YouTube API Key from environment variables
API_KEY = os.getenv('YOUTUBE_API_KEY')
if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY is not set in environment variables")

# MongoDB setup from external mongo.py
client = get_mongo_client()  # Method in mongo.py should return MongoClient connected instance  # Use actual DB name
collection = db['videos']      # Use actual collection name


def extract_playlist_id_from_url(playlist_url: str) -> str:
    """Extract the playlist ID from a YouTube playlist URL using regex."""
    match = re.match(r'https://www\.youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)', playlist_url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid playlist URL")


def insert_video_data(video_data: dict) -> bool:
    """Insert a single video document into MongoDB collection."""
    try:
        collection.insert_one(video_data)
        return True
    except errors.PyMongoError as e:
        print(f"MongoDB insert error for video '{video_data.get('video_url')}': {e}")
        return False


def extract_and_store_descriptions(playlist_url: str, chatbot_id: str, version_id: str, inserted_count=0) -> int:
    """Fetch videos from playlist and insert video descriptions, tags, keywords into MongoDB."""
    youtube = build('youtube', 'v3', developerKey=API_KEY, cache_discovery=False)

    try:
        playlist_id = extract_playlist_id_from_url(playlist_url)
    except ValueError as e:
        print(f"Error extracting playlist ID: {e}")
        return inserted_count

    next_page_token = None

    while True:
        pl_request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        pl_response = pl_request.execute()

        for item in pl_response.get('items', []):
            video_id = item['contentDetails']['videoId']
            video_url = f'https://www.youtube.com/watch?v={video_id}'

            video_info = youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()

            if not video_info.get('items'):
                print(f"No details found for video ID: {video_id}")
                continue

            video_snippet = video_info['items'][0]['snippet']

            title = video_snippet.get('title', 'No title available')
            description = video_snippet.get('description', 'No description available')
            tags = video_snippet.get('tags', [])          # Tags list or empty list
            keywords = video_snippet.get('keywords', '')   # Keywords usually not a standard field, default to empty string

            video_data = {
                'title': title,
                'video_url': video_url,
                'description': description,
                'chatbot_id': ObjectId(chatbot_id),
                'version_id': ObjectId(version_id),
                'tags': tags,
                'keywords': keywords
            }

            if insert_video_data(video_data):
                inserted_count += 1

        next_page_token = pl_response.get('nextPageToken')
        if not next_page_token:
            break

    print(f"Successfully inserted {inserted_count} video(s) into the database.")
    return inserted_count


# # === Usage example ===
# if __name__ == "__main__":
#     playlist_url = "https://youtube.com/playlist?list=PLmXKhU9FNesTpQNP_OpXN7WaPwGx7NWsq&si=ejz5QQokZaEVbRTV"
#     chatbot_id = "68418a5ea750b0a21067158a"  # Replace with actual chatbot ObjectId string
#     version_id = "68418a5ea750b0a21067158e"  # Replace with actual version ObjectId string

#     inserted_count = extract_and_store_descriptions(playlist_url, chatbot_id, version_id)
