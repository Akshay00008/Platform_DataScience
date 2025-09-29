import os
import re
from dotenv import load_dotenv
from bson import ObjectId
from googleapiclient.discovery import build
from pymongo import errors
from Databases.mongo import mongo_crud, DB_NAME  # Import DB_NAME and mongo_crud from mongo.py
import urllib.parse

load_dotenv()

API_KEY = os.getenv('YOUTUBE_API_KEY')
if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY is not set in environment variables")

COLLECTION_VIDEOS = "videos"

def mongo_operation(operation, collection_name=COLLECTION_VIDEOS, query=None, update=None):
    """Centralized wrapper for mongo_crud with imported db name."""
    return mongo_crud(
        host=None,
        port=None,
        db_name=DB_NAME,
        collection_name=collection_name,
        operation=operation,
        query=query or {},
        update=update or {}
    )

def extract_playlist_id_from_url(playlist_url: str) -> str:
    # Print the URL for debugging purposes
    print(f"Received Playlist URL: {playlist_url}")

    # Decode the URL in case it is URL encoded
    playlist_url = urllib.parse.unquote(playlist_url)
    print(f"Decoded Playlist URL: {playlist_url}")  # Debugging line to check the URL after decoding

    # Updated regex to capture playlist_id and ignore any additional parameters after it
    match = re.match(r'https://(?:www\.)?youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)', playlist_url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube playlist URL. Make sure the URL is in the correct format: https://www.youtube.com/playlist?list=<playlist_id>")

def insert_video_data(video_data: dict) -> bool:
    try:
        result = mongo_operation(operation='create', query=video_data)
        return bool(result)
    except errors.PyMongoError as e:
        print(f"MongoDB insert error for video '{video_data.get('video_url')}': {e}")
        return False

def extract_and_store_descriptions(playlist_url: str, chatbot_id: str, version_id: str, inserted_count=0) -> int:
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
                print(f"No video details found for video ID {video_id}")
                continue

            snippet = video_info['items'][0]['snippet']
            title = snippet.get('title', 'No title available')
            description = snippet.get('description', 'No description available')
            tags = snippet.get('tags', [])
            keywords = snippet.get('keywords', '')

            video_data = {
                "title": title,
                "video_url": video_url,
                "description": description,
                "chatbot_id": ObjectId(chatbot_id),
                "version_id": ObjectId(version_id),
                "tags": tags,
                "keywords": keywords
            }

            if insert_video_data(video_data):
                inserted_count += 1

        next_page_token = pl_response.get('nextPageToken')
        if not next_page_token:
            break

    print(f"Successfully inserted {inserted_count} video(s) into the database.")
    return inserted_count
