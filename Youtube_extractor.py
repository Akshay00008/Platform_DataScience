from pymongo import MongoClient, errors
import yt_dlp
from bson import ObjectId

# Configure your MongoDB client and database
client = MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")  # Update connection string if needed
db = client['ChatbotDB']  # DB name
collection = db['youtube_data']  # Collection name

def extract_and_store_descriptions(playlist_url, chatbot_id, version_id):
    ydl_opts = {
        'extract_flat': 'in_playlist',  # Don't download videos
        'quiet': True,
        'skip_download': True,
        'force_generic_extractor': True,
    }

    inserted_count = 0

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(playlist_url, download=False)
    except Exception as e:
        raise RuntimeError(f"Failed to extract playlist info: {e}")

    if not result or 'entries' not in result:
        raise ValueError("No videos found in the playlist or invalid playlist URL")

    for entry in result['entries']:
        try:
            video_url = entry['url']
            with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
                video_info = ydl.extract_info(video_url, download=False)

            description = video_info.get('description', '')
            description_lines = description.split('\n\n')
            first_two_lines = description_lines[:2]
            short_description = ' '.join(first_two_lines).strip()

            chatbot_oid = ObjectId(chatbot_id)
            version_oid = ObjectId(version_id)

            video_data = {
                'title': video_info.get('title'),
                'url': video_url,
                'description': short_description,
                'chatbot_id': chatbot_oid,
                'version_id': version_oid
            }

            try:
                collection.insert_one(video_data)
                inserted_count += 1
            except errors.PyMongoError as e:
                # Log the error, skip this record but continue with others
                print(f"MongoDB insert error for video '{video_url}': {e}")

        except Exception as e:
            # Log the error, skip this video but continue with others
            print(f"Error processing video entry '{entry}': {e}")

    return inserted_count