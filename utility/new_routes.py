from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import time
from threading import Thread, Lock
from werkzeug.middleware.proxy_fix import ProxyFix
from .On_boarding import chatbot
from utility.web_Scrapper import crawl_website
from Databases.mongo import Bot_Retrieval
from embeddings_creator import embeddings_from_gcb
from Youtube_extractor import extract_and_store_descriptions
from utility.logger_file import Logs
from bson import ObjectId
import pymongo

loggs = Logs()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
CORS(app)

# Thread tracking
active_threads = 0
lock = Lock()

# Store chatbot_id and version_id from onboarding
sync_info = {
    "chatbot_id": None,
    "version_id": None
}

def update_sync_status(chatbot_id, version_id):
    try:
        client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
        db = client["ChatbotDB"]
        collection = db['chatbotversions']

        try:
            chatbot_obj_id = ObjectId(chatbot_id)
            version_obj_id = ObjectId(version_id)
        except Exception:
            chatbot_obj_id = chatbot_id
            version_obj_id = version_id

        result = collection.update_one(
            {"chatbot_id": chatbot_obj_id, "version_id": version_obj_id},
            {"$set": {"sync_status": True}}
        )

        if result.modified_count > 0:
            loggs.Logging(f"✅ Sync status updated successfully for chatbot_id={chatbot_id}, version_id={version_id}")
        else:
            loggs.Logging(f"⚠️ No document updated for chatbot_id={chatbot_id}, version_id={version_id}")

    except Exception as e:
        loggs.Logging(f"❌ Failed to update sync status: {e}")

def mark_thread_done():
    global active_threads
    with lock:
        active_threads -= 1
        if active_threads == 0:
            loggs.Logging("✅ All background tasks completed. Status: completed")

def process_scraping(url):
    try:
        loggs.Logging(f"Started background scraping for URL: {url}")
        df = crawl_website(url)
        json_data = df.to_dict(orient="records")
        with open("website_data.json", "w") as f:
            json.dump(json_data, f, indent=4)
        loggs.Logging(f"Scraping complete for URL: {url}")
    except Exception as e:
        loggs.Logging(f"Error during background scraping: {str(e)}")
    finally:
        mark_thread_done()

def background_embedding_task(bucket, blobs):
    try:
        loggs.Logging(f"Started embedding for bucket: {bucket}, blobs: {blobs}")
        embeddings_from_gcb(bucket_name=bucket, blob_names=blobs)
        loggs.Logging(f"Completed embedding generation for blobs in bucket: {bucket}")
    except Exception as e:
        loggs.Logging(f"Error during embedding generation: {str(e)}")
    finally:
        mark_thread_done()

def background_scrape(url, chatbot, version):
    try:
        loggs.Logging(f"Started background scrape for playlist: {url}")
        count = extract_and_store_descriptions(url, chatbot, version)
        loggs.Logging(f"Successfully inserted {count} videos from {url} for chatbot {chatbot}")
    except Exception as e:
        loggs.Logging(f"Background scrape error: {str(e)}")
    finally:
        mark_thread_done()

@app.route("/Onboarding", methods=["POST"], strict_slashes=False)
def onboard():
    try:
        data = request.get_json(force=True)
        chatbot_id = data.get('chatbot_id')
        version_id = data.get('version_id')

        if not chatbot_id or not version_id:
            return jsonify({"error": "chatbot_id and version_id required"}), 400

        bot_data = Bot_Retrieval(chatbot_id, version_id)
        if not bot_data:
            return jsonify({"error": "No data found"}), 404

        sync_info["chatbot_id"] = chatbot_id
        sync_info["version_id"] = version_id

        Thread(target=sync_status_monitor).start()
        loggs.Logging(f"✅ Onboarding successful for chatbot_id={chatbot_id}, version_id={version_id}")
        return jsonify({"result": bot_data}), 200
    except Exception as e:
        loggs.Logging(f"Onboarding error: {e}")
        return jsonify({"error": "Internal server error"}), 500

def sync_status_monitor():
    chatbot_id = sync_info.get("chatbot_id")
    version_id = sync_info.get("version_id")
    if not chatbot_id or not version_id:
        loggs.Logging("❌ Missing chatbot_id or version_id for sync status monitor.")
        return

    loggs.Logging(f"🔁 Started monitoring sync status for chatbot_id={chatbot_id}, version_id={version_id}")
    try:
        while True:
            with lock:
                current_threads = active_threads
            if current_threads == 0:
                update_sync_status(chatbot_id, version_id)
                break
            time.sleep(120)
    except Exception as e:
        loggs.Logging(f"❌ Sync monitor error: {e}")

@app.route("/webscrapper", methods=["POST"], strict_slashes=False)
def scrapper():
    global active_threads
    try:
        data = request.get_json(force=True)
        url = data.get('url')
        if not url:
            return jsonify({"error": "Missing 'url' parameter"}), 400

        with lock:
            active_threads += 1
        Thread(target=process_scraping, args=(url,)).start()
        loggs.Logging(f"✅ Web scraping started for {url}")
        return jsonify({"result": "Scraping started in background."}), 200
    except Exception as e:
        loggs.Logging(f"Scraper error: {e}")
        return jsonify({"error": "Internal error"}), 500

@app.route("/file_uploads", methods=["POST"], strict_slashes=False)
def vector_embeddings():
    global active_threads
    try:
        data = request.get_json()
        blob_names = data.get('blob_names')
        bucket_name = data.get('bucket_name')

        if not blob_names or not bucket_name:
            return jsonify({"error": "Missing blob_names or bucket_name"}), 400

        with lock:
            active_threads += 1
        Thread(target=background_embedding_task, args=(bucket_name, blob_names)).start()
        loggs.Logging(f"✅ Embedding job started for bucket: {bucket_name}")
        return jsonify({"result": "Embedding started in background."}), 200
    except Exception as e:
        loggs.Logging(f"Embedding error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/youtube_url', methods=['POST'])
def extract():
    global active_threads
    try:
        data = request.json
        playlist_url = data.get('playlist_url')
        chatbot_id = data.get('chatbot_id')
        version_id = data.get('version_id')

        if not all([playlist_url, chatbot_id, version_id]):
            return jsonify({'error': 'playlist_url, chatbot_id, and version_id are required'}), 400

        with lock:
            active_threads += 1
        Thread(target=background_scrape, args=(playlist_url, chatbot_id, version_id)).start()
        loggs.Logging(f"✅ YouTube scraping started for chatbot_id={chatbot_id}")
        return jsonify({'message': 'YouTube scraping started in background.'}), 200
    except Exception as e:
        loggs.Logging(f"YouTube scraping error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/llm', methods=['POST'], strict_slashes=False)
def llm_endpoint():
    try:
        data = request.get_json()
        query = data.get("query")
        version_id = data.get("version_id")
        chatbot_id = data.get("chatbot_id")
        user_id = data.get("con_id")

        if not all([query, version_id, chatbot_id, user_id]):
            return jsonify({"error": "Missing required fields"}), 400

        result = chatbot(chatbot_id, version_id, query, user_id)
        loggs.Logging(f"✅ LLM query processed for chatbot_id={chatbot_id}, user_id={user_id}")
        return jsonify({"result": result})
    except Exception as e:
        loggs.Logging(f"LLM error: {e}")
        return jsonify({"error": f"LLM error: {str(e)}"}), 500

@app.route("/status", methods=["GET"])
def get_status():
    with lock:
        if active_threads == 0:
            loggs.Logging("✅ All tasks completed")
            return jsonify({"status": "completed"})
        else:
            loggs.Logging(f"⏳ {active_threads} task(s) still running")
            return jsonify({"status": f"{active_threads} task(s) still running"})
