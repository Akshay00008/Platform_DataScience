from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import time
import os 
from threading import Thread, Lock
from werkzeug.middleware.proxy_fix import ProxyFix
from .On_boarding import chatbot, recreate_faiss_index
from utility.web_Scrapper import crawl_website
from Databases.mongo_db import Bot_Retrieval, website_tag_saving
from embeddings_creator import embeddings_from_gcb, embeddings_from_website_content
from utility.Files_upload_description import description_from_gcs
from Youtube_extractor import extract_and_store_descriptions
from utility.website_tag_generator import new_generate_tags_from_gpt
from utility.logger_file import Logs
from bson import ObjectId
import utility.bots as bots
from utility.guideance_bot import run_guidance_pipeline
from utility.handoff import generate_handoff_guidance
from utility.retrain_bot import fetch_data, fetch_faqs_and_create_vector
from Databases.mongo import mongo_crud

loggs = Logs()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
CORS(app)

active_threads = 0
lock = Lock()
sync_info = {"chatbot_id": None, "version_id": None}

#DB_NAME = "ChatbotDB-DEV"
COLLECTION_CHATBOTVERSIONS = "chatbotversions"

def mongo_operation(operation, collection_name=COLLECTION_CHATBOTVERSIONS, query=None, update=None):
    """Centralized helper for mongo_crud calls."""
    return mongo_crud(
       
        collection_name=collection_name,
        operation=operation,
        query=query or {},
        update=update or {}
    )

def update_sync_status(chatbot_id, version_id):
    try:
        chatbot_obj_id = ObjectId(chatbot_id) if ObjectId.is_valid(chatbot_id) else chatbot_id
        version_obj_id = ObjectId(version_id) if ObjectId.is_valid(version_id) else version_id
        query = {"chatbot_id": chatbot_obj_id, "version_id": version_obj_id}

        update = {"sync_status": True}  # **No $set here**

        result = mongo_operation(operation="update", query=query, update=update)
        if getattr(result, 'modified_count', 0) > 0:
            loggs.info(f"✅ Sync status updated successfully for chatbot_id={chatbot_id}, version_id={version_id}")
        else:
            loggs.info(f"⚠️ No document updated for chatbot_id={chatbot_id}, version_id={version_id}")
    except Exception as e:
        loggs.info(f"❌ Failed to update sync status: {e}")


def mark_thread_done():
    global active_threads
    with lock:
        active_threads -= 1
        if active_threads == 0:
            loggs.info("✅ All background tasks completed. Status: completed")

def bucket_files(bucket_name, blob_names, chatbot_id, version_id):
    result = description_from_gcs(bucket_name, blob_names, chatbot_id, version_id)
    return {"Success": result}

def process_scraping(url, chatbot_id, version_id): 
    try:
        loggs.info(f"Started background scraping for URL: {url}")
        df = crawl_website(url)
        json_data = df.to_dict(orient="records")
        with open("website_data.json", "w") as f:
            json.dump(json_data, f, indent=4)
        loggs.info(f"Scraping complete for URL: {url}")
        website_taggers = new_generate_tags_from_gpt(json_data)
        website_tag_saving(website_taggers, chatbot_id, version_id)
        embeddings_from_website_content(json_data, chatbot_id, version_id)
        loggs.info(f"Tags and vectors generated for URL: {url}")
        target_vector = 'website'
        faisll_load = bots.load_faiss_index(chatbot_id, version_id, target_vector)
        query = "Get the over all website content to create a catelogue based on website content like Tilte , description, keywords, etc "
        top_chunks = bots.search_faiss(query, faisll_load)
        extracted_content_text = bots.generate_tags_and_buckets_from_json(top_chunks, chatbot_id, version_id, url)
        loggs.info(f"Tags and vectors generated for URL: {url}")
        # mark_thread_done()
    except Exception as e:
        loggs.info(f"Error during background scraping: {str(e)}")
    finally:
        mark_thread_done()

def background_embedding_task(bucket, blobs, chatbot_id, version_id):
    try:
        loggs.info(f"Started embedding for bucket: {bucket}, blobs: {blobs}")
        embeddings_from_gcb(chatbot_id, version_id, bucket_name=bucket, blob_names=blobs)
        result = description_from_gcs(bucket, blobs, chatbot_id, version_id)
        loggs.info(f"✅ Embedding job started for bucket: {bucket}")
        loggs.info(f"Completed embedding generation for blobs in bucket: {bucket}")
    except Exception as e:
        loggs.info(f"Error during embedding generation: {str(e)}")
    finally:
        mark_thread_done()

def background_scrape(url, chatbot, version):
    try:
        loggs.info(f"Started background scrape for playlist: {url}")
        count = extract_and_store_descriptions(url, chatbot, version)
        loggs.info(f"Successfully inserted {count} videos from {url} for chatbot {chatbot}")
    except Exception as e:
        loggs.info(f"Background scrape error: {str(e)}")
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
        loggs.info(f"✅ Onboarding successful for chatbot_id={chatbot_id}, version_id={version_id}")
        return jsonify({"result": bot_data}), 200
    except Exception as e:
        loggs.info(f"Onboarding error: {e}")
        return jsonify({"error": "Internal server error"}), 500

def sync_status_monitor():
    chatbot_id = sync_info.get("chatbot_id")
    version_id = sync_info.get("version_id")
    if not chatbot_id or not version_id:
        loggs.info("❌ Missing chatbot_id or version_id for sync status monitor.")
        return
    loggs.info(f"🔁 Started monitoring sync status for chatbot_id={chatbot_id}, version_id={version_id}")
    try:
        while True:
            with lock:
                current_threads = active_threads
            if current_threads == 0:
                update_sync_status(chatbot_id, version_id)
                break
            time.sleep(120)
    except Exception as e:
        loggs.info(f"❌ Sync monitor error: {e}")

@app.route("/webscrapper", methods=["POST"], strict_slashes=False)
def scrapper():
    global active_threads
    try:
        data = request.get_json(force=True)
        url = data.get('url')
        chatbot_id = data.get('chatbot_id')
        version_id = data.get('version_id')
        if not url:
            return jsonify({"error": "Missing 'url' parameter"}), 400
        with lock:
            active_threads += 1
        Thread(target=process_scraping, args=(url, chatbot_id, version_id)).start()
        loggs.info(f"✅ Web scraping started for {url}")
        return jsonify({"result": "Scraping started in background."}), 200
    except Exception as e:
        loggs.info(f"Scraper error: {e}")
        return jsonify({"error": "Internal error"}), 500

@app.route("/file_uploads", methods=["POST"], strict_slashes=False)
def vector_embeddings():
    global active_threads
    try:
        data = request.get_json()
        blob_names = data.get('blob_names')
        bucket_name = data.get('bucket_name')
        chatbot_id = data.get('chatbot_id')
        version_id = data.get('version_id')
        filenames = [blob.get("filename") for blob in blob_names] if blob_names else []
        file_info = [{
            "id": blob.get("id"),
            "filename": blob.get("filename"),
            "bucket_name": bucket_name,
            "chatbot_id": chatbot_id,
            "version_id": version_id
        } for blob in blob_names] if blob_names else []
        if not blob_names or not bucket_name:
            return jsonify({"error": "Missing blob_names or bucket_name"}), 400
        blob_names = filenames
        with lock:
            active_threads += 1
        Thread(target=background_embedding_task, args=(bucket_name, blob_names, chatbot_id, version_id)).start()
        return jsonify({"result": "Embedding started in background."}), 200
   
        # return jsonify({"result": "Embedding started in background."}), 200
    except Exception as e:
        loggs.info(f"Embedding error: {e}")
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
        loggs.info(f"✅ YouTube scraping started for chatbot_id={chatbot_id}")
        return jsonify({'message': 'YouTube scraping started in background.'}), 200
    except Exception as e:
        loggs.info(f"YouTube scraping error: {e}")
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
        elif query == "Live_agent_trigger":
            return {"result": "Do you want to connect with a Live agent", "Buttons": ["Yes Would Like To connect", "No thanks for the help"]}
        result = chatbot(chatbot_id, version_id, query, user_id)
        if "Let's get you connected to one of our live agents so they can assist you further. Would it be okay if I connect you now?" in result:
            return {"result": result, "Buttons": ["Yes, please connect me.", "No thank you, I am all set."]}
        loggs.info(f"✅ LLM query processed for chatbot_id={chatbot_id}, user_id={user_id}")
        return jsonify({"result": result})
    except Exception as e:
        loggs.info(f"LLM error: {e}")
        return jsonify({"error": f"LLM error: {str(e)}"}), 500

@app.route("/status", methods=["GET"])
def get_status():
    with lock:
        if active_threads == 0:
            loggs.info("✅ All tasks completed")
            return jsonify({"status": "completed"})
        else:
            loggs.info(f"⏳ {active_threads} task(s) still running")
            return jsonify({"status": f"{active_threads} task(s) still running"})

@app.route("/faqs", methods=["POST"])
def faqs_endpoint():
    data = request.get_json()
    query = data.get("query")
    chatbot_id = data.get("chatbot_id")
    version_id = data.get("version_id")
    top_k = data.get("top_k", 20)
    generated_faq_count = data.get("generated_faq_count", 50)
    vector = data.get('target_vector')
    faisll_load = bots.load_faiss_index(chatbot_id, version_id, vector)
    if not query or not chatbot_id or not version_id:
        return jsonify({"error": "query, chatbot_id, and version_id are required"}), 400
    try:
        top_chunks = bots.search_faiss(query, faisll_load, k=top_k)
        extracted_faq_text = bots.extract_existing_faqs(top_chunks)
        print("267")
        extracted_faqs = bots.parse_faq_text(extracted_faq_text)
        print("269")
        inserted_existing_count = bots.save_faqs_to_mongo(extracted_faqs, chatbot_id, version_id)
        print("271")
        generated_faq_text = bots.generate_faqs_from_vectors(top_chunks, target_count=generated_faq_count)
        print(generated_faq_text)
        generated_faqs = bots.parse_faq_text(generated_faq_text)
        inserted_generated_count = bots.save_faqs_to_mongo(generated_faqs, chatbot_id, version_id)
        return jsonify({
            "inserted_existing_faq_count": inserted_existing_count,
            "inserted_generated_faq_count": inserted_generated_count,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/guidance", methods=["POST"])
def guidance_endpoint():
    data = request.get_json()
    chatbot_id = data.get("chatbot_id")
    version_id = data.get("version_id")
    query = data.get("query", "overview")
    if not chatbot_id or not version_id:
        return jsonify({"error": "chatbot_id and version_id are required"}), 400
    try:
        guidance_docs = run_guidance_pipeline(chatbot_id, version_id, query=query)
        return jsonify({
            "inserted_guidance_count": len(guidance_docs),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/handoff-guidance", methods=["POST"])
def handoff_guidance_endpoint():
    data = request.get_json()
    chatbot_id = data.get("chatbot_id")
    version_id = data.get("version_id")
    query = data.get("query", "How can the chatbot assist users?")
    if not all([chatbot_id, version_id]):
        return jsonify({"error": "chatbot_id and version_id are required."}), 400
    try:
        guidance_text = generate_handoff_guidance(query, chatbot_id, version_id)
        return jsonify({"handoff_guidance": "generated hand_off successfully_text"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/retrain", methods=["POST"]) 
def retrain_bot():
    data = request.get_json()
    chatbot_id = data.get("chatbot_id")
    version_id = data.get("version_id")
    collection_name = data.get("collection_name")
    request_body = {
        "chatbot_id": chatbot_id,
        "version_id": version_id,
        "collection_name": collection_name
    }
    merged_result = fetch_data(request_body)
    if "faqs" in request_body["collection_name"]:
        faq_vector_status = fetch_faqs_and_create_vector(
            request_body["chatbot_id"],
            request_body["version_id"]
        )
    return merged_result

@app.route("/welcome_message", methods=["POST"], strict_slashes=False)
def welcome_message():
    data = request.get_json()
    message = data.get("message")
    lang = data.get("lang")
    translate = bots.translate_welcome_message(message, lang)
    return {"message": translate}


@app.route('/deployment', methods=['POST'])
def copy_faiss_index():
    data = request.get_json()
    old_chatbot_id = data.get('old_chatbot_id')
    old_version_id = data.get('old_version_id')
    new_chatbot_id = data.get('new_chatbot_id')
    new_version_id = data.get('new_version_id')

    # Validate required fields
    if not all([old_chatbot_id, old_version_id, new_chatbot_id, new_version_id]):
        return jsonify({'error': 'All chatbot and version IDs are required'}), 400

    try:
        # Call the function to recreate the FAISS index
        recreate_faiss_index(
            old_chatbot_id,
            old_version_id,
            new_chatbot_id,
            new_version_id,
            # embeddings  # Make sure embeddings are passed
        )
        loggs.info(f"✅ FAISS index copied from {old_chatbot_id} v{old_version_id} to {new_chatbot_id} v{new_version_id}")
        return jsonify({'message': 'FAISS index copy initiated successfully.'}), 200
    except Exception as e:
        loggs.error(f"❌ FAISS index copy error: {e}")
        return jsonify({'error': str(e)}), 500



   

