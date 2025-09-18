# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import json
# from werkzeug.middleware.proxy_fix import ProxyFix
# import logging
# from utility.web_Scrapper import crawl_website
# from Databases.mongo_db import Bot_Retrieval
# from Platform_DataScience.utility.On_boarding_old import chatbot
# from embeddings_creator import embeddings_from_gcb,embeddings_from_website_content
# from Youtube_extractor import extract_and_store_descriptions
# from threading import Thread

# # app = Flask(__name__)
# # CORS(app)  # Enable CORS if needed

# app = Flask(__name__)
# app.wsgi_app = ProxyFix(app.wsgi_app)

# logging.basicConfig(level=logging.INFO)

# def process_scraping(url):
#     try:
#         app.logger.info(f"Started background scraping for URL: {url}")
#         df = crawl_website(url)

#         # Save as JSON
#         json_data = df.to_dict(orient="records")
#         with open("website_data.json", "w") as f:
#             json.dump(json_data, f, indent=4)

#         # Optional: Do embeddings here
#         # result = embeddings_from_website_content()

#         app.logger.info(f"Scraping complete for URL: {url}")
#     except Exception as e:
#         app.logger.error(f"Error during background scraping: {str(e)}")



# @app.route("/webscrapper", methods=["POST"], strict_slashes=False)
# def scrapper():
#     try:
#         data = request.get_json(force=True)
#         if not data or 'url' not in data:
#             return jsonify({"error": "Missing 'url' parameter in JSON payload"}), 400

#         url = data['url']
#         app.logger.info(f"Received scraping request for URL: {url}")

#         thread = Thread(target=process_scraping, args=(url,))
#         thread.start()

#         return jsonify({"result": "URL received. Scraping started in background."}), 200

      

#     except Exception as e:
#         app.logger.error(f"Error scraping URL: {e}", exc_info=True)
#         return jsonify({"error": "The given URL does not have permission to scrape or an internal error occurred."}), 403

# def on_boarding_Data(chatbot_id, version_id):
#     Bot_information = Bot_Retrieval(chatbot_id, version_id)
#     return Bot_information  # Ensure function returns the data

# @app.route("/Onboarding", methods=["POST"], strict_slashes=False)
# def onboard():
#     try:
#         request_data = request.get_json(force=True)
#         if not request_data:
#             return jsonify({"error": "JSON payload required"}), 400

#         chatbot_id = request_data.get('chatbot_id')
#         version_id = request_data.get('version_id')

#         if not chatbot_id or not version_id:
#             return jsonify({"error": "Both 'chatbot_id' and 'version_id' must be provided"}), 400

#         bot_data = on_boarding_Data(chatbot_id, version_id)
#         app.logger.info(f"Retrieved onboarding data for chatbot_id: {chatbot_id}, version_id: {version_id}")

#         if not bot_data:
#             return jsonify({"error": "No data found for given chatbot_id and version_id"}), 404

#         return jsonify({"result": bot_data}), 200

#     except Exception as e:
#         app.logger.error(f"Error during onboarding data retrieval: {e}", exc_info=True)
#         return jsonify({"error": "Internal server error occurred"}), 500



# @app.route('/llm', methods=['POST'], strict_slashes=False)
# def llm_endpoint():
#     try:
#         data = request.get_json()
        

#         query = data.get("query")
#         version_id = data.get("version_id")
#         chatbot_id = data.get("chatbot_id")
#         user_id = data.get("user_id")

#         if not all([query, version_id, chatbot_id, user_id]):
#             return jsonify({"error": "Missing required fields: query, version_id, chatbot_id, user_id"}), 400

#         result = chatbot(chatbot_id, version_id, query, user_id)
#         return jsonify({"result": result})

#     except Exception as e:
#         return jsonify({"error": f"An error occurred in llm_endpoint: {str(e)}"}), 500
    


# @app.route('/file_uploads', methods=['POST'], strict_slashes=False)
# # def vector_embeddings():
# #     try:
# #         data = request.get_json()

# #         blob_names = data.get('blob_names')
# #         bucket_name = data.get('bucket_name')

# #         if not blob_names or not bucket_name:
# #             return jsonify({"error": "Missing required fields: blob_names, bucket_name"}), 400

# #         result = embeddings_from_gcb(bucket_name=bucket_name, blob_names=blob_names)
# #         return jsonify({"result": "success"})

# #     except Exception as e:
# #         return jsonify({"error": f"An error occurred in vector_embeddings: {str(e)}"}), 500
    
# @app.route('/file_uploads', methods=['POST'])
# def vector_embeddings():
#     try:
#         data = request.get_json()

#         blob_names = data.get('blob_names')
#         bucket_name = data.get('bucket_name')

#         if not blob_names or not bucket_name:
#             return jsonify({"error": "Missing required fields: blob_names, bucket_name"}), 400

#         # Background processing function
#         def background_embedding_task(bucket, blobs):
#             try:
#                 app.logger.info(f"Started embedding for bucket: {bucket}, blobs: {blobs}")
#                 result = embeddings_from_gcb(bucket_name=bucket, blob_names=blobs)
#                 app.logger.info(f"Completed embedding generation for blobs in bucket: {bucket}")
#             except Exception as e:
#                 app.logger.error(f"Error during embedding generation: {str(e)}")

#         # Start background thread
#         Thread(target=background_embedding_task, args=(bucket_name, blob_names)).start()

#         return jsonify({"result": "success"}), 200

#     except Exception as e:
#         return jsonify({"error": f"An error occurred in vector_embeddings: {str(e)}"}), 500
    


# # def extract():
# #     data = request.json
# #     playlist_url = data.get('playlist_url')
# #     chatbot_id=data.get('chatbot_id')
# #     version_id=data.get('version_id')
# #     if not playlist_url and chatbot_id and version_id:
# #         return jsonify({'error': 'playlist_url, chatbot_id and version_id  is required'}), 400

# #     try:
# #         count = extract_and_store_descriptions(playlist_url,chatbot_id,version_id)
# #         return jsonify({'message': f'Successfully inserted {count} videos into MongoDB'})
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500
# @app.route('/youtube_url', methods=['POST'])
# def extract():
#     data = request.json
#     playlist_url = data.get('playlist_url')
#     chatbot_id = data.get('chatbot_id')
#     version_id = data.get('version_id')

#     # Validation: All must be provided
#     if not all([playlist_url, chatbot_id, version_id]):
#         return jsonify({'error': 'playlist_url, chatbot_id, and version_id are required'}), 400

#     # Background scraping function
#     def background_scrape(url, chatbot, version):
#         try:
#             app.logger.info(f"Started background scrape for playlist: {url}")
#             count = extract_and_store_descriptions(url, chatbot, version)
#             app.logger.info(f"Successfully inserted {count} videos from {url} for chatbot {chatbot}")
#         except Exception as e:
#             app.logger.error(f"Background scrape error: {str(e)}")

#     # Start thread
#     Thread(target=background_scrape, args=(playlist_url, chatbot_id, version_id)).start()

#     return jsonify({'message': 'Playlist URL received. Video scraping started in background.'}), 200













