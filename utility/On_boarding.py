import os
import re
import json
import logging
import getpass
from typing import List, Optional, TypedDict
from datetime import datetime
from pymongo import errors
from bson import ObjectId
import numpy as np
import tiktoken
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.base import CallbackManagerForLLMRun
from langgraph.graph import START, StateGraph

from Databases.mongo import mongo, DB_NAME  # centralized singleton client and DB name
from Databases.mongo_db import Bot_Retrieval, company_Retrieval
from utility import bots
from utility.retrain_bot import fetch_data, fetch_faqs_and_create_vector
from utility.guideance_bot import run_guidance_pipeline
from utility.handoff import generate_handoff_guidance
from utility.Files_upload_description import description_from_gcs
from utility.web_Scrapper import crawl_website
from embeddings_creator import embeddings_from_gcb, embeddings_from_website_content
from utility.website_tag_generator import new_generate_tags
from Youtube_extractor import extract_and_store_descriptions
from utility.logger_file import Logs

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and Globals
MAX_TOKEN_LIMIT = 3_000_000
TOKEN_COLLECTION = "token_tracker"

encoding = tiktoken.get_encoding("cl100k_base")
llm = None  # Will initialize later

# Token counting
def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# Safe ObjectId conversion
def safe_objectid(value):
    try:
        return ObjectId(value)
    except Exception:
        return value

# Token usage tracking
def get_token_usage(chatbot_id: str) -> int:
    try:
        client = mongo
        db = client[DB_NAME]
        collection = db[TOKEN_COLLECTION]
        _id = safe_objectid(chatbot_id)
        record = collection.find_one({"chatbot_id": _id})
        return record.get("total_tokens_used", 0) if record else 0
    except Exception as e:
        logger.error(f"Error fetching token usage: {e}")
        return 0

def update_token_usage(chatbot_id: str, tokens_used: int):
    try:
        client = mongo
        db = client[DB_NAME]
        collection = db[TOKEN_COLLECTION]
        _id = safe_objectid(chatbot_id)
        collection.update_one(
            {"chatbot_id": _id},
            {
                "$inc": {"total_tokens_used": tokens_used},
                "$set": {"last_updated_at": datetime.utcnow()},
                "$setOnInsert": {"token_limit": MAX_TOKEN_LIMIT}
            },
            upsert=True
        )
    except Exception as e:
        logger.error(f"Error updating token usage: {e}")

# Initialization of LLM and embeddings
def init_models():
    global llm
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = getpass.getpass("Enter your OpenAI API key:")
            os.environ["OPENAI_API_KEY"] = api_key

        model_name = os.getenv("GPT_MODEL")
        model_provider = os.getenv("GPT_PROVIDER")

        if not api_key or not model_name or not model_provider:
            raise ValueError("OpenAI API key, GPT_MODEL or GPT_PROVIDER not set")

        llm = init_chat_model(model_name, model_provider=model_provider)
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return False

if not init_models():
    raise RuntimeError("LLM initialization failed")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Semantic similarity for fix scenarios
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1, v2 = np.array(vec1), np.array(vec2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm_product)

def check_fixscenarios(user_query: str, similarity_threshold=0.9) -> Optional[str]:
    try:
        client = mongo
        db = client[DB_NAME]
        collection = db["fixscenarios"]
        scenarios = list(collection.find({}))
        if not scenarios:
            return None
        query_vec = embeddings.embed_query(user_query)
        for scenario in scenarios:
            question = scenario.get("customer_question")
            if not question:
                continue
            scenario_vec = embeddings.embed_query(question)
            score = cosine_similarity(query_vec, scenario_vec)
            if score > similarity_threshold:
                return scenario.get("corrected_response")
        return None
    except Exception as e:
        logger.error(f"Fix scenarios semantic check failed: {e}")
        return None

# Chatbot core
conversations_state: dict[str, List[dict]] = {}
retrieval_cache: dict[tuple, List[Document]] = {}
llm_response_cache: dict[tuple, str] = {}

def chatbot(chatbot_id: str, version_id: str, prompt: str, user_id: str) -> str:
    try:
        current_usage = get_token_usage(chatbot_id)
        if current_usage > MAX_TOKEN_LIMIT:
            return "Token limit reached. Please try later."

        guidelines = fetch_data({"chatbot_id": chatbot_id, "version_id": version_id, "collection_name": ["guidance", "handoff"]})
        bot_info = Bot_Retrieval(chatbot_id, version_id)
        company_info = company_Retrieval()

        if user_id not in conversations_state:
            conversations_state[user_id] = [{"role": "user", "content": prompt}]
        else:
            conversations_state[user_id].append({"role": "user", "content": prompt})

        # Handoff keyword check
        handoff_descs = [d.get("description", "").lower() for d in guidelines.get("handoffscenarios", [])]
        handoff_keywords = [
            "fire", "melt", "burned", "melted", "burned up",
            "new product", "not present in your list",
            "price", "pricing",
            "speak with a live agent", "unable to resolve", "live agent"
        ]
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in handoff_keywords) or any(desc and desc in prompt_lower for desc in handoff_descs):
            response = "Let's get you connected to a live agent. May I connect you?"
            conversations_state[user_id].append({"role": "bot", "content": response})
            return response

        # Semantic Fix scenarios check
        fix_response = check_fixscenarios(prompt)
        if fix_response:
            conversations_state[user_id].append({"role": "bot", "content": fix_response})
            return fix_response

        greeting = bot_info[0].get("greeting_message", "Hello!")
        purpose = bot_info[0].get("purpose", "You are an AI assistant.")
        languages = bot_info[0].get("supported_languages", ["English"])
        tone = bot_info[0].get("tone_style", "Friendly")
        company_desc = company_info[0].get("bot_company", "You represent the organization.")

        cache_key = (user_id, chatbot_id, version_id, prompt.lower())
        if cache_key in llm_response_cache:
            return llm_response_cache[cache_key]

        response = PersonalChatbot(
            conversations_state[user_id], prompt, languages, purpose, tone, greeting,
            guidelines, company_desc, chatbot_id, version_id, user_id
        )
        llm_response_cache[cache_key] = response
        conversations_state[user_id].append({"role": "bot", "content": response})

        return response

    except Exception as e:
        logger.error(f"Chatbot processing failed: {e}")
        return f"Error occurred: {e}"

def PersonalChatbot(
    conversation_history: List[dict], prompt: str, languages: List[str], purpose: str,
    tone: str, greeting: str, guidelines: dict, company_desc: str,
    chatbot_id: str, version_id: str, user_id: str
):
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State) -> dict:
        cache_key = (chatbot_id, version_id, state["question"].lower())
        if cache_key in retrieval_cache:
            return {"context": retrieval_cache[cache_key]}

        try:
            base_dir = "/home/user/platform/faiss_indexes"
            index1 = os.path.join(base_dir, f"{chatbot_id}_{version_id}_faiss_index")
            index2 = os.path.join(base_dir, f"{chatbot_id}_{version_id}_faiss_index_website")

            if not (os.path.exists(index1) and os.path.exists(index2)):
                logger.error("FAISS index paths missing")
                return {"context": []}

            store1 = FAISS.load_local(index1, embeddings, allow_dangerous_deserialization=True)
            store2 = FAISS.load_local(index2, embeddings, allow_dangerous_deserialization=True)

            docs1 = store1.similarity_search(state["question"])
            docs2 = store2.similarity_search(state["question"])

            combined_docs = []
            seen = set()
            for doc in docs1 + docs2:
                if getattr(doc, "id", None) not in seen:
                    combined_docs.append(doc)
                    seen.add(getattr(doc, "id", None))

            retrieval_cache[cache_key] = combined_docs
            return {"context": combined_docs}

        except Exception as e:
            logger.error(f"Retrieve error: {e}")
            return {"context": []}

    def generate(state: State) -> dict:
        try:
            content = "\n\n".join(d.page_content for d in state["context"])
            messages = [
                SystemMessage(f"""
You are an AI assistant with purpose: {purpose}.
Languages: {languages}.
Greeting: {greeting}.
Conversation history: {conversation_history}.
Context: {content}.
Guidelines: {guidelines}.
Company info: {company_desc}.
Please respond professionally and according to guidelines.
"""),
                HumanMessage(state["question"])
            ]

            input_text = ''.join(m.content for m in messages if hasattr(m, "content"))
            input_tokens = count_tokens(input_text)

            response = llm.invoke(messages)
            output_tokens = count_tokens(response.content)

            total_tokens = input_tokens + output_tokens
            update_token_usage(chatbot_id, total_tokens)

            current_usage = get_token_usage(chatbot_id)
            if current_usage > MAX_TOKEN_LIMIT:
                return {"answer": "Token limit exceeded. Try again later."}

            return {"answer": response.content}

        except Exception as e:
            logger.error(f"Generate error: {e}")
            return {"answer": "Failed to generate response."}

    try:
        graph = StateGraph(State).add_sequence([retrieve, generate])
        graph.add_edge(START, "retrieve")
        compiled = graph.compile()

        result = compiled.invoke({"question": prompt})

        return result.get("answer", "No answer generated.")

    except Exception as e:
        logger.error(f"Conversation graph execution error: {e}")
        return f"Error during chat execution: {e}"
