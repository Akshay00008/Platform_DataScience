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
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain.community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph

from utility.retrain_bot import fetch_data
from Databases.mongo_db import Bot_Retrieval, company_Retrieval
from Databases.mongo import mongo, DB_NAME  # Import centralized Mongo client and DB_NAME
import utility.bots as bots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

encoding = tiktoken.get_encoding("cl100k_base")
MAX_TOKEN_LIMIT = 3_000_000

# Singleton MongoDB client from mongo.py
def safe_objectid(value):
    try:
        return ObjectId(value)
    except Exception:
        return value

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def get_token_usage_from_mongo(chatbot_id: str) -> int:
    try:
        client = mongo  # from mongo.py, singleton client
        db = client[DB_NAME]
        collection = db['token_tracker']
        _id = safe_objectid(chatbot_id)
        record = collection.find_one({"chatbot_id": _id})
        if record and "total_tokens" in record:
            return record["total_tokens"]
        return 0
    except Exception as e:
        logger.error(f"Error fetching token usage: {e}")
        return 0

def update_token_usage_in_mongo(chatbot_id: str, tokens_used: int):
    try:
        client = mongo
        db = client[DB_NAME]
        collection = db['token_tracker']
        _id = safe_objectid(chatbot_id)
        collection.update_one(
            {"chatbot_id": _id},
            {
                "$inc": {"total_tokens": tokens_used},
                "$set": {"last_updated_at": datetime.utcnow()},
                "$setOnInsert": {"token_limit": MAX_TOKEN_LIMIT}
            },
            upsert=True
        )
    except Exception as e:
        logger.error(f"Error updating token usage: {e}")

# Setup OpenAI client and embeddings
try:
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("GPT_model")
    model_provider = os.getenv("GPT_provider")
    if not api_key or not model_name or not model_provider:
        raise ValueError("Missing OpenAI API key or model config in environment.")
except Exception as e:
    logger.error(f"Initialization error: {e}")
    raise

llm = init_chat_model(model_name, model_provider=model_provider)

converstation_state: dict[str, List[dict]] = {}
retrieval_cache: dict[tuple, List[Document]] = {}
llm_response_cache: dict[tuple, str] = {}

def chatbot(chatbot_id: str, version_id: str, prompt: str, user_id: str) -> str:
    try:
        # Check token usage limits
        current_usage = get_token_usage_from_mongo(chatbot_id)
        if current_usage >= MAX_TOKEN_LIMIT:
            warning_msg = "Token limit exceeded for this chatbot. Please try again later."
            logger.warning(warning_msg)
            return warning_msg

        # Fetch guidance and handoff data
        request = {"chatbot_id": chatbot_id, "version_id": version_id, "collection_name": ["guidance", "handoff"]}
        guidelines = fetch_data(request)

        bot_info = Bot_Retrieval(chatbot_id, version_id)
        company_info = company_Retrieval()

        if user_id not in converstation_state:
            converstation_state[user_id] = [{"role": "user", "content": prompt}]
        else:
            converstation_state[user_id].append({"role": "user", "content": prompt})

        # Handoff keyword detection
        handoff_descs = [d.get("description", "").lower() for d in guidelines.get("handoffscenarios", [])]
        handoff_keywords = [
            "fire", "melt", "burned", "melted", "burned up",
            "new product", "not present in your list",
            "price", "pricing",
            "speak with a live agent", "unable to resolve", "live agent"
        ]
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in handoff_keywords) or \
           any(desc and (desc in prompt_lower or prompt_lower in desc) for desc in handoff_descs):
            response = "Let's get you connected to a live agent. May I connect you now?"
            converstation_state[user_id].append({"role": "bot", "content": response})
            return response

        # Semantic fix check
        fix_resp = check_fixscenarios(prompt)
        if fix_resp:
            converstation_state[user_id].append({"role": "bot", "content": fix_resp})
            return fix_resp

        greeting = bot_info[0].get("greeting_message", "Hello!")
        purpose = bot_info[0].get("purpose", "You are an AI assistant.")
        languages = bot_info[0].get("supported_languages", ["English"])
        tone = bot_info[0].get("tone_style", "Friendly and professional")
        company_desc = company_info[0].get("bot_company", "You are an AI assistant.")

        cache_key = (user_id, chatbot_id, version_id, prompt.lower())
        if cache_key in llm_response_cache:
            logger.info("Returning cached response")
            return llm_response_cache[cache_key]

        llm_response = Personal_chatbot(converstation_state[user_id], prompt, languages, purpose, tone, greeting, guidelines, company_desc, chatbot_id, version_id, user_id)

        llm_response_cache[cache_key] = llm_response
        converstation_state[user_id].append({"role": "bot", "content": llm_response})

        return llm_response

    except Exception as e:
        logger.error(f"Error in chatbot: {e}")
        return f"Error: {e}"

def Personal_chatbot(converstation_history: List[dict], prompt: str, languages: List[str], purpose: str, tone_style: str, greeting: str, guidelines: dict, company_info: str, chatbot_id: str, version_id: str, user_id: str):
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State) -> dict:
        cache_key = (chatbot_id, version_id, state["question"].lower())
        if cache_key in retrieval_cache:
            logger.info("Using cached retrieval")
            return {"context": retrieval_cache[cache_key]}
        try:
            import os
            base_dir = "/home/user/platform/faiss_indexes"
            index_1 = os.path.join(base_dir, f"{chatbot_id}_{version_id}_faiss_index")
            index_2 = os.path.join(base_dir, f"{chatbot_id}_{version_id}_faiss_index_website")
            if not os.path.exists(index_1) or not os.path.exists(index_2):
                logger.error("One or more FAISS indexes missing")
                return {"context": []}
            store_1 = FAISS.load_local(index_1, embeddings, allow_dangerous_deserialization=True)
            store_2 = FAISS.load_local(index_2, embeddings, allow_dangerous_deserialization=True)
            docs_1 = store_1.similarity_search(state["question"])
            docs_2 = store_2.similarity_search(state["question"])
            combined = []
            seen = set()
            for d in docs_1 + docs_2:
                if getattr(d, "id", None) not in seen:
                    combined.append(d)
                    seen.add(getattr(d, "id", None))
            retrieval_cache[cache_key] = combined
            return {"context": combined}
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return {"context": []}

    def generate(state: State) -> dict:
        try:
            content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = [
                SystemMessage(f"""
                    You are an AI assistant with purpose: {purpose}.
                    Languages supported: {languages}.
                    Greeting: {greeting}.
                    Conversation history: {converstation_history}.
                    Context: {content}.
                    Guidelines: {guidelines}.
                    Style: {tone_style}.
                    Company info: {company_info}.
                    Respond with professionalism and according to guidelines.
                """),
                HumanMessage(state["question"])
            ]
            input_text = "".join(m.content for m in messages if hasattr(m, "content"))
            input_tokens = count_tokens(input_text)
            response = llm.invoke(messages)
            output_tokens = count_tokens(response.content)
            total_tokens = input_tokens + output_tokens
            update_token_usage_in_mongo(chatbot_id, total_tokens)
            if get_token_usage_from_mongo(chatbot_id) > MAX_TOKEN_LIMIT:
                return {"answer": "Token limit exceeded."}
            return {"answer": response.content}
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return {"answer": "Error generating response."}

    try:
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        result = graph.invoke({"question": prompt})
        return result.get("answer", "No answer generated.")
    except Exception as e:
        logger.error(f"Graph execution error: {e}")
        return f"Error in conversation: {e}"
