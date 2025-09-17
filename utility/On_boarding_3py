import os
import getpass
import logging
from typing import List, Optional, TypedDict
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
import tiktoken

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from utility.retrain_bot import fetch_data
from Databases.mongo import Bot_Retrieval, company_Retrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Token counting setup
encoding = tiktoken.get_encoding("cl100k_base")
MAX_TOKEN_LIMIT = 3_000_000

# Mongo connection config
MONGO_CONNECTION_STR = "mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017/"
DATABASE_NAME = "ChatbotDB-DEV"
TOKEN_COLLECTION_NAME = "token_tracker"

class MongoConnectionSingleton:
    _client: Optional[MongoClient] = None

    @classmethod
    def get_client(cls, connection_str: str) -> MongoClient:
        if cls._client is None:
            cls._client = MongoClient(connection_str)
        return cls._client

def safe_objectid(value):
    try:
        return ObjectId(value)
    except Exception:
        return value

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def get_token_usage_from_mongo(chatbot_id: str) -> int:
    try:
        client = MongoConnectionSingleton.get_client(MONGO_CONNECTION_STR)
        db = client[DATABASE_NAME]
        collection = db[TOKEN_COLLECTION_NAME]
        _id = safe_objectid(chatbot_id)
        record = collection.find_one({"chatbot_id": _id})
        if record and "total_tokens_used" in record:
            return record["total_tokens_used"]
        else:
            return 0
    except Exception as e:
        logger.error(f"Error fetching token usage from MongoDB: {e}")
        return 0

def update_token_usage_in_mongo(chatbot_id: str, tokens_used: int):
    try:
        client = MongoConnectionSingleton.get_client(MONGO_CONNECTION_STR)
        db = client[DATABASE_NAME]
        collection = db[TOKEN_COLLECTION_NAME]
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
        logger.error(f"Error updating token usage in MongoDB: {e}")

try:
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI:")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    api = os.getenv('OPENAI_API_KEY')
    model = os.getenv('GPT_model')
    model_provider = os.getenv('GPT_model_provider')
    if not api or not model or not model_provider:
        raise ValueError("Please check the OPENAI_API_KEY, GPT_model, and GPT_model_provider.")
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1, v2 = np.array(vec1), np.array(vec2)
    dot = np.dot(v1, v2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def check_fixscenarios(user_query: str,
                       mongo_connection_str: str = MONGO_CONNECTION_STR,
                       database_name: str = DATABASE_NAME,
                       similarity_threshold: float = 0.9) -> Optional[str]:
    try:
        client = MongoConnectionSingleton.get_client(mongo_connection_str)
        db = client[database_name]
        collection = db.fixscenarios
        scenarios = list(collection.find({}))  # Optionally filter

        if not scenarios:
            return None

        query_vec = embeddings.embed_query(user_query)
        for scenario in scenarios:
            ques = scenario.get("customer_question", "")
            if not ques:
                continue
            scenario_vec = embeddings.embed_query(ques)
            score = cosine_similarity(query_vec, scenario_vec)
            if score > similarity_threshold:
                return scenario.get("corrected_response")
        return None
    except Exception as err:
        logger.error(f"Error during fixscenarios semantic check: {err}")
        return None

def load_llm(key: str, model_provider: str, model_name: str):
    try:
        if not all([key, model_provider, model_name]):
            raise ValueError("Missing LLM configuration.")
        os.environ["API_KEY"] = key
        return init_chat_model(model_name, model_provider=model_provider)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

llm = load_llm(api, model_provider, model)

converstation_state: dict[str, List[dict]] = {}
retrieval_cache: dict[tuple, List[Document]] = {}
llm_response_cache: dict[tuple, str] = {}

def chatbot(chatbot_id: str, version_id: str, prompt: str, user_id: str) -> str:
    try:
        # Load current token usage from DB once per chatbot call
        current_token_usage = get_token_usage_from_mongo(chatbot_id)
        if current_token_usage >= MAX_TOKEN_LIMIT:
            warning_msg = "Token usage limit exceeded for this chatbot. Please try again later."
            logger.warning(warning_msg)
            return warning_msg

        request_body = {
            "chatbot_id": chatbot_id,
            "version_id": version_id,
            "collection_name": ["guidance", "handoff"]
        }
        guidelines = fetch_data(request_body)
        print(guidelines)  # Debug

        Bot_information = Bot_Retrieval(chatbot_id, version_id)
        bot_company = company_Retrieval()

        if user_id not in converstation_state:
            converstation_state[user_id] = [{'role': 'user', 'content': prompt}]
        converstation_history = converstation_state[user_id]
        converstation_state[user_id].append({'role': 'user', 'content': prompt})

        # Handoff check
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
            handoff_response = "Let's get you connected to one of our live agents so they can assist you further. Would it be okay if I connect you now?"
            converstation_state[user_id].append({'role': 'bot', 'content': handoff_response})
            return handoff_response

        # Fixscenarios check
        fix_response = check_fixscenarios(prompt)
        if fix_response:
            converstation_state[user_id].append({'role': 'bot', 'content': fix_response})
            return fix_response

        greeting = Bot_information[0].get('greeting_message', "Hello!")
        purpose = Bot_information[0].get('purpose',
                                         "You are an AI assistant helping users with their queries on behalf of the organization. "
                                         "You provide clear and helpful responses while avoiding personal details and sensitive data.")
        languages = Bot_information[0].get('supported_languages', ["English"])
        tone_and_style = Bot_information[0].get('tone_style', "Friendly and professional")
        company_info = bot_company[0].get('bot_company',
                                         "You are an AI assistant representing the organization. "
                                         "Your task is to help customers with their needs and guide them with relevant information, "
                                         "without disclosing personal user data or sensitive company records.")

        cache_key_llm = (user_id, chatbot_id, version_id, prompt.lower())
        if cache_key_llm in llm_response_cache:
            logger.info("Returning cached LLM response")
            return llm_response_cache[cache_key_llm]

        llm_response = Personal_chatbot(converstation_history, prompt, languages, purpose, tone_and_style,
                                       greeting, guidelines, company_info, chatbot_id, version_id, user_id)

        llm_response_cache[cache_key_llm] = llm_response
        converstation_state[user_id].append({'role': 'bot', 'content': llm_response})
        return llm_response

    except Exception as e:
        logger.error(f"Error in chatbot function: {e}")
        return f"An error occurred: {e}"

def Personal_chatbot(converstation_history: List[dict], prompt: str, languages: List[str], purpose: str,
                    tone_and_style: str, greeting: str, guidelines: dict, company_info: str,
                    chatbot_id: str, version_id: str, user_id: str) -> str:
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State) -> dict:
        cache_key = (chatbot_id, version_id, state['question'].lower())
        if cache_key in retrieval_cache:
            logger.info("Using cached retrieval results")
            return {"context": retrieval_cache[cache_key]}
        try:
            faiss_index_dir = "/home/bramhesh_srivastav/platformdevelopment/faiss_indexes"
            faiss_index_filename = f"{chatbot_id}_{version_id}_faiss_index"
            faiss_index_website = f"{chatbot_id}_{version_id}_faiss_index_website"
            faiss_path_1 = os.path.join(faiss_index_dir, faiss_index_filename)
            faiss_path_2 = os.path.join(faiss_index_dir, faiss_index_website)
            if not os.path.exists(faiss_path_1):
                raise FileNotFoundError(f"FAISS index not found at {faiss_path_1}")
            if not os.path.exists(faiss_path_2):
                raise FileNotFoundError(f"FAISS index not found at {faiss_path_2}")
            logger.info(f"Loading FAISS index from {faiss_path_1}")
            new_vector_store = FAISS.load_local(faiss_path_1, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loading FAISS index from {faiss_path_2}")
            new_vector_store_1 = FAISS.load_local(faiss_path_2, embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS indexes loaded successfully.")
            retrieved_docs = new_vector_store.similarity_search(state['question'])
            retrieved_docs_2 = new_vector_store_1.similarity_search(state['question'])
            logger.info("retrieved_docs indexes completed successfully.")
            combined_docs = []
            seen_docs = set()
            for doc in retrieved_docs:
                if doc.id not in seen_docs:
                    combined_docs.append(doc)
                    seen_docs.add(doc.id)
            for doc in retrieved_docs_2:
                if doc.id not in seen_docs:
                    combined_docs.append(doc)
                    seen_docs.add(doc.id)
            logger.info("combined_docs returned successfully.")
            retrieval_cache[cache_key] = combined_docs
            return {"context": combined_docs}
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return {"context": []}

    def generate(state: State) -> dict:
        try:
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = [
                SystemMessage(
                    f"""
                    Role: You are a personal chatbot with the following purpose: {purpose}.
                    You can communicate fluently in the following languages: {languages}.
                    {greeting} Always keep the conversation context in mind, including the chat history:
                    {converstation_history}
                    Do not respond to the query or question which is not related to the company: {company_info}
                    You also have access to context derived from document scores:
                    {docs_content}
                    Maintain a tone and style that aligns with the following guidelines: {guidelines}
                    Maintain a tone and style: {tone_and_style}
                    --When user asks about "delivery status" or "Order status" or 
                    --"Warranty and Service Requests or Order and Shipping Queries" or 
                    --"Specialized Product Configurations[example : Schematic diagrams or wiring diagram]" or 
                    --"Payment or Billing Issues" or
                    --"if user says not working" or 
                    --"user says These words [\"fire\", \"melt\", \"burned\", \"melted\", \"burned up\"] and add synonyms of these or related words" or 
                    --"If user asks about the new product or product not present in your list" or
                    --"if you find the words like [price, pricing, burned up] word in user query" or 
                    --"When the user explicitly asks to speak with a live agent or mentions they are unable to resolve their issue with the chatbot alone" 
                    -- For the above cases please respond with: "Let's get you connected to one of our live agents so they can assist you further. Would it be okay if I connect you now?"
                    """
                ),
                HumanMessage(f"{state['question']}")
            ]

            input_text = "".join([msg.content if hasattr(msg, "content") else "" for msg in messages])
            input_tokens = count_tokens(input_text)

            response = llm.invoke(messages)

            output_tokens = count_tokens(response.content)
            total_tokens = input_tokens + output_tokens

            # Update tokens in MongoDB and check limit
            update_token_usage_in_mongo(chatbot_id, total_tokens)
            # Optionally, re-fetch usage to enforce check (to keep simple we trust update)
            # or use local variable + eventual consistency approach
            # Here simple enforcement:
            current_usage = get_token_usage_from_mongo(chatbot_id)
            if current_usage > MAX_TOKEN_LIMIT:
                logger.warning(f"Token limit exceeded for chatbot {chatbot_id}")
                return {"answer": "Sorry, this chatbot has reached the token usage limit. Please try again later."}

            return {"answer": response.content}
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return {"answer": "Sorry, something went wrong in generating a response."}

    try:
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        response = graph.invoke({"question": prompt})
        return response.get('answer', "No response generated.")
    except Exception as e:
        logger.error(f"Error in conversation graph: {e}")
        return f"An error occurred during conversation: {e}"
