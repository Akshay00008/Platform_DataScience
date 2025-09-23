import os
import getpass
import logging
from typing import List, Optional, TypedDict
from datetime import datetime
from dotenv import load_dotenv
from bson import ObjectId
import numpy as np
import tiktoken
from pymongo import MongoClient

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph

from utility.retrain_bot import fetch_data
from Databases.mongo_db import Bot_Retrieval, company_Retrieval
from Databases.mongo import mongo_crud  # Centralized mongo_crud import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
MAX_TOKEN_LIMIT = 3_000_000
TOKEN_COLLECTION = "token_tracker"

# Initialize tokenizer for token counting
encoding = tiktoken.get_encoding("cl100k_base")


def safe_objectid(value):
    try:
        if ObjectId.is_valid(value):
            return ObjectId(value)
        else:
            logger.warning(f"Value is not a valid ObjectId: {value}. Using original value instead.")
            return value
    except Exception as e:
        logger.warning(f"Exception in safe_objectid: {e}. Using original value: {value}")
        return value


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def mongo_operation(operation, collection_name, query=None, update=None, start=0, stop=10, **kwargs):
    # Added **kwargs to forward additional options like upsert if mongo_crud supports it
    return mongo_crud(
        collection_name=collection_name,
        operation=operation,
        query=query or {},
        update=update or {},
        start=start,
        stop=stop,
        **kwargs
    )


def get_token_usage_from_mongo(chatbot_id: str) -> int:
    try:
        _id = safe_objectid(chatbot_id)
        record = mongo_operation("findone", TOKEN_COLLECTION, query={"chatbot_id": _id})
        if record and "total_tokens_used" in record:
            return record["total_tokens_used"]
        return 0
    except Exception as e:
        logger.error(f"Error fetching token usage: {e}")
        return 0


def update_token_usage_in_mongo(chatbot_id: str, tokens_used: int):
    try:
        _id = safe_objectid(chatbot_id)
        query = {"chatbot_id": _id}
        update_doc = {
            "$inc": {"total_tokens_used": tokens_used},
            "$set": {"last_updated_at": datetime.utcnow()},
            "$setOnInsert": {"token_limit": MAX_TOKEN_LIMIT}
        }
        result = mongo_operation(
            "update",
            TOKEN_COLLECTION,
            query=query,
            update=update_doc,
            upsert=True  # Ensures create if not present
        )
        if not result or (hasattr(result, 'modified_count') and result.modified_count == 0):
            logger.warning(f"No documents updated for chatbot_id {_id}, upsert might have created new document.")
    except Exception as e:
        logger.error(f"Error updating token usage: {e}")


# Initialize embeddings and LLM from environment
try:
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI:")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("GPT_model")
    model_provider = os.getenv("GPT_model_provider")
    if not api_key or not model_name or not model_provider:
        raise ValueError("Missing API key or model settings")
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


def check_fixscenarios(user_query: str, similarity_threshold: float = 0.9) -> Optional[str]:
    try:
        scenarios = mongo_operation("read", "fixscenarios")
        if not scenarios:
            return None
        query_vec = embeddings.embed_query(user_query)
        for scenario in scenarios:
            question = scenario.get("customer_question", "")
            if not question:
                continue
            scenario_vec = embeddings.embed_query(question)
            score = cosine_similarity(query_vec, scenario_vec)
            if score > similarity_threshold:
                return scenario.get("corrected_response")
        return None
    except Exception as err:
        logger.error(f"Error in fixscenarios check: {err}")
        return None


def load_llm(api_key: str, model_provider: str, model_name: str):
    try:
        if not all([api_key, model_provider, model_name]):
            raise ValueError("Missing LLM configuration.")
        os.environ["API_KEY"] = api_key
        return init_chat_model(model_name, model_provider=model_provider)
    except Exception as e:
        logger.error(f"LLM init failed: {e}")
        raise


llm = load_llm(api_key, model_provider, model_name)


# Conversation and response cache
conversation_state: dict[str, List[dict]] = {}
retrieval_cache: dict[tuple, List[Document]] = {}
llm_response_cache: dict[tuple, str] = {}


def chatbot(chatbot_id: str, version_id: str, prompt: str, user_id: str) -> str:
    try:
        current_usage = get_token_usage_from_mongo(chatbot_id)
        if current_usage >= MAX_TOKEN_LIMIT:
            warning_msg = "Token usage limit exceeded for this chatbot. Please try again later."
            logger.warning(warning_msg)
            return warning_msg

        request_body = {
            "chatbot_id": chatbot_id,
            "version_id": version_id,
            "collection_name": ["guidance", "handoff", "handoffbuzzwords"]
        }
        guidelines = fetch_data(request_body)
        logger.debug(f"Guidelines: {guidelines}")

        bot_info = Bot_Retrieval(chatbot_id, version_id)
        bot_company = company_Retrieval()

        if user_id not in conversation_state:
            conversation_state[user_id] = []
        conversation_state[user_id].append({"role": "user", "content": prompt})

        prompt_lower = prompt.lower()

        handoff_descs = [d.get("description", "").lower() for d in guidelines.get("handoffscenarios", [])]

        handoffbuzzwords_raw = guidelines.get("handoffbuzzwords", [])
        handoff_buzzwords = []
        for bw_item in handoffbuzzwords_raw:
            buzzwords_list = bw_item.get("buzzwords", [])
            for buzzword in buzzwords_list:
                handoff_buzzwords.append(buzzword.lower())
        handoff_buzzwords = list(set(handoff_buzzwords))

        base_handoff_keywords = [
            "fire", "melt", "burned", "melted", "burned up",
            "new product", "not present in your list",
            "price", "pricing",
            "speak with a live agent", "unable to resolve", "live agent"
        ]
        handoff_keywords = base_handoff_keywords + handoff_buzzwords

        if any(keyword in prompt_lower for keyword in handoff_keywords) or \
           any(desc and (desc in prompt_lower or prompt_lower in desc) for desc in handoff_descs):
            handoff_msg = (
                "Let's get you connected to one of our live agents so they can assist you further. "
                "Would it be okay if I connect you now?"
            )
            conversation_state[user_id].append({"role": "bot", "content": handoff_msg})
            return handoff_msg

        fix_resp = check_fixscenarios(prompt)
        if fix_resp:
            conversation_state[user_id].append({"role": "bot", "content": fix_resp})
            return fix_resp

        greeting = bot_info[0].get("greeting_message", "Hello!")
        purpose = bot_info[0].get(
            "purpose",
            "You are an AI assistant helping users with their queries on behalf of the organization. "
            "You provide clear and helpful responses while avoiding personal details and sensitive data."
        )
        languages = bot_info[0].get("supported_languages", ["English"])
        tone_style = bot_info[0].get("tone_style", "Friendly and professional")
        company_info = bot_company[0].get(
            "bot_company",
            "You are an AI assistant representing the organization. "
            "Your task is to help customers with their needs and guide them with relevant information, "
            "without disclosing personal user data or sensitive company records."
        )

        cache_key = (user_id, chatbot_id, version_id, prompt_lower)
        if cache_key in llm_response_cache:
            logger.info("Returning cached LLM response")
            return llm_response_cache[cache_key]

        llm_resp = Personal_chatbot(
            conversation_state[user_id], prompt, languages, purpose, tone_style, greeting, guidelines,
            company_info, chatbot_id, version_id, user_id
        )
        llm_response_cache[cache_key] = llm_resp
        conversation_state[user_id].append({"role": "bot", "content": llm_resp})

        return llm_resp

    except Exception as e:
        logger.error(f"Error in chatbot function: {e}")
        return f"An error occurred: {e}"


def Personal_chatbot(conversation_history: List[dict], prompt: str, languages: List[str], purpose: str,
                     tone_and_style: str, greeting: str, guidelines: dict, company_info: str,
                     chatbot_id: str, version_id: str, user_id: str) -> str:

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State) -> dict:
        cache_key = (chatbot_id, version_id, state["question"].lower())
        if cache_key in retrieval_cache:
            logger.info("Using cached retrieval results")
            return {"context": retrieval_cache[cache_key]}
        try:
            import os
            faiss_dir = "/home/bramhesh_srivastav/Platform_DataScience/faiss_indexes"
            faiss_file_1 = f"{chatbot_id}_{version_id}_faiss_index"
            faiss_file_2 = f"{chatbot_id}_{version_id}_faiss_index_website"
            path1 = os.path.join(faiss_dir, faiss_file_1)
            path2 = os.path.join(faiss_dir, faiss_file_2)

            vector_store_1 = None
            vector_store_2 = None

            if os.path.exists(path1):
                try:
                    vector_store_1 = FAISS.load_local(path1, embeddings, allow_dangerous_deserialization=True)
                    logger.info(f"Loaded FAISS index from {path1}")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index {path1}: {e}")
            else:
                logger.info(f"FAISS index not found at {path1}")

            if os.path.exists(path2):
                try:
                    vector_store_2 = FAISS.load_local(path2, embeddings, allow_dangerous_deserialization=True)
                    logger.info(f"Loaded FAISS index from {path2}")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index {path2}: {e}")
            else:
                logger.info(f"FAISS index not found at {path2}")

            if not vector_store_1 and not vector_store_2:
                logger.error("No FAISS indexes loaded; returning empty context")
                retrieval_cache[cache_key] = []
                return {"context": []}

            combined_docs = []
            seen_ids = set()

            if vector_store_1:
                docs1 = vector_store_1.similarity_search(state["question"])
                for d in docs1:
                    doc_id = getattr(d, "id", None)
                    if doc_id not in seen_ids:
                        combined_docs.append(d)
                        seen_ids.add(doc_id)

            if vector_store_2:
                docs2 = vector_store_2.similarity_search(state["question"])
                for d in docs2:
                    doc_id = getattr(d, "id", None)
                    if doc_id not in seen_ids:
                        combined_docs.append(d)
                        seen_ids.add(doc_id)

            logger.info("Successfully combined retrieval documents")
            retrieval_cache[cache_key] = combined_docs
            return {"context": combined_docs}

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {"context": []}

    def generate(state: State) -> dict:
        try:
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = [
                SystemMessage(
                    f"""
                    Role: You are a personal chatbot with the purpose: {purpose}.
                    Fluent in languages: {languages}.
                    Greeting: {greeting}
                    Conversation history: {conversation_history}
                    Company info: {company_info}
                    Context from documents: {docs_content}
                    Maintain tone/style: {tone_and_style}
                    Special keywords trigger connection to live agent.
                    """
                ),
                HumanMessage(state["question"])
            ]

            input_text = "".join([msg.content if hasattr(msg, "content") else "" for msg in messages])
            input_tokens = count_tokens(input_text)

            response = llm.invoke(messages)

            output_tokens = count_tokens(response.content)
            total_tokens = input_tokens + output_tokens

            update_token_usage_in_mongo(chatbot_id, total_tokens)
            current_usage = get_token_usage_from_mongo(chatbot_id)
            if current_usage > MAX_TOKEN_LIMIT:
                logger.warning(f"Token limit exceeded for chatbot {chatbot_id}")
                return {"answer": "Sorry, this chatbot has reached the token usage limit. Please try again later."}

            return {"answer": response.content}

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return {"answer": "Sorry, something went wrong generating the response."}

    try:
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        response = graph.invoke({"question": prompt})
        return response.get("answer", "No response generated.")
    except Exception as e:
        logger.error(f"Error in conversation graph: {e}")
        return f"An error occurred during conversation: {e}"
