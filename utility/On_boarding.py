import os
import getpass
import logging
from typing import List, TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from utility.retrain_bot import fetch_data
from Databases.mongo import Bot_Retrieval, company_Retrieval

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

try:
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    api = os.getenv('OPENAI_API_KEY')
    model = os.getenv('GPT_model')
    model_provider = os.getenv('GPT_model_provider')
    if not api or not model or not model_provider:
        raise ValueError("Please check the OPENAI_API_KEY, GPT_model, and GPT_model_provider.")

except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise


def load_llm(key, model_provider, model_name):
    try:
        if not all([key, model_provider, model_name]):
            raise ValueError("Missing LLM configuration in secrets.")
        os.environ["API_KEY"] = key
        return init_chat_model(model_name, model_provider=model_provider)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


# Load the LLM (Large Language Model)
llm = load_llm(api, model_provider, model)

# Conversation state to maintain user history
converstation_state = {}

def chatbot(chatbot_id, version_id, prompt, user_id):
    try:
        request_body = {
            "chatbot_id": chatbot_id,
            "version_id": version_id,
            "collection_name": ["guidance", "handoff"]
        }

        guidelines = fetch_data(request_body)
        print("Guidelines generated successfully")

        # Retrieve bot information from the database
        Bot_information = Bot_Retrieval(chatbot_id, version_id)
        if not Bot_information:
            raise ValueError(f"No bot information found for chatbot_id {chatbot_id} and version_id {version_id}")

        bot_company= company_Retrieval()
        if not bot_company:
            raise ValueError(f"No bot company information found ")

        # Initialize conversation history if not already present
        if user_id not in converstation_state:
            converstation_state[user_id] = [{'role': 'user', 'content': prompt}]

        converstation_history = converstation_state[user_id]
        converstation_state[user_id].append({'role': 'user', 'content': prompt})

        greeting = Bot_information[0].get('greeting_message', "Hello!")
        purpose = Bot_information[0].get('purpose', "General assistance")
        languages = Bot_information[0].get('supported_languages', ["English"])
        tone_and_style = Bot_information[0].get('tone_style', "Friendly and professional")
        company_info=bot_company[0].get('bot_company')
        print("Received bot info")

        # Avoid adding empty guidelines to the response
        if guidelines.get('guidanceflows') or guidelines.get('handoffscenarios'):
            llm_response = Personal_chatbot(converstation_history, prompt, languages, purpose, tone_and_style, greeting, guidelines,company_info,chatbot_id,version_id)
        else:
            llm_response = Personal_chatbot(converstation_history, prompt, languages, purpose, tone_and_style, greeting, guidelines,company_info,chatbot_id,version_id)

        converstation_state[user_id].append({'role': 'bot', 'content': llm_response})

        return llm_response

    except Exception as e:
        logger.error(f"Error in chatbot function: {e}")
        return f"An error occurred: {e}"


def Personal_chatbot(converstation_history, prompt, languages, purpose, tone_and_style, greeting, guidelines,company_info,chatbot_id,version_id):
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

        
   
    def retrieve(state: State):
        try:
            # faiss_index_dir = "C:\\Users\\hp\\Desktop\\Platform_16-05-2025\\faiss_indexes"  # Adjust for your environment
            faiss_index_dir = "/home/bramhesh_srivastav/Platform_DataScience/faiss_indexes"  # For Linux/Mac

            # Construct unique filenames for both FAISS indexes
            faiss_index_filename = f"{chatbot_id}_{version_id}_faiss_index"
            faiss_index_website = f"{chatbot_id}_{version_id}_faiss_index_website"

            # Construct the full paths to the FAISS index files
            faiss_path_1 = os.path.join(faiss_index_dir, faiss_index_filename)
            faiss_path_2 = os.path.join(faiss_index_dir, faiss_index_website)

            # Check if the first FAISS index exists
            if not os.path.exists(faiss_path_1):
                raise FileNotFoundError(f"FAISS index not found at {faiss_path_1}")

            # Check if the second FAISS index exists
            if not os.path.exists(faiss_path_2):
                raise FileNotFoundError(f"FAISS index not found at {faiss_path_2}")

            # Load both FAISS indexes
            logger.info(f"Loading FAISS index from {faiss_path_1}")
            new_vector_store = FAISS.load_local(faiss_path_1, embeddings, allow_dangerous_deserialization=True)

            logger.info(f"Loading FAISS index from {faiss_path_2}")
            new_vector_store_1 = FAISS.load_local(faiss_path_2, embeddings, allow_dangerous_deserialization=True)

            logger.info("FAISS indexes loaded successfully.")

            # Retrieve documents from both FAISS indices
            retrieved_docs = new_vector_store.similarity_search(state['question'])
            retrieved_docs_2 = new_vector_store_1.similarity_search(state['question'])

            logger.info("retrieved_docs indexes completed successfully.")

            # Combine results from both indices, ensuring no duplicates based on document ID
            combined_docs = []

            # Using a set to track unique document IDs (or another unique identifier)
            seen_docs = set()

            # Add documents from the first index
            for doc in retrieved_docs:
                # Assuming 'doc.id' is unique or 'doc.content' for uniqueness
                if doc.id not in seen_docs:
                    combined_docs.append(doc)
                    seen_docs.add(doc.id)  # Use doc.id or another identifier to ensure uniqueness

            # Add documents from the second index
            for doc in retrieved_docs_2:
                if doc.id not in seen_docs:
                    combined_docs.append(doc)
                    seen_docs.add(doc.id)

            logger.info("combined_docs returned successfully.")



            # print("context:", combined_docs)
            return {"context": combined_docs}

        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return {"context": []}

    def generate(state: State):
        try:
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = [
                SystemMessage(
                    f"""
                    Role: You are a personal chatbot with the following purpose: {purpose}.
                    You can communicate fluently in the following languages: {languages}.
                    {greeting} Always keep the conversation context in mind, including the chat history:
                    {converstation_history}
                    Do not respond to the query or question which is not related to the company : {company_info}
                    You also have access to context derived from document scores:
                    {docs_content}
                    Maintain a tone and style that aligns with the following guidelines:{guidelines}
                    Maintain a tone and style : {tone_and_style}
                    --When user asks about "delivery status" or "Order status" or 
                    --"Warranty and Service Requests or Order and Shipping Queries" or 
                    --"Specialized Product Configurations[example : Schematic diagrams or wiring diagram]" or 
                    --"Payment or Billing Issues" or
                    --"if user says not working " or 
                    --"user says These words ["fire", "melt", "burned", "melted", "burned up] and add synonyms of these or related word's" or 
                    --" If user asks about the new product or product not present in your list" or
                    --" if you find the words like [price, pricing,burned up] word in user query " or 
                    --"When the user explicitly asks to speak with a live agent or mentions they are unable to resolve their issue with the chatbot alone " 
                    -- for the above mention cases please staright away say or responsd with the following message "Let's get you connected to one of our live agents so they can assist you further. Would it be okay if I connect you now?"
                     """
                ),
                HumanMessage(f"{state['question']}")
            ]
            response = llm.invoke(messages)

            
            return {"answer": response.content}

        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return {"answer": "Sorry, something went wrong in generating a response."}

    try:
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        # Invoke the state machine with the prompt
        response = graph.invoke({"question": prompt})

        return response.get('answer', "No response generated.")
    except Exception as e:
        logger.error(f"Error in conversation graph: {e}")
        return f"An error occurred during conversation: {e}"




#     def retrieve(state: State):
#         try:
#             new_vector_store = FAISS.load_local("website_faiss_index", embeddings, allow_dangerous_deserialization=True)
#             new_vector_store_1=FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#             retrieved_docs = new_vector_store.similarity_search(state['question'])
#             retrieved_docs_2 = new_vector_store_1.similarity_search(state['question'])
#             return {"context": (retrieved_docs)}
#         except Exception as e:
#             logger.error(f"Error in document retrieval: {e}")
#             return {"context": []}

#     def generate(state: State):
#         try:
     
#             docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#             messages = [
#                 SystemMessage(
#                     f"""
# Role: You are a personal chatbot with the following purpose: {purpose}.
# You can communicate fluently in the following languages: {languages}.
# When the user greets you, start with: "{greeting}", and then introduce your purpose.
# Always keep the conversation context in mind, including the chat history:
# {converstation_history}
# You also have access to context derived from document scores:
# {docs_content}
# Maintain a tone and style that aligns with the following guidelines:
# {tone_and_style}
# Please reply as "Would like to connect you to the live agent  for the following guidelines :
# {guidelines}
# """
#                 ),
#                 HumanMessage(f"{state['question']}")
#             ]
#             response = llm.invoke(messages)
#             return {"answer": response.content, 
#                     }
#         except Exception as e:
#             logger.error(f"Error in LLM generation: {e}")
#             return {"answer": "Sorry, something went wrong in generating a response."}

#     try:
#         graph_builder = StateGraph(State).add_sequence([retrieve, generate])
#         graph_builder.add_edge(START, "retrieve")
#         graph = graph_builder.compile()
#         response = graph.invoke({"question": prompt})
#         return response.get('answer', "No response generated.")
#     except Exception as e:
#         logger.error(f"Error in conversation graph: {e}")
#         return f"An error occurred during conversation: {e}"