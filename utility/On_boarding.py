from typing import List, TypedDict
from pydantic import BaseModel
from Databases.mongo import Bot_Retrieval
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from embeddings_creator import embeddings_from_gcb
import os 
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import getpass
import logging
from utility.retrain_bot import fetch_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.disable()
logger = logging.getLogger(__name__)


try:
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    load_dotenv()

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

llm = load_llm(api, model_provider, model)

converstation_state = {}

def chatbot(chatbot_id, version_id, prompt, user_id):
    try:

        request_body = {
        "chatbot_id": chatbot_id,
        "version_id": version_id,
        "collection_name": ["guidance", "handoff"]
        }


        guidelines = fetch_data(request_body)
        print("Guidelines generated succefull")
        Bot_information = Bot_Retrieval(chatbot_id, version_id)
        if not Bot_information:
            raise ValueError(f"No bot information found for chatbot_id {chatbot_id} and version_id {version_id}")

        if user_id not in converstation_state:
            converstation_state[user_id] = [{'role': 'user', 'content': prompt}]

        converstation_history = converstation_state[user_id]
        converstation_state[user_id].append({'role': 'user', 'content': prompt})

        greeting = Bot_information[0].get('greeting_message', "Hello!")
        purpose = Bot_information[0].get('purpose', "General assistance")
        languages = Bot_information[0].get('supported_languages', ["English"])
        tone_and_style = Bot_information[0].get('tone_style', "Friendly and professional")
        print("receive bot info")
        llm_response = Personal_chatbot(converstation_history, prompt, languages, purpose, tone_and_style, greeting,guidelines)
        converstation_state[user_id].append({'role': 'bot', 'content': llm_response})

        return llm_response

    except Exception as e:
        logger.error(f"Error in chatbot function: {e}")
        print("Error in chatbot function: {e}")
        print(e)
        return f"An error occurred: {e}"

def Personal_chatbot(converstation_history, prompt, languages, purpose, tone_and_style, greeting,guidelines):
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State):
        try:
            new_vector_store = FAISS.load_local("website_faiss_index", embeddings, allow_dangerous_deserialization=True)
            retrieved_docs = new_vector_store.similarity_search(state['question'])
            return {"context": retrieved_docs}
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
When the user greets you, start with: "{greeting}", and then introduce your purpose.
Always keep the conversation context in mind, including the chat history:
{converstation_history}
You also have access to context derived from document scores:
{docs_content}
Maintain a tone and style that aligns with the following guidelines:
{tone_and_style}
Please reply as "Would like to connect you to the live agent  for the following guidelines :
{guidelines}
"""
                ),
                HumanMessage(f"{state['question']}")
            ]
            response = llm.invoke(messages)
            return {"answer": response.content, 
                    }
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