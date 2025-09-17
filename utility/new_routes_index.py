from .new_routes import onboard, scrapper,vector_embeddings,extract,llm_endpoint,get_status,faqs_endpoint,guidance_endpoint,handoff_guidance_endpoint,retrain_bot,welcome_message
from flask import Blueprint


routes = Blueprint('routes' ,__name__)

routes.route(("/Onboarding"), methods = ['POST', 'GET'])(onboard)
routes.route(("/webscrapper"), methods = ['POST', 'GET'])(scrapper)
routes.route(("/file_uploads"), methods = ['POST', 'GET'])(vector_embeddings)
routes.route(("/youtube_url"), methods = ['POST', 'GET'])(extract)
routes.route(("/llm"), methods = ['POST', 'GET'])(llm_endpoint)
routes.route(("/status"), methods = ['POST', 'GET'])(get_status)
routes.route(("/faqs"), methods = ['POST', 'GET'])(faqs_endpoint)
routes.route(("/guidance"), methods = ['POST', 'GET'])(guidance_endpoint)
routes.route(("/handoff-guidance"), methods = ['POST', 'GET'])(handoff_guidance_endpoint)
routes.route(("/retrain"), methods = ['POST', 'GET'])(retrain_bot)
routes.route(("/welcome_message"), methods = ['POST', 'GET'])(welcome_message)