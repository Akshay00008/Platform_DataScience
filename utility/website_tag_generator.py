from langchain_openai import ChatOpenAI
import json
import logging

# Optional: Enable logging
logging.basicConfig(level=logging.INFO)

def generate_structured_prompt(json_data):
    try:
        json_preview = json.dumps(json_data[:10], indent=2)  # Safely preview a portion
    except Exception as e:
        logging.error(f"Error processing JSON: {e}")
        json_preview = "{}"

    return f"""
You are an expert Information Architect and Taxonomist. Your task is to deeply analyze the provided website content in JSON format and transform it into a clearly organized, human-readable structure tailored to the specific domain of the website. This transformation should enhance content discoverability, improve navigation, and reflect best practices in modern web taxonomy.

Analyze the content and generate 4 to 5 high-level main categories, each containing 3 to 5 subcategories, followed by a relevant FAQ section. All content should be specific, concise, and adapted to the nature of the website.

1️⃣ CATEGORY IDENTIFICATION  
- Identify the website’s industry type based on its content. Choose from (but not limited to):  
  Manufacturing, Restaurant, Education, Healthcare, E-commerce, Hospitality, SaaS, Non-Profit, Government, Automotive, Real Estate, Fitness, Entertainment, Financial Services, Travel  
- Based on the identified type, define 4 to 5 relevant MAIN CATEGORIES that best represent the website’s structure. These categories should reflect how a real-world website in that industry is typically organized.  
  - Examples:  
    - *Manufacturing*: Products, Capabilities, Industries Served, Compliance  
    - *Restaurant*: Menu, Location, Services, Chef’s Specials  
    - *SaaS*: Features, Pricing, Integrations, Use Cases  

2️⃣ STRUCTURED EXTRACTION  
For each main category:  
- Identify 3 to 5 meaningful subcategories or content segments.  
- Extract only the most relevant and useful information under each subcategory.  
- Do not include filler or marketing fluff — focus on clarity and specificity.  
- Use the following strict format to present the output:

[Main Category Name]  
├─ [Subcategory 1]: [Item 1], [Item 2], [Item 3]  
├─ [Subcategory 2]: [Concise description]  
└─ [Subcategory 3]: [Data point or key info]

3️⃣ FAQ GENERATION  
Create or extract 2 to 4 relevant FAQs based on the content. These should reflect actual user questions a visitor might have. Use this format:

FAQs:  
- Q: [Question related to content, product, service, etc.]  
  A: [Direct, informative answer]  
- Q: [Generate a logical question if missing]  
  A: [Accurate answer inferred from context]

FAQs should cover areas like services, product delivery, pricing, contact, customization, or technical support. If the JSON does not contain questions, generate them logically.

📌 RULES  
- Adapt categories/subcategories to fit the actual data — do not force a rigid structure.  
- Avoid placeholders, markdown symbols, or extraneous formatting.  
- Maintain strict indentation and bullet hierarchy as shown above.  
- Output only the structured content — no markdown headers, explanations, or enclosing tags.  
- Do not include any backticks, markdown code blocks, or JSON formatting (` ```json `).  
- Prioritize clarity, usability, and clean organization.  

🧠 Purpose: The goal is to transform raw, unstructured website content into a logically categorized structure that supports user understanding, content planning, and future integration into knowledge systems or CMS platforms.

---  




JSON Data:
{json_preview}
"""

def new_generate_tags_from_gpt(json_data):
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        prompt = generate_structured_prompt(json_data)
        response = llm.predict(prompt)

        if not response:
            logging.warning("No response returned from model.")
            print("No response returned from model.")
            return {"error": "Empty response from GPT model"}
        
        return {"tags": response}

    except Exception as e:
        logging.exception("Error generating tags from GPT.")
        print("Error generating tags from GPT.")
        return {"error": str(e)}
