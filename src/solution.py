import lancedb
from google import genai
import json

with open(".api_key.txt", "r") as f:
    api_key = f.read().strip()
client = genai.Client(api_key=api_key)

db = lancedb.connect("./company_emb_database")
table = db.open_table("company_descriptions")

print("\n--- Company Matcher Ready ---")
while True:
    user_query = input("\nDescribe the kind of company you are looking for (or 'exit'): ")
    if user_query.lower() == 'exit':
        break

    prompt = f"""
    Write an entry describing a hypothetical company which would provide the necessary products or services 
    for a user searching for: {user_query}. 

    Only provide the entry, no other text. The entry MUST be valid JSON formatted exactly like this:
    {{
        "website": "",
        "operational_name": "",
        "year_founded": "",
        "address": "",
        "employee_count": "",
        "revenue": "",
        "primary_naics": "",
        "description": "",
        "business_model": [],
        "target_markets": [],
        "core_offerings": [],
        "is_public": boolean,
        "secondary_naics": ""
    }}

    The description field should be a concise summary of the company and its target market.

    The business_model field should be a list of strings describing the company's business model 
    (e.g. "B2B", "SaaS", "Subscription", etc.).

    The target_markets field should be a list of strings describing the company's target markets 
    (e.g. "Small Businesses", "Healthcare", "E-commerce", etc.).

    The core_offerings field should be a list of strings describing the company's core products or services 
    (e.g. "Cloud Storage", "AI Consulting", "E-commerce Platform", etc.).

    Other fields should be filled in with plausible values based on the description, 
    or be left uncompletted if there is no plausible value.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
    except Exception as e:
        print(f"Error generating company description: {e}")
        continue
    
    target_description = response.text.strip()
    print(f"\nPowerfull LLM description:\n{target_description}")

    results = (
        table.search(target_description)
        .metric("cosine")
        .limit(50)
        .to_pandas()
    )

    print(f"\n--- Top 25 Matches Found ---")
    for i, row in results.iterrows():
        company_info = row['data']
        
        name = company_info.get('name', 'N/A')
        website = company_info.get('website', 'N/A')
        
        score = 1 - row['_distance']
        
        print(f"{i+1}. Score: [{score:.4f}] Name: {name} | Website: {website}")