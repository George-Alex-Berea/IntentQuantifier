from sentence_transformers import SentenceTransformer
import lancedb
from lancedb.embeddings import get_registry
import numpy as np
import pandas as pd

df = pd.read_json("companies.jsonl", lines=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

'''sentences = [
   "TerraFuel Solutions este specializată în furnizarea și distribuția de combustibil de înaltă calitate, optimizat special pentru utilaje grele și echipamente de construcții. Înțelegem că timpul înseamnă bani, motiv pentru care am eliminat timpii morți cauzați de lipsa alimentării.",
   "Rompetrol is a Romanian company specialized in petroleum refining, petrochemical operations, and the distribution of fuel products. The company operates as a subsidiary of KMG International and manages integrated refineries in Romania, Moldova, Bulgaria, and Georgia, serving the automotive, industrial, and energy sectors. Rompetrol also provides industrial products, wholesale fuel supply, and e-Mobility services, and is certified for quality, health, safety, and environmental management.",
   "OMV PETROM Marketing SRL is a Romanian company engaged in the natural gas and oil industry, operating in the Exploration & Production, Refining & Marketing, and Chemicals sectors. The company provides products and services such as gasoline, diesel, aviation fuel, bitumen, and chemical products, and manages fuel stations and gas storage facilities across multiple European countries. OMV also offers e-mobility solutions and is involved in sustainability initiatives, including circular economy solutions and carbon emission reduction.",
   "S.C. European Drinks S.A., DBA European Drinks, is a Romanian company specialized in the production and distribution of mineral water, carbonated water, energy drinks, beer, and various soft drinks. The company operates as a holding with subsidiaries in the food and drink sector, serving both domestic and international markets. European Drinks also engages in community support initiatives and participates in international industry events."
   ]'''

embeddings = model.encode(df['description'].tolist())

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


while True:
    query = input("Enter a query (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    query_embedding = model.encode([query])[0]
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]

    # Get the indexes of the 25 most similar companies
    top_indices = np.argsort(similarities)[-25:][::-1]
    print("Top 25 similar companies:")
    for idx in top_indices:
        print(f"{df['name'][idx]}: {similarities[idx]:.4f}")