from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load a pre-trained model (all-MiniLM-L6-v2 is fast and accurate)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Define our phrases
sentences = [
   "TerraFuel Solutions este specializată în furnizarea și distribuția de combustibil de înaltă calitate, optimizat special pentru utilaje grele și echipamente de construcții. Înțelegem că timpul înseamnă bani, motiv pentru care am eliminat timpii morți cauzați de lipsa alimentării.",
   "Rompetrol is a Romanian company specialized in petroleum refining, petrochemical operations, and the distribution of fuel products. The company operates as a subsidiary of KMG International and manages integrated refineries in Romania, Moldova, Bulgaria, and Georgia, serving the automotive, industrial, and energy sectors. Rompetrol also provides industrial products, wholesale fuel supply, and e-Mobility services, and is certified for quality, health, safety, and environmental management.",
   "OMV PETROM Marketing SRL is a Romanian company engaged in the natural gas and oil industry, operating in the Exploration & Production, Refining & Marketing, and Chemicals sectors. The company provides products and services such as gasoline, diesel, aviation fuel, bitumen, and chemical products, and manages fuel stations and gas storage facilities across multiple European countries. OMV also offers e-mobility solutions and is involved in sustainability initiatives, including circular economy solutions and carbon emission reduction.",
   "S.C. European Drinks S.A., DBA European Drinks, is a Romanian company specialized in the production and distribution of mineral water, carbonated water, energy drinks, beer, and various soft drinks. The company operates as a holding with subsidiaries in the food and drink sector, serving both domestic and international markets. European Drinks also engages in community support initiatives and participates in international industry events."
   ]

# 3. Encode them into the embedding space (this returns a NumPy array)
embeddings = model.encode(sentences)

# 4. Calculate Cosine Similarity between the first two
# (Higher score = closer in direction)
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

sim_plant = cosine_similarity(embeddings[0], embeddings[1])  # Photosynthesis vs Plant
sim_pizza = cosine_similarity(embeddings[0], embeddings[2])  # Photosynthesis vs Pizza
sim_3 = cosine_similarity(embeddings[0], embeddings[3])  # Photosynthesis vs Pizza

print(f"Similarity 1: {sim_plant:.4f}")
print(f"Similarity 2: {sim_pizza:.4f}")
print(f"Similarity 3: {sim_3:.4f}")