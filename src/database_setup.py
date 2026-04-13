import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
import json
from typing import Optional

db = lancedb.connect("./company_emb_database")
emb_model = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")

class CompanyInfo(LanceModel):
    name: Optional[str] = None
    website: Optional[str] = None

class Phrases(LanceModel):
    text: str = emb_model.SourceField()
    vector: Vector(emb_model.ndims()) = emb_model.VectorField()
    data: CompanyInfo

table = db.create_table("company_descriptions", schema=Phrases, mode="overwrite")

def stream_jsonl(filename):
    with open(filename, "r") as f:
        for line in f:
            raw_json = line.strip()
            parsed_json = json.loads(raw_json)
            if raw_json:
                yield {
                "text": raw_json, 
                "data": {
                    "name": parsed_json.get("operational_name"), 
                    "website": parsed_json.get("website")
                }
            }

print("\n--- Populating Database ---")
print("Could take a moment depending on the size of the dataset and your machine's capabilities...")

company_jsonl_path = input("Enter the path to the company descriptions JSONL file: ")
table.add(list(stream_jsonl(company_jsonl_path)))
print("Data added to the database.")

print("Creating index for cosine similarity search...")
table.create_index(metric="cosine")

print("\n--- Database Setup Complete ---")