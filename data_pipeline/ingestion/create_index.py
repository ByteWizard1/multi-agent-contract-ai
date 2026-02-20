import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get API key from environment (recommended)
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Create Pinecone client
pc = Pinecone(api_key=pinecone_api_key, environment="us-east1-gcp")  # adjust environment if needed

# Index details
index_name = "contract-clauses-dataset"

dimension = 1536  # matches Titan Text Embeddings V2

# Create index if it doesn't exist
existing_indexes = [idx.name for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index '{index_name}' already exists.")

# Connect to index
index = pc.Index(index_name)
print(f"Connected to index '{index_name}'")
