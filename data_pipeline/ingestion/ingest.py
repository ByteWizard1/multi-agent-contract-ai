import pandas as pd
from pinecone import Pinecone, ServerlessSpec
import boto3
import json
import os
from dotenv import load_dotenv
import ftfy
import time

# ----------------------------------------
# Load environment variables
# ----------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# ----------------------------------------
# Pinecone Setup (1536 dim Titan)
# ----------------------------------------
index_name = "contract-clauses-dataset"
embed_dim = 1536

pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embed_dim,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# ----------------------------------------
# Load dataset & sanitize data
# ----------------------------------------
df = pd.read_csv("legal_docs.csv")
df['totalwords'] = df['totalwords'].fillna(0).astype(int)
df['totalletters'] = df['totalletters'].fillna(0).astype(int)

# ----------------------------------------
# Bedrock Titan V1 Setup
# ----------------------------------------
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def titan_embed(text: str):
    payload = {
        "inputText": text
    }

    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    body = json.loads(response['body'].read())
    return body['embedding']


# ----------------------------------------
# Ingest + Debug Logging + JSON Export
# ----------------------------------------
batch = []
BATCH_SIZE = 50
debug_log = []

for idx, row in df.iterrows():
    clause_text = ftfy.fix_text(str(row['clause_text']))
    clause_type = str(row['clause_type'])
    totalwords = int(row['totalwords'])
    totalletters = int(row['totalletters'])

    # --- Debug print ---
    print(f"\n[{idx}] CLAUSE_TEXT:\n{clause_text}\n{'-'*100}")

    embedding = titan_embed(clause_text)

    metadata = {
        "clause_text": clause_text,
        "clause_type": clause_type,
        "totalwords": totalwords,
        "totalletters": totalletters
    }

    batch.append((str(idx), embedding, metadata))

    # add to debug file
    debug_log.append({
        "id": str(idx),
        "clause_text": clause_text,
        "clause_type": clause_type,
        "totalwords": totalwords,
        "totalletters": totalletters
    })

    if len(batch) >= BATCH_SIZE:
        index.upsert(vectors=batch)
        batch = []
        time.sleep(0.4)

if batch:
    index.upsert(vectors=batch)

# save preview json
with open("ingested_preview.json", "w", encoding="utf-8") as f:
    json.dump(debug_log, f, indent=2, ensure_ascii=False)

print("\n📄 Debug saved to ingested_preview.json")
print("✅ Completed ingestion: Titan V1 (1536 dims) → Pinecone")
