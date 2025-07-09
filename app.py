import os
import traceback
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone

# Load environment variables
load_dotenv(dotenv_path=Path(".") / ".env")

# Debug print
pinecone_key = os.getenv("PINECONE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
print("DEBUG: PINECONE KEY =", pinecone_key)

if not pinecone_key or not openai_key:
    raise Exception("🔐 API keys are missing. Check your .env file.")

# Initialize clients
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("rag-consultant-demo")

print("Available indexes:", pc.list_indexes())

# Create Flask app
app = Flask(__name__)

@app.route("/rag", methods=["POST"])
def rag_query():
    try:
        query = request.json.get("question")
        if not query:
            return jsonify({"error": "No question provided."}), 400

        # Get embedding
        query_embed = client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding

        # Query Pinecone
        results = index.query(vector=query_embed, top_k=5, include_metadata=True)
        context = "\n\n".join([
            match.metadata.get("text", "") for match in results.matches if match.metadata
        ])

        # Build GPT prompt
        messages = [
            {"role": "system", "content": "Answer using only the context provided."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        # Generate response using new SDK
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )

        answer = completion.choices[0].message.content.strip()
        return jsonify({"answer": answer})

    except Exception as e:
        print("🔥 ERROR:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

