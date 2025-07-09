import os
import traceback
from flask import Flask, request, jsonify
from openai import OpenAI
from pinecone import Pinecone

# Load .env only if running locally
if os.environ.get("RENDER") != "true":
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(dotenv_path=Path(".") / ".env")
    except ImportError:
        print("Skipping .env loading in production")

# Load API keys
pinecone_key = os.getenv("PINECONE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if not pinecone_key or not openai_key:
    raise Exception("🔐 API keys are missing. Check your .env file or Render environment settings.")

# Initialize clients
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("rag-consultant-demo")

# Create Flask app
app = Flask(__name__)

@app.route("/rag", methods=["POST"])
def rag_query():
    try:
        query = request.json.get("question")
        if not query:
            return jsonify({"error": "No question provided."}), 400

        # Get embedding from OpenAI
        query_embed = client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding

        # Query Pinecone
        results = index.query(vector=query_embed, top_k=2, include_metadata=True)

        # Build compact context safely
        chunks = []
        max_total_chars = 1200  # ultra-safe
        current_total = 0

        for match in results.matches:
            if match.metadata and "text" in match.metadata:
                chunk = match.metadata["text"].strip()[:300]
                if current_total + len(chunk) > max_total_chars:
                    break
                chunks.append(chunk)
                current_total += len(chunk)

        context = "\n\n".join(chunks)

        if not context:
            return jsonify({"answer": "No relevant context found."})

        # Build GPT prompt
        messages = [
            {"role": "system", "content": "Answer using only the context provided."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        # Call OpenAI
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )

        answer = completion.choices[0].message.content.strip()

        # Final safety cap for answer size
        if len(answer) > 1000:
            answer = answer[:1000] + "..."

        # Debug print to logs
        print(f"✔️ Context chars: {len(context)} | Answer chars: {len(answer)}")

        return jsonify({"answer": answer})

    except Exception as e:
        print("🔥 ERROR:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


    except Exception as e:
        print("🔥 ERROR:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
