from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Enhanced insurance policy knowledge base with metadata
policy_documents = [
    {
        "text": "Health insurance covers hospital stays up to 30 days per year",
        "approval": "Approved",
        "next_steps": "Submit hospital admission documents within 30 days of discharge",
        "category": "hospitalization"
    },
    {
        "text": "Dental procedures have limited coverage with a maximum of $1,500 per year",
        "approval": "Partially Approved",
        "next_steps": "Get pre-authorization for major dental work",
        "category": "dental"
    },
    {
        "text": "Pre-existing conditions are not covered during the first 12 months of the policy",
        "approval": "Rejected",
        "next_steps": "Reapply after 12 months of continuous coverage",
        "category": "pre-existing"
    },
    {
        "text": "Maternity benefits require a 12-month waiting period before coverage begins",
        "approval": "Conditionally Approved",
        "next_steps": "Submit proof of 12-month policy duration",
        "category": "maternity"
    },
    {
        "text": "Knee surgeries are covered after 6 months of continuous coverage",
        "approval": "Approved",
        "next_steps": "Submit surgeon's recommendation and MRI reports",
        "category": "surgery"
    },
    {
        "text": "Emergency room visits are covered with a $250 copayment",
        "approval": "Approved",
        "next_steps": "Pay copayment at time of service",
        "category": "emergency"
    },
    {
        "text": "Prescription drugs are covered with a tiered copayment system",
        "approval": "Approved",
        "next_steps": "Present insurance card at pharmacy",
        "category": "pharmacy"
    },
    {
        "text": "Mental health services are covered for up to 20 sessions per year",
        "approval": "Approved",
        "next_steps": "Get referral from primary care physician",
        "category": "mental-health"
    }
]

# Extract just the text for embeddings
documents = [doc["text"] for doc in policy_documents]

# Initialize embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    query = request.json.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Encode query and find similar documents
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)
        most_similar_idx = np.argmax(similarities)
        similarity_score = similarities[0][most_similar_idx]
        
        # Get full document info
        result = policy_documents[most_similar_idx]
        
        return jsonify({
            "query": query,
            "policy_text": result["text"],
            "approval_status": result["approval"],
            "next_steps": result["next_steps"],
            "category": result["category"],
            "similarity_score": float(similarity_score),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)