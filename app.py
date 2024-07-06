from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load pre-trained Sentence Transformers model
model_name = "roberta-base-nli-mean-tokens"
model = SentenceTransformer(model_name)

def is_valid_input(data):
    """Basic input validation."""
    if not isinstance(data, dict):
        return False
    if "student_answer" not in data or "professor_answer" not in data:
        return False
    if not isinstance(data["student_answer"], str) or not isinstance(data["professor_answer"], str):
        return False
    return True

@app.route("/calculate_similarity", methods=["POST"])
def calculate_similarity_en():
    data = request.json
    if not is_valid_input(data):
        return jsonify({"error": "Invalid input"}), 400

    student_answers = data["student_answer"]
    professor_answers = data["professor_answer"]

    # Generate embeddings for English sentences
    embeddings = model.encode([student_answers, professor_answers])

    # Compute cosine similarity between the embeddings
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    similarity_scores = {"similarity_score": similarity_score.item()}

    return jsonify(similarity_scores)