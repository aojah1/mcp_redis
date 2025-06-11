import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from llm.oci_embedding_model import initialize_embedding_model

# Ensure NLTK punkt tokenizer is available
nltk.download('punkt', quiet=True)


# ---------- BLEU Score Calculation ----------
def compute_bleu(reference: str, candidate: str) -> float:
    """Compute sentence-level BLEU with smoothing."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)


# ---------- ROUGE-L Score Calculation ----------

def compute_rouge_l(reference: str, candidate: str) -> float:
    # Initialize a scorer for ROUGE-L (with stemming)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    # fmeasure is the harmonic-mean F1 score
    rouge_l_f1 = scores['rougeL'].fmeasure
    return rouge_l_f1


# ---------- Cosine Similarity Calculation ----------
def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ---------- Example Usage (Banking/Insurance Domain) ----------

# Reference and candidate for translation-like example
ref_translation = "The customer wants to check their account balance."
candidate_good = "The customer wants to check account balance."
candidate_bad = "He is interested in credit card application."

# Reference and candidate for summary-like example
ref_summary = "Customer requests claim status and pending documents for policy coverage."
candidate_sum_good = "Customer asked for claim status and needed documents."
candidate_sum_bad = "Agent talked about marketing offers."

# Compute BLEU and ROUGE-L scores
bleu_good = compute_bleu(ref_translation, candidate_good)
bleu_bad = compute_bleu(ref_translation, candidate_bad)
rouge_good = compute_rouge_l(ref_translation, candidate_good)
rouge_bad = compute_rouge_l(ref_translation, candidate_bad)

bleu_sum_good = compute_bleu(ref_summary, candidate_sum_good)
bleu_sum_bad = compute_bleu(ref_summary, candidate_sum_bad)
rouge_sum_good = compute_rouge_l(ref_summary, candidate_sum_good)
rouge_sum_bad = compute_rouge_l(ref_summary, candidate_sum_bad)

# Prepare texts for embedding similarity
texts = [
    "Customer wants to check account balance.",
    "I need to see my balance details.",
    "She spent the afternoon sculpting a miniature orchid out of polymer clay."
]
embed_model = initialize_embedding_model()
embeddings = np.array(embed_model.embed_documents(texts))

cos_sim_good = compute_cosine_similarity(embeddings[0], embeddings[1])
cos_sim_bad = compute_cosine_similarity(embeddings[0], embeddings[2])

# Display results
print("=== Translation BLEU & ROUGE-L Scores (Banking) ===")
print(f"Good Candidate BLEU: {bleu_good:.4f}, ROUGE-L: {rouge_good:.4f}")
print(f"Bad Candidate BLEU:  {bleu_bad:.4f}, ROUGE-L:  {rouge_bad:.4f}\n")

print("=== Summary BLEU & ROUGE-L Scores (Insurance) ===")
print(f"Good Summary BLEU: {bleu_sum_good:.4f}, ROUGE-L: {rouge_sum_good:.4f}")
print(f"Bad Summary BLEU:  {bleu_sum_bad:.4f}, ROUGE-L:  {rouge_sum_bad:.4f}\n")

print("=== Cosine Similarities (Cohere Embeddings) ===")
print(f"Similar Texts Cosine:    {cos_sim_good:.4f}  (good)")
print(f"Dissimilar Texts Cosine: {cos_sim_bad:.4f}  (bad)")