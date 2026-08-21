import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# Load dataset
df = pd.read_csv(
    "E:/PROJECTS/Search Engine/app/b2b_product_search_lesson1.csv"
)


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Convert product names into a list
product_names = df["product_name"].fillna("").tolist()


# Create embeddings for all products
product_embeddings = model.encode(
    product_names,
    normalize_embeddings=True
)


def semantic_search(query, top_k=5):

    # Convert user query into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # Calculate similarity
    scores = cos_sim(
        query_embedding,
        product_embeddings
    )[0]

    # Get indexes of top matching products
    top_indices = scores.argsort(
        descending=True
    )[:top_k]

    # Get matching rows
    results = df.iloc[
        top_indices.cpu().numpy()
    ].copy()

    # Add similarity score
    results["semantic_score"] = scores[
        top_indices
    ].cpu().numpy()

    return results


# Test
results = semantic_search(
    "stainless metal tube",
    top_k=5
)

print(
    results[
        [
            "product_id",
            "product_name",
            "category",
            "price",
            "semantic_score"
        ]
    ]
)