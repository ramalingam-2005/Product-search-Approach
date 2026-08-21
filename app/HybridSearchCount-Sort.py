import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import re
from rapidfuzz import fuzz

df=pd.read_csv("app/b2b_product_search_lesson1.csv")
#----------------------------KeyWordMatching--------------------------
def normalize_text(text):
    text = str(text).lower()

    # Replace hyphens and special characters with spaces
    text = re.sub(r"[-_/]", " ", text)

    # Remove other punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
def tokenize(text):
    text = str(text).lower()

    tokens = re.findall(r"\b\w+\b", text)

    return tokens

def calculate_score(query, product_name):

    query_tokens = tokenize(query)
    product_tokens = tokenize(product_name)

    matches = 0

    for token in query_tokens:
        if token in product_tokens:
            matches += 1

    if len(query_tokens) == 0:
        return 0

    return matches / len(query_tokens)

def search(query):

    df["score"] = df["product_name"].apply(
    lambda x: calculate_score(query, x)
    )
    results = df[df["score"] > 0.5]

    results = results.sort_values(
    by="score",
    ascending=False
    )
    return results
#------------------------------Semantic Searching--------------------------
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
#---------------------------Typos---------------------------
def searchTypos(query, threshold=80):

    query = normalize_text(query)

    results = []

    for _, row in df.iterrows():

        product_name = normalize_text(row["product_name"])

        score = fuzz.token_set_ratio(
            query,
            product_name
        )

        if score >= threshold:
            results.append(
                {
                    "product_id": row["product_id"],
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "price": row["price"],
                    "MOQ": row["MOQ"],
                    "supplier": row["supplier"],
                    "location": row["location"],
                    "score": score
                }
            )

    return pd.DataFrame(results).sort_values(
        by="score",
        ascending=False
    )

def hybrid_search(query, top_k=5):

    # 1. Keyword / Exact matching
    exact_results = search(query)

    # 2. Fuzzy / Typo matching
    fuzzy_results = searchTypos(query)

    # 3. Semantic matching
    semantic_results = semantic_search(
        query,
        top_k=20
    )

    # -----------------------------------
    # UNION ALL product IDs
    # -----------------------------------

    all_results = pd.concat(
        [
            exact_results[["product_id"]],
            fuzzy_results[["product_id"]],
            semantic_results[["product_id"]]
        ],
        ignore_index=True
    )

    # -----------------------------------
    # GROUP BY product_id
    # COUNT occurrences
    # ORDER BY count DESC
    # -----------------------------------

    product_counts = (
        all_results
        .groupby("product_id")
        .size()
        .reset_index(name="match_count")
        .sort_values(
            by="match_count",
            ascending=False
        )
    )

    # -----------------------------------
    # Join with original dataframe
    # to get complete product details
    # -----------------------------------

    final_results = product_counts.merge(
        df,
        on="product_id",
        how="left"
    )

    return final_results.head(top_k)
query = "Stianles Steel Pipe"

results = hybrid_search(
    query,
    top_k=20
)

print(
    results[
        [
            "product_id",
            "product_name",
            "category",
            "match_count"
        ]
    ]
)