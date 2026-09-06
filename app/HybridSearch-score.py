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
#------------------------------Semantic Searching--------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# Product names
product_names = df["product_name"].fillna("").tolist()

# Whole-product embeddings
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
    print(results)
    return results
#---------------------------Typos---------------------------
def calculate_fuzzy_score(query, product_name):

    query_tokens = tokenize(query)
    product_tokens = tokenize(product_name)

    if not query_tokens or not product_tokens:
        return 0

    total_score = 0

    for q_token in query_tokens:

        best_score = max(
            fuzz.ratio(q_token, p_token)
            for p_token in product_tokens
        )

        total_score += best_score

    return total_score / len(query_tokens)


def searchTypos(query, threshold=70):

    results = []

    for _, row in df.iterrows():

        score = calculate_fuzzy_score(
            query,
            row["product_name"]
        )

        if score >= threshold:

            results.append({
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "price": row["price"],
                "MOQ": row["MOQ"],
                "supplier": row["supplier"],
                "location": row["location"],
                "score": score
            })

    return pd.DataFrame(results).sort_values(
        by="score",
        ascending=False
    )
def hybrid_search(
    query,
    top_k=5,
    keyword_weight=0.3,
    fuzzy_weight=0.5,
    semantic_weight=0.2
):

    # -----------------------------------
    # 1. Keyword Search
    # -----------------------------------

    keyword_results = search(query)

    keyword_results = keyword_results[
        ["product_id", "score"]
    ].rename(
        columns={"score": "keyword_score"}
    )

    keyword_results = keyword_results.drop_duplicates(
        subset=["product_id"]
    )
    print(keyword_results)

    # -----------------------------------
    # 2. Fuzzy Search
    # -----------------------------------

    fuzzy_results = searchTypos(query)

    fuzzy_results = fuzzy_results[
        ["product_id", "score"]
    ].rename(
        columns={"score": "fuzzy_score"}
    )

    fuzzy_results["fuzzy_score"] = (
        fuzzy_results["fuzzy_score"] / 100
    )

    fuzzy_results = fuzzy_results.drop_duplicates(
        subset=["product_id"]
    )
    print(fuzzy_results)

    # -----------------------------------
    # 3. Semantic Search
    # -----------------------------------

    semantic_results = semantic_search(
        query,
        top_k=20
    )

    semantic_results = semantic_results[
        ["product_id", "semantic_score"]
    ]

    semantic_results = semantic_results.drop_duplicates(
        subset=["product_id"]
    )

    # -----------------------------------
    # UNION of all product IDs
    # -----------------------------------

    all_results = pd.concat(
        [
            keyword_results[["product_id"]],
            fuzzy_results[["product_id"]],
            semantic_results[["product_id"]]
        ],
        ignore_index=True
    ).drop_duplicates()

    # -----------------------------------
    # Merge all scores
    # -----------------------------------

    final_results = all_results.merge(
        keyword_results,
        on="product_id",
        how="left"
    )

    final_results = final_results.merge(
        fuzzy_results,
        on="product_id",
        how="left"
    )

    final_results = final_results.merge(
        semantic_results,
        on="product_id",
        how="left"
    )

    # Missing scores become 0
    final_results = final_results.fillna(0)

    # -----------------------------------
    # Weighted Fusion
    # -----------------------------------

    final_results["final_score"] = (
        final_results["keyword_score"] * keyword_weight
        +
        final_results["fuzzy_score"] * fuzzy_weight
        +
        final_results["semantic_score"] * semantic_weight
    )
    print(
        final_results[
            [
                "product_id",
                "keyword_score",
                "fuzzy_score",
                "semantic_score",
                "final_score"
            ]
        ].sort_values(
            "final_score",
            ascending=False
        ).head(30)
    )

    # -----------------------------------
    # Rank results
    # -----------------------------------

    final_results = final_results.sort_values(
        by="final_score",
        ascending=False
    )

    # -----------------------------------
    # Get product details
    # -----------------------------------

    final_results = final_results.merge(
        df.drop(columns=["score"], errors="ignore"),
        on="product_id",
        how="left"
    )
   

    return final_results.head(top_k)
# query = "Stianles Steel Pipe"

# results = hybrid_search(
#     query,
#     top_k=20
# )

# print(
#     results[
#         [
#             "product_id",
#             "product_name",
#             "keyword_score",
#             "fuzzy_score",
#             "semantic_score",
#             "final_score"
#         ]
#     ]
# )
query = "Stianles Steel Pipe"

print("KEYWORD")
print(
    search(query)[
        ["product_id", "product_name", "score"]
    ].head(20)
)

print("\nFUZZY")
print(
    searchTypos(query)[
        ["product_id", "product_name", "score"]
    ].head(20)
)

print("\nSEMANTIC")
print(
    semantic_search(query, top_k=20)[
        ["product_id", "product_name", "semantic_score"]
    ].head(20)
)