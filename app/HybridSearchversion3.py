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

# Product names
product_names = df["product_name"].fillna("").tolist()

# Whole-product embeddings
product_embeddings = model.encode(
    product_names,
    normalize_embeddings=True
)


#------------------------------Semantic + Word-Level Fuzzy--------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

product_names = df["product_name"].fillna("").tolist()


def semantic_search(
    query,
    top_k=5,
    typo_threshold=0.95,
    fuzzy_weight_typo=0.5,
    semantic_weight_typo=0.5,
    fuzzy_weight_normal=0.5,
    semantic_weight_normal=0.5
):

    query_words = tokenize(query)

    if not query_words:
        return pd.DataFrame()

    # Encode query words once
    query_embeddings = model.encode(
        query_words,
        normalize_embeddings=True
    )

    product_scores = []

    for product_name in product_names:

        product_words = tokenize(product_name)

        if not product_words:
            product_scores.append(0)
            continue

        # Encode product words
        product_embeddings_words = model.encode(
            product_words,
            normalize_embeddings=True
        )

        word_scores = []

        # -----------------------------------
        # Process each query word
        # -----------------------------------

        for query_idx, query_word in enumerate(query_words):

            query_embedding = query_embeddings[query_idx]

            best_combined_score = 0

            # Compare query word with every product word
            for product_idx, product_word in enumerate(product_words):

                # -------------------------------
                # Fuzzy score
                # -------------------------------

                fuzzy_score = (
                    fuzz.ratio(
                        query_word,
                        product_word
                    ) / 100
                )

                # -------------------------------
                # Semantic score
                # -------------------------------

                semantic_score = cos_sim(
                    query_embedding,
                    product_embeddings_words[product_idx]
                ).item()

                # Cosine similarity can sometimes be negative
                semantic_score = max(
                    0,
                    semantic_score
                )

                # -------------------------------
                # Dynamic weighting
                # -------------------------------

                # If fuzzy score is less than 95%,
                # treat it as a possible typo.
                if fuzzy_score < typo_threshold:

                    word_score = (
                        fuzzy_weight_typo * fuzzy_score
                        +
                        semantic_weight_typo * semantic_score
                    )

                else:

                    # Normal/exact word:
                    # give semantic more importance.
                    word_score = (
                        fuzzy_weight_normal * fuzzy_score
                        +
                        semantic_weight_normal * semantic_score
                    )

                # Keep the best matching product word
                best_combined_score = max(
                    best_combined_score,
                    word_score
                )

            word_scores.append(
                best_combined_score
            )

        # -----------------------------------
        # Final product semantic score
        # -----------------------------------

        if word_scores:
            final_score = sum(word_scores) / len(word_scores)
        else:
            final_score = 0

        product_scores.append(final_score)

    # -----------------------------------
    # Create results
    # -----------------------------------

    results = df.copy()

    results["semantic_score"] = product_scores

    # Sort
    results = results.sort_values(
        by="semantic_score",
        ascending=False
    )

    return results.head(top_k)
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
    keyword_weight=0.5,
  
    semantic_weight=0.5
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
        final_results["semantic_score"] * semantic_weight
    )
    print(
        final_results[
            [
                "product_id",
                "keyword_score",
                
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
query = "Citirc Acid Powder"

results = hybrid_search(
    query,
    top_k=20
)

print(
    results[
        [
            "product_id",
            "product_name",
            "keyword_score",
          
            "semantic_score",
            "final_score"
        ]
    ]
)
# query = "Stianles Steel Pipe"

# print("KEYWORD")
# print(
#     search(query)[
#         ["product_id", "product_name", "score"]
#     ].head(20)
# )

# print("\nFUZZY")
# print(
#     searchTypos(query)[
#         ["product_id", "product_name", "score"]
#     ].head(20)
# )

# print("\nSEMANTIC")
# print(
#     semantic_search(query, top_k=20)[
#         ["product_id", "product_name", "semantic_score"]
#     ].head(20)
# )