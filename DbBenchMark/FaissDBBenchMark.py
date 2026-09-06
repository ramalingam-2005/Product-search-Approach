import time
import numpy as np
import pandas as pd
import faiss

import re
from collections import Counter
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PRODUCT_FILE = "app/b2b_product_search_lesson1.csv"
TEST_FILE = "app/test_dataset_top2_products.csv"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

TOP_K = 10
SPELLING_THRESHOLD = 70


# ============================================================
# 1. LOAD DATA
# ============================================================

products = pd.read_csv(PRODUCT_FILE)
tests = pd.read_csv(TEST_FILE)

print("Products:", len(products))
print("Test queries:", len(tests))

print("\nProduct columns:")
print(products.columns.tolist())

print("\nTest columns:")
print(tests.columns.tolist())


# ============================================================
# 2. PREPARE PRODUCT TEXT
# ============================================================

TEXT_COLUMNS = [
    "product_name",
    "category",
    "brand",
    "description"
]

existing_columns = [
    col for col in TEXT_COLUMNS
    if col in products.columns
]

products["search_text"] = (
    products[existing_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
)

print("\nUsing product columns:", existing_columns)


# ============================================================
# 3. BUILD VOCABULARY FOR SPELLING CORRECTION
# ============================================================

print("\nBuilding spelling vocabulary...")

word_frequency = Counter()

for column in existing_columns:

    for value in products[column].dropna().astype(str):

        words = re.findall(
            r"[a-zA-Z0-9]+",
            value.lower()
        )

        word_frequency.update(words)


unique_words = list(word_frequency.keys())

print("Unique words:", len(unique_words))


# ============================================================
# 4. SPELLING CORRECTION
# ============================================================

def correct_word(
    word,
    unique_words,
    threshold=70
):

    result = process.extractOne(
        word.lower(),
        unique_words,
        scorer=fuzz.ratio
    )

    if result is None:
        return word

    matched_word, score, index = result

    if score >= threshold:
        return matched_word

    return word


def correct_query(
    query,
    unique_words,
    threshold=70
):

    words = query.split()

    corrected_words = []

    for word in words:

        corrected = correct_word(
            word,
            unique_words,
            threshold
        )

        corrected_words.append(corrected)

    return " ".join(corrected_words)


# ============================================================
# TEST SPELLING CORRECTION
# ============================================================

test_query = "stainles steel pype"

corrected = correct_query(
    test_query,
    unique_words,
    SPELLING_THRESHOLD
)

print("\nSpelling correction test:")
print("Original :", test_query)
print("Corrected:", corrected)


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# 6. CREATE PRODUCT EMBEDDINGS
# ============================================================

print("\nCreating product embeddings...")

embedding_start = time.perf_counter()

product_embeddings = model.encode(
    products["search_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embedding_time = (
    time.perf_counter() - embedding_start
)

product_embeddings = np.asarray(
    product_embeddings,
    dtype="float32"
)

print(
    "Embedding shape:",
    product_embeddings.shape
)

print(
    "Embedding time:",
    round(embedding_time, 3),
    "seconds"
)


# ============================================================
# 7. BUILD FAISS INDEX
# ============================================================

dimension = product_embeddings.shape[1]

print("\nBuilding FAISS index...")

index_start = time.perf_counter()

# Normalized embeddings + Inner Product
# = Cosine Similarity

index = faiss.IndexFlatIP(dimension)

index.add(product_embeddings)

index_time = (
    time.perf_counter() - index_start
)

print(
    "FAISS vectors:",
    index.ntotal
)

print(
    "Vector dimension:",
    dimension
)

print(
    "Index build time:",
    round(index_time, 3),
    "seconds"
)


# ============================================================
# 8. SEARCH + METRICS
# ============================================================

hit_at_1 = 0

recall_at_5_values = []
recall_at_10_values = []

reciprocal_ranks = []

search_times = []

results = []


for _, row in tests.iterrows():

    original_query = str(row["query"])

    # --------------------------------------------------------
    # SPELLING CORRECTION
    # --------------------------------------------------------

    query = correct_query(
        original_query,
        unique_words,
        SPELLING_THRESHOLD
    )

    expected_product = str(
        row["expected_product_name"]
    ).strip().lower()


    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # --------------------------------------------------------
    # FAISS SEARCH
    # --------------------------------------------------------

    search_start = time.perf_counter()

    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    search_time = (
        time.perf_counter() - search_start
    )

    search_times.append(search_time)


    # --------------------------------------------------------
    # RETRIEVED PRODUCTS
    # --------------------------------------------------------

    retrieved_indices = indices[0]

    retrieved_products = [
        str(
            products.iloc[i]["product_name"]
        ).strip().lower()

        for i in retrieved_indices

        if i != -1
    ]


    # --------------------------------------------------------
    # FIND EXPECTED PRODUCT RANK
    # --------------------------------------------------------

    rank = None

    for position, product_name in enumerate(
        retrieved_products,
        start=1
    ):

        if product_name == expected_product:

            rank = position
            break


    # --------------------------------------------------------
    # HIT@1
    # --------------------------------------------------------

    hit = 1 if rank == 1 else 0

    hit_at_1 += hit


    # --------------------------------------------------------
    # RECALL@5
    # --------------------------------------------------------

    recall_5 = (
        1
        if rank is not None and rank <= 5
        else 0
    )

    recall_at_5_values.append(
        recall_5
    )


    # --------------------------------------------------------
    # RECALL@10
    # --------------------------------------------------------

    recall_10 = (
        1
        if rank is not None and rank <= 10
        else 0
    )

    recall_at_10_values.append(
        recall_10
    )


    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    if rank is not None:
        reciprocal_rank = 1 / rank
    else:
        reciprocal_rank = 0

    reciprocal_ranks.append(
        reciprocal_rank
    )


    # --------------------------------------------------------
    # SAVE QUERY RESULT
    # --------------------------------------------------------

    results.append({

        "query_id":
            row["query_id"],

        "original_query":
            original_query,

        "corrected_query":
            query,

        "expected_product":
            expected_product,

        "rank":
            rank,

        "hit@1":
            hit,

        "recall@5":
            recall_5,

        "recall@10":
            recall_10,

        "reciprocal_rank":
            reciprocal_rank,

        "top_result":
            (
                retrieved_products[0]
                if retrieved_products
                else None
            ),

        "top_score":
            float(scores[0][0])
    })


# ============================================================
# 9. FINAL METRICS
# ============================================================

num_queries = len(tests)

hit_at_1_score = (
    hit_at_1 / num_queries
)

recall_at_5_score = np.mean(
    recall_at_5_values
)

recall_at_10_score = np.mean(
    recall_at_10_values
)

mrr_score = np.mean(
    reciprocal_ranks
)

average_search_time_ms = (
    np.mean(search_times) * 1000
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FAISS BENCHMARK RESULTS")
print("=" * 60)

print(
    f"Embedding model       : "
    f"{EMBEDDING_MODEL}"
)

print(
    f"Vector dimension      : "
    f"{dimension}"
)

print(
    f"Number of products    : "
    f"{len(products)}"
)

print(
    f"Number of queries     : "
    f"{num_queries}"
)

print(
    f"Embedding time        : "
    f"{embedding_time:.3f} sec"
)

print(
    f"FAISS index time      : "
    f"{index_time:.3f} sec"
)

print(
    f"Average search time   : "
    f"{average_search_time_ms:.3f} ms"
)

print()

print(
    f"Hit@1                 : "
    f"{hit_at_1_score:.4f}"
)

print(
    f"Recall@5              : "
    f"{recall_at_5_score:.4f}"
)

print(
    f"Recall@10             : "
    f"{recall_at_10_score:.4f}"
)

print(
    f"MRR                   : "
    f"{mrr_score:.4f}"
)

print("=" * 60)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "faiss_benchmark_results.csv",
    index=False
)

print(
    "\nSaved: faiss_benchmark_results.csv"
)


# ============================================================
# 12. SHOW SAMPLE RESULTS
# ============================================================

print("\nSample results:")

print(
    results_df[
        [
            "query_id",
            "original_query",
            "corrected_query",
            "expected_product",
            "rank",
            "hit@1",
            "recall@5",
            "recall@10",
            "reciprocal_rank"
        ]
    ]
    .head(20)
    .to_string(index=False)
)