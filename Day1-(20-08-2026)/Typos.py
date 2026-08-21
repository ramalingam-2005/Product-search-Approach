#Edge cases
"""
Abbreviations – SS Pipe vs Stainless Steel Pipe

Partial-word matches – pipe matching pipeline
Synonyms – Tube vs Pipe
"""
#Rapid Fuzz + Normalization
import pandas as pd
import re
from rapidfuzz import fuzz

df = pd.read_csv(
    "E:/PROJECTS/Search Engine/app/b2b_product_search_lesson1.csv"
)


def normalize_text(text):
    text = str(text).lower()

    # Replace hyphens and special characters with spaces
    text = re.sub(r"[-_/]", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def searchTypos(query, threshold=70):

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
results = searchTypos("steel stianleesss pipes")

print(results.to_string())