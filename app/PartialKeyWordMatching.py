"""
#Edge cases
Singular/plural – Pipe vs Pipes
Abbreviations – SS Pipe vs Stainless Steel Pipe
Typos – Stainles Steel Pipe vs Stainless Steel Pipe

Partial-word matches – pipe matching pipeline
Synonyms – Tube vs Pipe
"""

import pandas as pd
import re

df = pd.read_csv("app/b2b_product_search_lesson1.csv")


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

query = "stainless steel pipe"

df["score"] = df["product_name"].apply(
    lambda x: calculate_score(query, x)
)
results = df[df["score"] > 0]

results = results.sort_values(
    by="score",
    ascending=False
)

# print(results[[
#     "product_name",
#     "category",
#     "price",
#     "supplier",
#     "score"
# ]].head(10))