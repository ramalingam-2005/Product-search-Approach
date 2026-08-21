#Keyword Matching
"""
#Edge cases
Singular/plural – Pipe vs Pipes
Abbreviations – SS Pipe vs Stainless Steel Pipe
Typos – Stainles Steel Pipe vs Stainless Steel Pipe
Word-order differences – Steel Stainless Pipe vs Stainless Steel Pipe
Extra/missing words – Stainless Pipe vs Stainless Steel Pipe

Partial-word matches – pipe matching pipeline
Synonyms – Tube vs Pipe
"""
#Keyword matching+Normalize the text

import pandas as pd
import pandas as pd
import re

def normalize_text(text):
    text = str(text).lower()

    # Replace hyphens and special characters with spaces
    text = re.sub(r"[-_/]", " ", text)

    # Remove other punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
def search(query):
  query=normalize_text(query)
  results = df[
        df["product_name"]
        .str.lower()
        .str.contains(query, na=False)
    ]
  return results


df=pd.read_csv("E:/PROJECTS/Search Engine/app/b2b_product_search_lesson1.csv")
# results = search("Stainless steel     Pipe")

# print(results[[
#     "product_id",
#     "product_name",
#     "category",
#     "price",
#     "MOQ",
#     "supplier",
#     "location"
# ]])

