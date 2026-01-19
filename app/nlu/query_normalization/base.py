# app/nlu/query_normalizers/base.py
import re, string

def basic_cleanup(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("’", "'")  # unicode apostrophe
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
