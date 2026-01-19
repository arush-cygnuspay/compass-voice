# app/nlu/query_normalizers/query_normalizer.py
from app.nlu.intent_resolution.intent import Intent

class QueryNormalizer:
    def normalize(self, text: str, intent: Intent) -> str:
        raise NotImplementedError
