# app/nlu/query_normalization/query_normalizer_pipeline.py
from app.nlu.query_normalization.item_query_normalizer import ItemQueryNormalizer
from app.nlu.query_normalization.choice_query_normalizer import ChoiceQueryNormalizer
from app.state_machine.conversation_state import ConversationState

class QueryNormalizationPipeline:
    """
    Phase-aware normalization.
    Decides which normalizer to apply based on conversation state.
    """

    def __init__(self):
        self.item_normalizer = ItemQueryNormalizer()
        self.choice_normalizer = ChoiceQueryNormalizer()

    def normalize(self, text: str, intent, state: ConversationState) -> str:
        if state == ConversationState.IDLE:
            return self.item_normalizer.normalize(text, intent)

        if state in {
            ConversationState.WAITING_FOR_SIDE,
            ConversationState.WAITING_FOR_MODIFIER,
            ConversationState.WAITING_FOR_SIZE,
        }:
            return self.choice_normalizer.normalize(text, intent)

        return text
