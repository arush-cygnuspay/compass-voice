# app/nlu/query_normalization/query_normalizer_pipeline.py
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.choice_query_normalizer import ChoiceQueryNormalizer
from app.nlu.query_normalization.item_query_normalizer import ItemQueryNormalizer
from app.nlu.query_normalization.menu_info_query_normalizer import MenuInfoQueryNormalizer
from app.state_machine.conversation_state import ConversationState


class QueryNormalizationPipeline:
    """
    Phase-aware normalization.
    """

    def __init__(self):
        self.item_normalizer = ItemQueryNormalizer()
        self.choice_normalizer = ChoiceQueryNormalizer()
        self.menu_info_normalizer = MenuInfoQueryNormalizer()

    # app/nlu/query_normalization/query_normalizer_pipeline.py

    def normalize(self, text: str, intent: Intent, state: ConversationState) -> str:

        # 🚫 Cart overlays must NEVER be menu-normalized
        if intent in {
            Intent.SHOW_CART,
            Intent.SHOW_TOTAL,
            Intent.CLEAR_CART,
        }:
            return text  # raw, untouched

        # 🔹 Price inquiry (item-scoped)
        if intent == Intent.ASK_PRICE:
            return self.item_normalizer.normalize(text, intent)

        # 🔹 Menu info queries
        if intent == Intent.ASK_MENU_INFO:
            return self.menu_info_normalizer.normalize(text, intent)

        # 🔹 Item resolution
        if state == ConversationState.IDLE:
            return self.item_normalizer.normalize(text, intent)

        # 🔹 Choice resolution
        if state in {
            ConversationState.WAITING_FOR_SIDE,
            ConversationState.WAITING_FOR_MODIFIER,
            ConversationState.WAITING_FOR_SIZE,
        }:
            return self.choice_normalizer.normalize(text, intent)

        return text

