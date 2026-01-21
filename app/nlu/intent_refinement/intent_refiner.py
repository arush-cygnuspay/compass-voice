# app/nlu/intent_refinement/intent_refiner.py

from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.state_machine.conversation_state import ConversationState


class IntentRefiner:
    """
    Refines linguistically ambiguous intents using menu knowledge.

    This is the ONLY layer where menu semantics are allowed
    to influence intent resolution.

    Responsibilities:
    -----------------
    - Resolve ADD_ITEM vs ASK_MENU_INFO ambiguity
    - Use MenuRepository as the single source of truth
    - Remain pure, deterministic, and side-effect free

    Non-responsibilities:
    ---------------------
    - State mutation
    - Cart inspection
    - Conversational decisions
    - Natural language parsing
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def refine(
        self,
        intent: Intent,
        normalized_text: str,
        state: ConversationState,
    ) -> Intent:
        """
        Refine intent using menu dominance rules.

        Parameters:
        -----------
        intent:
            The intent produced by pure NLU (regex-based).
        normalized_text:
            Text after query normalization.
        state:
            Current conversation state.

        Returns:
        --------
        Intent:
            Final intent to be routed by the StateRouter.
        """

        # -------------------------------------------------
        # Scope guards
        # -------------------------------------------------
        if state != ConversationState.IDLE:
            return intent

        if intent not in {
            Intent.ADD_ITEM,
            Intent.ASK_MENU_INFO,
            Intent.UNKNOWN,
        }:
            return intent

        # -------------------------------------------------
        # Resolve menu dominance
        # -------------------------------------------------
        result: MenuQueryResult = self.menu_repo.resolve_menu_query(
            normalized_text
        )

        # -------------------------------------------------
        # CATEGORY dominance → informational
        # -------------------------------------------------
        if result.type in {
            MenuQueryType.CATEGORY,
            MenuQueryType.CATEGORY_AMBIGUOUS,
        }:
            return Intent.ASK_MENU_INFO

        # -------------------------------------------------
        # CATEGORY with exactly one item → treat as item
        # Example: category "Soup of the Day"
        # -------------------------------------------------
        if result.type == MenuQueryType.CATEGORY_SINGLE_ITEM:
            return Intent.ADD_ITEM

        # -------------------------------------------------
        # ITEM ambiguity → informational
        # -------------------------------------------------
        if result.type == MenuQueryType.ITEM_AMBIGUOUS:
            return Intent.ASK_MENU_INFO

        # -------------------------------------------------
        # Clear ITEM dominance
        # -------------------------------------------------
        if result.type == MenuQueryType.ITEM:
            # Respect explicit ASK phrasing
            if intent == Intent.ASK_MENU_INFO:
                return Intent.ASK_MENU_INFO

            return Intent.ADD_ITEM

        # -------------------------------------------------
        # NOT FOUND → leave intent unchanged
        # -------------------------------------------------
        return intent
