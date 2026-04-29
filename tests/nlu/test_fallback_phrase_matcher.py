# tests/nlu/test_fallback_phrase_matcher.py
"""Unit tests for FallbackPhraseMatcher."""
import unittest

from app.nlu.fallback_phrase_matcher import FallbackPhraseMatcher


class TestFallbackPhraseMatcherAgentRequest(unittest.TestCase):
    def setUp(self):
        self.matcher = FallbackPhraseMatcher()

    # --- positive cases -------------------------------------------------------

    def test_bare_word_agent(self):
        self.assertTrue(self.matcher.match_agent_request("agent"))

    def test_bare_word_human(self):
        self.assertTrue(self.matcher.match_agent_request("human"))

    def test_bare_word_operator(self):
        self.assertTrue(self.matcher.match_agent_request("operator"))

    def test_phrase_connect_me_to_a_person(self):
        self.assertTrue(self.matcher.match_agent_request("connect me to a person"))

    def test_phrase_let_me_talk_to_someone(self):
        self.assertTrue(self.matcher.match_agent_request("let me talk to someone"))

    def test_phrase_i_need_an_agent(self):
        self.assertTrue(self.matcher.match_agent_request("I need an agent"))

    def test_stuck_phrase(self):
        self.assertTrue(self.matcher.match_agent_request("I am stuck"))

    def test_phrase_having_trouble(self):
        self.assertTrue(self.matcher.match_agent_request("having trouble"))

    def test_phrase_not_working(self):
        self.assertTrue(self.matcher.match_agent_request("this is not working"))

    def test_case_insensitive(self):
        self.assertTrue(self.matcher.match_agent_request("AGENT"))
        self.assertTrue(self.matcher.match_agent_request("Human"))

    def test_embedded_word_agent(self):
        # "agent" appears as substring inside a longer utterance
        self.assertTrue(self.matcher.match_agent_request("can I speak to an agent please"))

    # --- negative cases -------------------------------------------------------

    def test_empty_string(self):
        self.assertFalse(self.matcher.match_agent_request(""))

    def test_whitespace_only(self):
        self.assertFalse(self.matcher.match_agent_request("   "))

    def test_unrelated_utterance(self):
        self.assertFalse(self.matcher.match_agent_request("I would like a burger"))

    def test_order_related_utterance(self):
        self.assertFalse(self.matcher.match_agent_request("yes please"))

    def test_false_positive_instead_of_waiting(self):
        # "instead of waiting" must NOT trigger agent request
        self.assertFalse(self.matcher.match_agent_request("instead of waiting"))


class TestFallbackPhraseMatcherQuantityCorrection(unittest.TestCase):
    def setUp(self):
        self.matcher = FallbackPhraseMatcher()

    # --- positive cases -------------------------------------------------------

    def test_instead_of_pattern(self):
        self.assertTrue(self.matcher.match_quantity_correction("2 instead of 1"))

    def test_instead_of_in_longer_utterance(self):
        self.assertTrue(
            self.matcher.match_quantity_correction("make it 3 instead of 2")
        )

    def test_make_it_prefix(self):
        self.assertTrue(self.matcher.match_quantity_correction("make it 2"))

    def test_change_it_to_prefix(self):
        self.assertTrue(self.matcher.match_quantity_correction("change it to 3"))

    def test_set_it_to_prefix(self):
        self.assertTrue(self.matcher.match_quantity_correction("set it to 5"))

    def test_case_insensitive_prefix(self):
        self.assertTrue(self.matcher.match_quantity_correction("Make It 2"))

    # --- negative cases -------------------------------------------------------

    def test_empty_string(self):
        self.assertFalse(self.matcher.match_quantity_correction(""))

    def test_unrelated_utterance(self):
        self.assertFalse(self.matcher.match_quantity_correction("I want a coke"))

    def test_false_positive_instead_of_waiting(self):
        # Triggers because "instead of" is a substring — this is expected
        # behaviour for the phrase fallback path (low-confidence safety net).
        # NLU-first path would gate this before reaching phrase matching.
        self.assertTrue(self.matcher.match_quantity_correction("instead of waiting"))

    def test_no_prefix_no_instead(self):
        self.assertFalse(self.matcher.match_quantity_correction("give me one burger"))


if __name__ == "__main__":
    unittest.main()
