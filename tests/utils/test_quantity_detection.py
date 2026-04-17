import unittest

from app.utils.quantity_detection import detect_quantity, normalize_quantity


class QuantityDetectionTests(unittest.TestCase):
    def test_normalize_quantity_handles_units_and_dozens(self):
        self.assertEqual(normalize_quantity("2 pcs"), 2)
        self.assertEqual(normalize_quantity("half dozen"), 6)
        self.assertEqual(normalize_quantity("one dozen"), 12)
        self.assertIsNone(normalize_quantity("a burger"))

    def test_detect_quantity_supports_incremental_another(self):
        self.assertEqual(
            detect_quantity("another"),
            {"type": "incremental", "value": 1},
        )


if __name__ == "__main__":
    unittest.main()
