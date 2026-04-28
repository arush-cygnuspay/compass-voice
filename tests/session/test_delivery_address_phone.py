import unittest

from app.state_machine.models.delivery_address import DeliveryAddress


class DeliveryAddressPhoneTests(unittest.TestCase):
    def test_has_phone_number_false_for_none(self):
        address = DeliveryAddress(customer_phone_number=None)
        self.assertFalse(address.has_phone_number)

    def test_has_phone_number_false_for_empty_string(self):
        address = DeliveryAddress(customer_phone_number="")
        self.assertFalse(address.has_phone_number)

    def test_has_phone_number_false_for_whitespace_only(self):
        address = DeliveryAddress(customer_phone_number="   \t\n ")
        self.assertFalse(address.has_phone_number)

    def test_has_phone_number_true_for_valid_value(self):
        address = DeliveryAddress(customer_phone_number="+15555550123")
        self.assertTrue(address.has_phone_number)

    def test_normalized_phone_number_returns_digits_only(self):
        address = DeliveryAddress(customer_phone_number="+1 (415) 555-1234")
        self.assertEqual(address.normalized_phone_number(), "14155551234")

    def test_normalized_phone_number_returns_none_for_letters_only(self):
        address = DeliveryAddress(customer_phone_number="not-a-number-abc")
        self.assertIsNone(address.normalized_phone_number())

    def test_normalized_phone_number_returns_none_for_missing_value(self):
        address = DeliveryAddress(customer_phone_number=None)
        self.assertIsNone(address.normalized_phone_number())

    def test_normalized_phone_number_returns_none_for_whitespace(self):
        address = DeliveryAddress(customer_phone_number="   ")
        self.assertIsNone(address.normalized_phone_number())


if __name__ == "__main__":
    unittest.main()
