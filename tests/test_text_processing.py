import unittest

from src.text_processing import clean_text


class ExpenseCleaningTests(unittest.TestCase):
    def test_expense_descriptions(self):
        cases = [
            ("[debit] UBER*123 Ride to Airport!", "debit uber ride to airport"),
            ("دفعت فاتورة الإنترنت", "دفعت فاتورة الإنترنت"),
            ("دَفَعْتُ فاتـورة WiFi، ١٢٣!", "دفعت فاتورة wifi"),
            ("تاكسي/باص", "تاكسي باص"),
            ("١٢٣، 45!!", ""),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(clean_text(raw), expected)
