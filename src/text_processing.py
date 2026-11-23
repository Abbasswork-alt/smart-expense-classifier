import re
import unicodedata


def clean_text(text):
    text = unicodedata.normalize("NFC", text.lower())
    # Drop optional Arabic vowel marks and elongation without splitting words.
    text = re.sub(r"[\u0640\u064b-\u0652\u0670]", "", text)
    letters = "".join(char if char.isalpha() else " " for char in text)
    return " ".join(letters.split())
