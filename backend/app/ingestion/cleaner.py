import re


class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        # Normalize unicode spaces
        text = text.replace('\xa0', ' ')
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        # Clean up OCR artifacts (simple version)
        text = re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)
        return text.strip()

    @staticmethod
    def clean_batch(texts: list[str]) -> list[str]:
        return [TextCleaner.clean(text) for text in texts]
