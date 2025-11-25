

class CharTokenizer:
    """
    Very simple character-level tokenizer
    - Builds vocabulary from a given text string
    - Provides encode(text) -> list[int]
    - Provides decode(list[int]) -> text
    """

    def __init__(self, text: str):
        # Get all unique characters in the text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # maps char -> int and int -> char
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def decode(self, s: str):
        """Convert string to list of integers."""
        return [self.stoi[ch] for ch in s]
    
    def encode(self, ids):
         """Convert list of integers back to string."""
         return [self.itos[i] for i in ids]