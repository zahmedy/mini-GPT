

from typing import List, Sequence


class CharTokenizer:
    """
    Very simple character-level tokenizer
    - Builds vocabulary from a given text string
    - Provides encode(text) -> list[int]
    - Provides decode(list[int]) -> text
    """

    def __init__(self, text: str) -> None:
        print("\n" + "="*80)
        print("TOKENIZER INITIALIZATION")
        print("="*80)

        # Get all unique characters in the text
        chars = sorted(list(set(text)))

        if "<unk>" not in chars:
            chars = ["<unk>"] + chars

        self.vocab_size = len(chars)

        print(f"Total unique characters in corpus: {len(chars)}")
        print(f"Vocabulary: {chars[:20]}{'...' if len(chars) > 20 else ''}")
        print(f"Vocab size: {self.vocab_size}")

        # maps char -> int and int -> char
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.unk_id = self.stoi["<unk>"]

    def encode(self, s: str) -> List[int]:
        """Convert string to list of integers."""
        ids = [self.stoi.get(ch, self.unk_id) for ch in s]
        if len(s) <= 50:
            print(f"\nENCODE: '{s}' -> {ids}")
        return ids
    
    def decode(self, ids: Sequence[int]) -> str:
        """Convert list of integers back to string."""
        result = "".join(self.itos[i] for i in ids)
        if len(ids) <= 50:
            print(f"DECODE: {list(ids)} -> '{result}'")
        return result
