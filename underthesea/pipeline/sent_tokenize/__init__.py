"""Vietnamese sentence tokenization using a Punkt-style algorithm."""

import json
import string
from os.path import dirname, join

sentence_tokenizer = None


class PunktSentenceTokenizer:
    """Custom Vietnamese sentence tokenizer (Punkt-style).

    This is a simplified implementation that handles common Vietnamese sentence
    boundaries while respecting abbreviations.
    """

    SENT_END_CHARS = frozenset('.?!')

    def __init__(self, abbrev_types=None):
        self.abbrev_types = abbrev_types or set()

    def sentences_from_text(self, text):
        """Tokenize text into sentences.

        Args:
            text: Input text string.

        Returns:
            List of sentence strings.
        """
        if not text or not text.strip():
            return []

        sentences = []
        start = 0
        i = 0
        n = len(text)

        while i < n:
            if text[i] in self.SENT_END_CHARS:
                # Detect ellipsis (3+ dots)
                if text[i] == '.':
                    dot_end = i
                    while dot_end < n and text[dot_end] == '.':
                        dot_end += 1
                    if dot_end - i >= 3:
                        # Ellipsis: check what follows
                        after = dot_end
                        while after < n and text[after] in ' \t':
                            after += 1
                        if after >= n or text[after].isupper():
                            # Ellipsis at end of sentence or before uppercase
                            sentence = text[start:dot_end].strip()
                            if sentence:
                                sentences.append(sentence)
                            while dot_end < n and text[dot_end] in ' \t\n\r':
                                dot_end += 1
                            start = dot_end
                        # Skip past all dots (not a boundary or already handled)
                        i = dot_end
                        continue

                # Check if this is a real sentence boundary
                if self._is_sentence_boundary(text, i):
                    # Find the end of the sentence (include punctuation)
                    end = i + 1

                    # Skip any trailing sentence-ending punctuation (e.g., "?!")
                    while end < n and text[end] in self.SENT_END_CHARS:
                        end += 1

                    # Include trailing whitespace in boundary detection but not in sentence
                    sentence = text[start:end].strip()
                    if sentence:
                        sentences.append(sentence)

                    # Move start past whitespace
                    while end < n and text[end] in ' \t\n\r':
                        end += 1

                    start = end
                    i = end
                    continue
            i += 1

        # Add remaining text as final sentence
        final = text[start:].strip()
        if final:
            sentences.append(final)

        return sentences

    def _is_sentence_boundary(self, text, pos):
        """Check if position marks a real sentence boundary.

        Args:
            text: The full text.
            pos: Position of the punctuation character.

        Returns:
            True if this is a sentence boundary, False otherwise.
        """
        if text[pos] == '.':
            # Numeric period: digit.digit (e.g., 1.500.000, 15.12.2025)
            if pos > 0 and text[pos - 1].isdigit():
                next_char_pos = pos + 1
                if next_char_pos < len(text) and text[next_char_pos].isdigit():
                    return False

            # Check if period follows an abbreviation
            word = self._get_preceding_word(text, pos)
            if word and word.lower() in self.abbrev_types:
                return False

        # Check if there's a following character that indicates sentence continuation
        next_pos = pos + 1
        # Skip any additional punctuation
        while next_pos < len(text) and text[next_pos] in self.SENT_END_CHARS:
            next_pos += 1

        # Skip whitespace
        while next_pos < len(text) and text[next_pos] in ' \t':
            next_pos += 1

        # If we've reached end of text, it's a boundary
        if next_pos >= len(text):
            return True

        # If next char is newline, it's likely a boundary
        if text[next_pos] in '\n\r':
            return True

        # Default: treat as boundary
        return True

    def _get_preceding_word(self, text, pos):
        """Get the word preceding the given position.

        Args:
            text: The full text.
            pos: Position to look before.

        Returns:
            The preceding word or None.
        """
        end = pos
        start = pos - 1

        # Move back through alphanumeric characters and periods (for abbreviations like "e.g")
        while start >= 0 and (text[start].isalnum() or text[start] == '.'):
            start -= 1

        word = text[start + 1:end]
        return word if word else None


def _load_model():
    """Load the sentence tokenizer model."""
    global sentence_tokenizer
    if sentence_tokenizer is not None:
        return

    params_path = join(dirname(__file__), 'punkt_params.json')
    with open(params_path, encoding='utf-8') as f:
        data = json.load(f)

    abbrev_types = set(data.get('abbrev_types', []))

    # Add custom abbreviations for Vietnamese and common English abbreviations
    custom_abbrevs = [
        'g.m.t', 'e.g', 'dr', 'vs', '000', 'mr', 'mrs', 'prof', 'inc',
        'tp', 'ts', 'ths', 'th', 'k.l', 'a.w.a.k.e', 't', 'a.i', '</i', 'g.w',
        'ass', 'u.n.c.l.e', 't.e.s.t', 'd.c', 've…', 'f.t', 'b.b', 'z.e', 's.g', 'm.p',
        'g.u.y', 'l.c', 'g.i', 'j.f', 'r.r', 'v.i', 'm.h', 'a.s', 'bs', 'c.k', 'aug',
        't.d.q', 'b…', 'ph', 'j.k', 'e.l', 'o.t', 's.a'
    ]
    abbrev_types.update(custom_abbrevs)

    # Add single uppercase and lowercase letters as abbreviations
    abbrev_types.update(string.ascii_uppercase)
    abbrev_types.update(string.ascii_lowercase)

    sentence_tokenizer = PunktSentenceTokenizer(abbrev_types)


def sent_tokenize(text):
    """Tokenize Vietnamese text into sentences.

    Args:
        text: Input Vietnamese text string.

    Returns:
        List of sentence strings.

    Examples:
        >>> sent_tokenize("Xin chào. Tôi là Claude.")
        ['Xin chào.', 'Tôi là Claude.']
    """
    _load_model()
    return sentence_tokenizer.sentences_from_text(text)
