import json
from models import Word


def normalize_text(text):
    return text.lower()


def extract_words_from_freq_file(
    n=2000, data_folder="/workspaces/data", filename="freq_vie_1M_2018-freq.txt"
):
    """Extracts words from a frequency file and returns a list of Word objects."""
    words = []
    with open(f"{data_folder}/{filename}", "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:  # Ensure the line has exactly 3 parts
                id = int(parts[0])
                text = parts[1]
                freq = int(parts[2])
                words.append(Word(id, normalize_text(text), freq))
            if len(words) >= n:  # Stop after collecting 'n' words
                break
    return words


def export_words_to_js_file(words, output_file="data/VietnameseWords.js"):
    """Exports a list of Word objects to a JavaScript file in the specified format."""
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("const VietnameseWords = [\n")
        for word in words:
            file.write(
                f'  {{ word: "{word.text}", partOfSpeech: "", frequency: {word.freq} }},\n'
            )
        file.write("];\n")
    print(f"File '{output_file}' created successfully with {len(words)} entries.")


def export_words_to_jsonl_file(words, output_file="data/VietnameseWords.jsonl"):
    """Exports a list of Word objects to a JSONL file."""
    with open(output_file, "w", encoding="utf-8") as file:
        for word in words:
            word_data = {"word": word.text, "partOfSpeech": "", "frequency": word.freq}
            file.write(json.dumps(word_data, ensure_ascii=False) + "\n")
    print(f"File '{output_file}' created successfully with {len(words)} entries.")


if __name__ == "__main__":
    # Extract words from the frequency file
    words = extract_words_from_freq_file(n=2000)

    # Export words to JavaScript and JSON files
    export_words_to_js_file(words)
    export_words_to_jsonl_file(words)
