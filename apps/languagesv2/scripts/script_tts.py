import json
import os
import requests
import uuid
from models import Word
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory setup
AUDIO_DIR = "/workspaces/data/audios"
AUDIOS_FILE_PATH = "data/Audios.jsonl"
AUDIOS_TSX_FILE_PATH = "data/Audios.tsx"
ZALO_API_KEY = os.getenv("ZALO_API_KEY")


def load_existing_metadata():
    """Load existing metadata into a set for fast lookup."""
    if not os.path.exists(AUDIOS_FILE_PATH):
        return set()

    existing_metadata = set()
    with open(AUDIOS_FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                existing_metadata.add((data["word"], data["speaker_id"]))
            except json.JSONDecodeError:
                continue  # Skip any malformed JSON lines
    return existing_metadata


# Initialize metadata set
existing_metadata = load_existing_metadata()


def generate_key(word, speaker_id):
    """Generate a unique UUIDv3 key for the combination of word and speaker_id."""
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{word}_{speaker_id}"))


def is_downloaded(key):
    """Check if an audio file with a specific key already exists in the audio folder."""
    return os.path.exists(os.path.join(AUDIO_DIR, f"{key}.wav"))


def save_audio_metadata(word, speaker_id, audio_id):
    """Save audio metadata to a JSONL file if it doesn't already exist."""
    if (word, speaker_id) not in existing_metadata:
        with open(AUDIOS_FILE_PATH, "a", encoding="utf-8") as f:
            json_line = json.dumps(
                {"word": word, "speaker_id": speaker_id, "audio_id": audio_id},
                ensure_ascii=False,
            )
            f.write(json_line + "\n")
        existing_metadata.add((word, speaker_id))  # Update the set
        print(f"Metadata saved for '{word}' with speaker_id {speaker_id}")
    else:
        print(f"Metadata already exists for '{word}' with speaker_id {speaker_id}")


def tts(text, speaker_ids=[1], encode_type=0):
    """
    Synthesize speech for the given text if not already downloaded.

    Args:
        text (str): Text to be synthesized into speech.
        speaker_ids (list[int]): List of IDs representing speaker voice types.
            1 - South women
            2 - Northern women
            3 - South men
            4 - Northern men
        encode_type (int): Encoding type for the speech synthesis (default is 0).

    Returns:
        dict: Dictionary mapping speaker IDs to paths of the synthesized audio files.
    """
    file_paths = {}

    for speaker_id in speaker_ids:
        # Generate key for file naming
        key = generate_key(text, speaker_id)

        # Check if the audio is already downloaded
        file_path = os.path.join(AUDIO_DIR, f"{key}.wav")
        if is_downloaded(key):
            # Verify if file size is greater than zero
            if os.path.getsize(file_path) > 0:
                print(
                    f"File for '{text}' with speaker_id {speaker_id} already exists at: {file_path}"
                )
                file_paths[speaker_id] = file_path
                continue
            else:
                print(
                    f"File for '{text}' with speaker_id {speaker_id} is empty, retrying download..."
                )

        # API setup
        url = "https://api.zalo.ai/v1/tts/synthesize"
        headers = {
            "apikey": ZALO_API_KEY,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"input": text, "speaker_id": speaker_id, "encode_type": encode_type}

        # Retry mechanism
        for attempt in range(5):  # Try up to 5 times
            response = requests.post(url, headers=headers, data=data)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("error_code") == 0:
                    audio_url = response_data["data"]["url"]
                    audio_response = requests.get(audio_url)

                    # Save audio file
                    os.makedirs(AUDIO_DIR, exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(audio_response.content)

                    # Verify if file is downloaded correctly
                    if os.path.getsize(file_path) > 0:
                        print(f"Audio saved at: {file_path}")
                        file_paths[speaker_id] = file_path
                        save_audio_metadata(text, speaker_id, key)
                        break  # Exit retry loop if download is successful
                    else:
                        print(
                            f"Download attempt {attempt + 1} failed for '{text}' with speaker_id {speaker_id}. Retrying..."
                        )
                        os.remove(file_path)  # Remove the empty file
                else:
                    print("Error:", response_data.get("error_message"))
                    break  # No retry for API response errors
            else:
                print(
                    f"Failed to call API (status code {response.status_code}) on attempt {attempt + 1}. Retrying..."
                )

    return file_paths


def read_words(file_path):
    words = []
    with open(file_path, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                data = json.loads(line)
                word = Word(id=idx, text=data["word"], freq=data["frequency"])
                words.append(word)
            except Exception as e:
                print("Exception:", e)
                print("Line:", line)
    return words


def convert_jsonl_to_tsx():
    """
    Convert the Audios.jsonl file to a TypeScript file (Audios.tsx).
    """
    audios = []

    # Read from the JSONL file
    with open(AUDIOS_FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                audios.append(data)
            except json.JSONDecodeError:
                continue  # Skip any malformed JSON lines

    # Generate TypeScript content
    with open(AUDIOS_TSX_FILE_PATH, "w", encoding="utf-8") as tsx_file:
        tsx_file.write("const Audios = [\n")

        for audio in audios:
            tsx_file.write("  " + json.dumps(audio, ensure_ascii=False) + ",\n")

        tsx_file.write("];\n\n")
        tsx_file.write("export default Audios;\n")

    print(f"Audios.tsx file created at: {AUDIOS_TSX_FILE_PATH}")


if __name__ == "__main__":
    # Initialize audio directory and metadata file
    os.makedirs(AUDIO_DIR, exist_ok=True)

    file_path = "data/VietnameseWords.jsonl"
    words = read_words(file_path)

    # Get the first 10 words
    # active_words = words[:10]

    active_words = words

    # Download audio for each word in the first 10 if not already downloaded
    for word in active_words:
        print(f"Processing word: {word.text}")
        tts(word.text, speaker_ids=[1, 2])

    convert_jsonl_to_tsx()
