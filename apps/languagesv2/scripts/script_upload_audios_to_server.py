import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define directories and server details
SRC_DIR = "/workspaces/data/audios"
HOST_DIR = "/root/data"
UTS_HOST = os.getenv("UTS_HOST")
UTS_USERNAME = os.getenv("UTS_USERNAME")
UTS_PASSWORD = os.getenv("UTS_PASSWORD")

# Check if required environment variables are set
if not UTS_HOST or not UTS_USERNAME or not UTS_PASSWORD:
    print(
        "Error: Missing UTS_HOST, UTS_USERNAME, or UTS_PASSWORD environment variable."
    )
    exit(1)


# Sync command with sshpass for password-based authentication
def sync_audio_to_server():
    command = f'sshpass -p "{UTS_PASSWORD}" rsync -avz {SRC_DIR} {UTS_USERNAME}@{UTS_HOST}:{HOST_DIR}'
    print(f"Executing command: {command}")
    os.system(command)  # Executes rsync with specified directories


# Run the sync function
sync_audio_to_server()
