# Python code for youtube.playlistItems.list
# To set up YouTube API: https://developers.google.com/youtube/v3/getting-started
# To use channel PlaylistItems: https://developers.google.com/youtube/v3/docs/playlistItems/list

import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from os.path import dirname, join
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

PROJECT_FOLDER = dirname(dirname(dirname(__file__)))
DATASETS_FOLDER = join(PROJECT_FOLDER, "datasets")
COL_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL")

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]


def playlist_youtube_api(playlist):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "client_secret.json"

    # Get credentials and create an API client
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, scopes)
    credentials = flow.run_console()
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials)

    request = youtube.playlistItems().list(
        part="contentDetails",
        maxResults=50,
        playlistId=playlist
    )
    response = request.execute()
    return response


class COLReader:
    def __init__(self, response):
        self.response = response
        self.items = self.response["items"]

        data = []
        for item in self.items:
            id = item["contentDetails"]["videoId"]
            doc_url = "".join(["https://www.youtube.com/watch?v=", item["contentDetails"]["videoId"]])
            date = datetime.strptime(item["contentDetails"]["videoPublishedAt"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d")
            try:
                transcript = TextFormatter().format_transcript(
                    YouTubeTranscriptApi.get_transcript(id, languages=["vi"]))
            except Exception:
                print("Could not extract video: %s" % doc_url)

            if transcript:
                sentences = transcript.split("\n")
                for sentence in sentences:
                    if len(sentence.replace("♪", "").strip()):
                        s = {
                            "doc_url": doc_url,
                            "date": date,
                            "sentence": sentence.replace("♪", "").strip()
                        }

                        data.append(s)

        self.data = data

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    playlists = ["PLH_v4r_pvudV5ZrNx9HldKLICIjUSCRLb"]
    print("YouTube playlists: %s" % len(playlists))
    print("You will be required to authenticate each!")

    # Get all lyrics from playlists
    all_data = []
    for pl in playlists:
        response = playlist_youtube_api(pl)
        data = COLReader(response).data
        print("Playlist videos: %s" % len(data))
        all_data.extend(data)
    print("Sample\n", all_data[0])
    print("Total lyric lines %s" % len(all_data))

    # Write to file
    content = ""
    for i, s in enumerate(all_data):
        doc_url = "# doc_url = " + s["doc_url"]
        date = "# date = " + s["date"]
        sent_id = "# sent_id = " + str(i + 1)
        sent = s["sentence"]
        content += "\n".join([doc_url, date, sent_id, sent, "\n"])

    target_file = join(COL_FOLDER, "corpus", "raw", "lyrics.txt")
    with open(target_file, "w") as f:
        f.write(content)
