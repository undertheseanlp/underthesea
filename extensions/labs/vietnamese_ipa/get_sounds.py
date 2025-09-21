from time import sleep

import requests

from underthesea.utils import logger

url = 'https://api.fpt.ai/hmi/tts/v5'


class FPTAI:
    @staticmethod
    def tts(text, output_file):
        payload = text
        headers = {
            'api-key': 'WW5ZxG2GZBKyilP5joBjcPfiQsTcshON',
            'speed': '',
            'voice': 'thuminh'
        }

        url = 'https://api.fpt.ai/hmi/tts/v5'
        response = requests.request('POST', url, data=payload.encode('utf-8'), headers=headers)

        mp3_file_url = response.json()['async']

        r = requests.get(mp3_file_url, allow_redirects=True)
        with open(output_file, 'wb') as f:
            f.write(r.content)


def download_file(file_url, output_file, retries=0):
    if retries < 0:
        message = f"Failed to download {file_url}"
        logger.error(message)
        return

    r = requests.get(file_url, allow_redirects=True)
    if len(r.content) == 0:
        message = f"Retry download {file_url}"
        logger.info(message)
        download_file(file_url, output_file, retries=retries - 1)
        return
    if len(r.content) > 0:
        with open(output_file, 'wb') as f:
            f.write(r.content)
    else:
        message = f'File {file_url} cannot download.'
        logger.error(message)


class ZaloAI:
    @staticmethod
    def tts(text, output_file):
        headers = {
            'apikey': 'mcsKsLLUX23ekzfhShOcXVHLVvlXEE9q',
            'content-type': 'application/x-www-form-urlencoded',
        }

        # data = f'input={text}'.encode('utf-8')
        data = {"input": text, "speaker_id": 2, "quality": 1}

        response = requests.post('https://api.zalo.ai/v1/tts/synthesize', headers=headers, data=data)
        mp3_file_url = response.json()['data']['url']

        download_file(file_url=mp3_file_url, output_file=output_file, retries=3)


if __name__ == '__main__':
    # text = 'chào'
    # service = 'fpt'
    # output = f'outputs/sound/{service}/{text}.mp3'
    # FPTAI.tts(text, output)
    # ZaloAI.tts("bay", "bay.mp3")
    with open('../text_normalize/outputs/syllables.txt') as f:
        lines = f.readlines()
        all_texts = [line.strip() for line in lines]
    SOUND_DOWNLOAD_SUCCESS_FILE = '../text_normalize/outputs/syllables_sound_download_success.txt'
    SOUND_DOWNLOAD_FAILED_FILE = '../text_normalize/outputs/syllables_sound_download_failed.txt'
    with open(SOUND_DOWNLOAD_SUCCESS_FILE) as f:
        lines = f.readlines()
        texts = [line.strip() for line in lines]
        syllables_sound_download_success = set(texts)
    with open(SOUND_DOWNLOAD_FAILED_FILE) as f:
        lines = f.readlines()
        texts = [line.strip() for line in lines]
        syllables_sound_download_failed = set(texts)
    MAX_FAILED = 5
    COUNT_FAILED = 0
    MAX_CALL = 6000
    COUNT_CALL = 0
    for text in all_texts:
        if text not in syllables_sound_download_success and text not in syllables_sound_download_failed:
            sleep(0.05)
            COUNT_CALL += 1
            if MAX_CALL is not None:
                if COUNT_CALL >= MAX_CALL:
                    break
            try:
                service = 'zalo'
                output = f'outputs/sound/{service}/{text}.mp3'
                ZaloAI.tts(text, output)
                # add to download success
                syllables_sound_download_success.add(text)
                logger.info(f'Download sound {text} success.')
                with open(SOUND_DOWNLOAD_SUCCESS_FILE, "w") as f:
                    content = "\n".join(syllables_sound_download_success)
                    f.write(content)
            except Exception:
                COUNT_FAILED += 1
                syllables_sound_download_failed.add(text)
                with open(SOUND_DOWNLOAD_FAILED_FILE, "w") as f:
                    content = "\n".join(syllables_sound_download_failed)
                    f.write(content)
                logger.info(f'Download sound {text} failed.')
                if COUNT_FAILED > 5:
                    message = 'Max failed exceeded.'
                    logger.error(message)
                    raise Exception(message)
