from underthesea.file_utils import MODELS_FOLDER
from underthesea.pipeline.say.viettts_.hifigan.mel2wave import mel2wave
from underthesea.pipeline.say.viettts_.nat.text2mel import text2mel
from underthesea.pipeline.say.viettts_ import nat_normalize_text
import numpy as np
from playsound import playsound
import time
from os.path import join


model_path = join(MODELS_FOLDER, "VIET_TTS_V0_4_1")


def text_to_speech(text):
    # prevent too long text
    if len(text) > 500:
        text = text[:500]
    text = nat_normalize_text(text)
    mel = text2mel(
        text,
        join(model_path, "lexicon.txt"),
        0.2,
        join(model_path, "acoustic_latest_ckpt.pickle"),
        join(model_path, "duration_latest_ckpt.pickle"),
    )
    wave = mel2wave(mel, join(model_path, "config.json"), join(model_path, "hk_hifi.pickle"))
    return (wave * (2**15)).astype(np.int16)


def say(text, outfile="sound.wav", play=False):
    y = text_to_speech(text)
    # write y array to sound.wav
    import soundfile as sf
    sf.write(outfile, y, 16_000)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"\n{time_str}: {text}")
    if play:
        playsound(outfile)
    return 16_000, y


def think_and_say(text, outfile="sound.wav"):
    # create two processes, one for thinking, one for saying
    import multiprocessing
    p1 = multiprocessing.Process(target=terminal_thinking)
    p2 = multiprocessing.Process(target=say, args=(text, outfile))
    p1.start()
    p2.start()
    # while p2 is end, terminate p1
    p2.join()
    p1.terminate()
    playsound(outfile)


def terminal_thinking():
    # each 300ms, print a dot
    import time
    while True:
        print(".", end="")
        time.sleep(0.5)


if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    think_and_say("xin chào")
    think_and_say("Có đấy. Tình yêu giống như một ly kem, nếu không ăn nó đúng lúc, nó sẽ tan chảy. Haha, anh thấy thế nào?")
