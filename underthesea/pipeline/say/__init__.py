from viettts_.hifigan.mel2wave import mel2wave
from viettts_.nat.text2mel import text2mel
from viettts_ import nat_normalize_text
import numpy as np
from playsound import playsound
import time


def text_to_speech(text):
    # prevent too long text
    if len(text) > 500:
        text = text[:500]
    text = nat_normalize_text(text)
    mel = text2mel(
        text,
        "lexicon.txt",
        0.2,
        "acoustic_latest_ckpt.pickle",
        "duration_latest_ckpt.pickle",
    )
    wave = mel2wave(mel, "config.json", "hk_hifi.pickle")
    return (wave * (2**15)).astype(np.int16)


def say(text):
    y = text_to_speech(text)
    # write y array to sound.wav
    import soundfile as sf
    sf.write("sound.wav", y, 16_000)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"\n{time_str}: {text}")
    return 16_000, y


def think_and_say(text):
    # create two processes, one for thinking, one for saying
    import multiprocessing
    p1 = multiprocessing.Process(target=terminal_thinking)
    p2 = multiprocessing.Process(target=say, args=(text,))
    p1.start()
    p2.start()
    # while p2 is end, terminate p1
    p2.join()
    p1.terminate()
    playsound("sound.wav")


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
