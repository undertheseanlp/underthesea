import time

_VALID_BACKENDS = ("viettts", "vieneu")


def _synthesize_viettts(text):
    from os.path import join

    import numpy as np

    from underthesea.file_utils import MODELS_FOLDER
    from underthesea.pipeline.tts.viettts_ import nat_normalize_text
    from underthesea.pipeline.tts.viettts_.hifigan.mel2wave import mel2wave
    from underthesea.pipeline.tts.viettts_.nat.text2mel import text2mel

    model_path = join(MODELS_FOLDER, "VIET_TTS_V0_4_1")
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
    y = (wave * (2**15)).astype(np.int16)
    return 16_000, y


def _synthesize_vieneu(text):
    try:
        from vieneu import Vieneu
    except ImportError:
        raise ImportError(
            "The 'vieneu' package is required for backend='vieneu'. "
            "Install it with: pip install underthesea[voice-vieneu]"
        )

    import numpy as np
    engine = Vieneu()
    audio = engine.infer(text)
    sample_rate = 48_000
    if audio.dtype != np.int16:
        if np.issubdtype(audio.dtype, np.floating):
            peak = np.max(np.abs(audio)) or 1.0
            audio = (audio / peak * 32767).astype(np.int16)
        else:
            audio = audio.astype(np.int16)
    return sample_rate, audio


def text_to_speech(text, backend="viettts"):
    if backend == "viettts":
        return _synthesize_viettts(text)
    elif backend == "vieneu":
        return _synthesize_vieneu(text)
    else:
        raise ValueError(
            f"Unknown TTS backend '{backend}'. Choose from: {_VALID_BACKENDS}"
        )


def tts(text, outfile="sound.wav", play=False, backend="viettts"):
    sample_rate, y = text_to_speech(text, backend=backend)
    import soundfile as sf
    sf.write(outfile, y, sample_rate)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"\n{time_str}: {text}")
    if play:
        from playsound3 import playsound
        playsound(outfile)
    return sample_rate, y


say = tts


def think_and_tts(text, outfile="sound.wav", backend="viettts"):
    import multiprocessing
    p1 = multiprocessing.Process(target=terminal_thinking)
    p2 = multiprocessing.Process(target=tts, args=(text, outfile), kwargs={"backend": backend})
    p1.start()
    p2.start()
    p2.join()
    p1.terminate()
    from playsound3 import playsound
    playsound(outfile)


def terminal_thinking():
    import time
    while True:
        print(".", end="")
        time.sleep(0.5)


if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    think_and_tts("xin chào")
    think_and_tts("Có đấy. Tình yêu giống như một ly kem, nếu không ăn nó đúng lúc, nó sẽ tan chảy. Haha, anh thấy thế nào?")
