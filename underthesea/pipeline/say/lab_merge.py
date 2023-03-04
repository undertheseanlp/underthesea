from pydub import AudioSegment

if __name__ == '__main__':
    sound1 = AudioSegment.from_wav("sound.wav")
    sound2 = AudioSegment.from_wav("chill.wav")

    # overlay sound2 over sound1 at the 5th second mark
    combined_sounds = sound1.overlay(sound2)

    combined_sounds.export("combined_sounds.wav", format="wav")
