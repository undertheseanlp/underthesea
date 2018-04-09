#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Truong Do <truongdq54@gmail.com>
#

import os
import sys
if not sys.version_info >= (3, 0):
    raise Exception("The current version of hts only supports python > 3")

import htsengine


root_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(root_path, "models")
voice_model = os.path.join(root_path, 'model_v1.htsvoice')

class SpeechError(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super(ValidationError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors

def sanity_check():
    if not os.path.isdir(model_path) or \
            not os.path.isfile(voice_model):
        raise SpeechError("Model directory or voice file does not exists")

def speak(text):
    """
    Vietnamese text-to-speech

    Parameters
    ==========

    text: {unicode, str}
        input text sentence

    Returns
    =======
    audio_path:   the audio path

    Examples
    --------
    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import speak
    >>> sentence = "Nghi vấn 4 thi thể Triều Tiên trôi dạt bờ biển Nhật Bản"
    >>> speak(sentence)
    """


    if not text:
        return
    sanity_check() # Raise SpeechError exception if anything went wrong

    import tempfile
    from subprocess import Popen,PIPE,STDOUT


    fd_text, fname_text = tempfile.mkstemp()
    fd_wav, fname_wav = tempfile.mkstemp(suffix=".wav")
    dirpath_full_ctx = tempfile.mkdtemp()
    dirpath_mono_ctx = tempfile.mkdtemp()

    try:
        open(fname_text, "w").write("test|" + text)

        cmd_ana = "{cmd} {model} {fname_text} {full_dir} {mono_dir}".format(
                cmd=os.path.join(root_path, "vita_ana"), model=model_path,
                fname_text=fname_text, full_dir=dirpath_full_ctx,
                mono_dir=dirpath_mono_ctx
                )
        out = Popen(cmd_ana.split(), stderr=STDOUT,stdout=PIPE)
        t = out.communicate()[0]
        if out.returncode:
            return

        label = [line.rstrip() for line in open(os.path.join(dirpath_full_ctx, "test.lab"))]

        s, f, n, a = htsengine.synthesize(voice_model, label)
        import wave
        wavFile = wave.open(fname_wav, 'wb')
        wavFile.setsampwidth(s)
        wavFile.setframerate(f)
        wavFile.setnchannels(n)
        wavFile.writeframesraw(a)
        wavFile.close()
        return fname_wav
    finally:
        import shutil
        shutil.rmtree(dirpath_full_ctx)
        shutil.rmtree(dirpath_mono_ctx)
        os.remove(fname_text)


if __name__ == "__main__":
    print(speak("xin chào các bạn"))
