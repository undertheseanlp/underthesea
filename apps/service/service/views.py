import underthesea as uts
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from underthesea.dictionary import Dictionary
import json

from underthesea.pipeline.ipa import viet2ipa


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def word_sent(request):
    result = {}
    try:
        text = json.loads(request.body.decode("utf-8"))["text"]
        tags = uts.word_sent(text)
        result["output"] = tags
    except Exception as e:
        print(e)
        result = {"error": "Bad request!"}
    return JsonResponse(result)


@csrf_exempt
def pos_tag(request):
    result = {}
    try:
        text = json.loads(request.body.decode("utf-8"))["text"]
        tags = uts.pos_tag(text)
        result["output"] = tags
    except Exception as e:
        print(e)
        result = {"error": "Bad request!"}
    return JsonResponse(result)


@csrf_exempt
def chunking(request):
    result = {}
    try:
        text = json.loads(request.body.decode("utf-8"))["text"]
        tags = uts.chunk(text)
        result["output"] = tags
    except Exception as e:
        print(e)
        result = {"error": "Bad request!"}
    return JsonResponse(result)


@csrf_exempt
def ner(request):
    result = {}
    try:
        text = json.loads(request.body.decode("utf-8"))["text"]
        tags = uts.ner(text)
        result["output"] = tags
    except Exception as e:
        print(e)
        result = {"error": "Bad request!"}
    return JsonResponse(result)


@csrf_exempt
def classification(request):
    result = {}
    try:
        data = json.loads(request.body.decode("utf-8"))
        text = data["text"]
        domain = data["domain"] if data["domain"] != "general" else None
        tags = uts.classify(text, domain=domain)
        result["output"] = tags
    except Exception as e:
        print(e)
        result = {"error": "Bad request!"}
    return JsonResponse(result)


@csrf_exempt
def sentiment(request):
    result = {}
    try:
        data = json.loads(request.body.decode("utf-8"))
        text = data["text"]
        tags = uts.sentiment(text, domain="bank")
        result["output"] = tags
    except Exception as e:
        print(e)
        result = {"error": "Bad request!"}
    return JsonResponse(result)


@csrf_exempt
def dictionary(request):
    result = {}
    uts_dict = Dictionary.Instance()
    try:
        text = json.loads(request.body.decode("utf-8"))["text"]
        word = uts_dict.lookup(text)
        result["output"] = word
    except Exception as e:
        print(e)
        result = {"error": "Bad request!"}
    return JsonResponse(result)


def generate_id(word):
    return "".join([hex(ord(c)) for c in word])


@csrf_exempt
def ipa(request):
    result = {}
    try:
        text = json.loads(request.body.decode("utf-8"))["text"]
        result["ipa"] = viet2ipa(text)
        sound_repo = "https://github.com/undertheseanlp/underthesea/releases/download/open-data-voice-ipa/"
        sound_url = sound_repo + generate_id(text) + ".mp3"
        result["audio_src"] = sound_url
    except Exception as e:
        print(e)
        result = {"error": "Bad request!"}
    return JsonResponse(result)
