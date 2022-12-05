from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json
from .helpers import BSBIIndex, VBEPostings
from django.core.files import File
from django.contrib.staticfiles.storage import staticfiles_storage
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
def meedle_view(request):
    return HttpResponse("<h1> Welcome to Meedle</h1>")

def endpoint_test(request, keyword):
    data = {
        "resp": "ok",
        "keyword": keyword,
    }
    return JsonResponse(data, safe=False)

@csrf_exempt 
def search_query(request):
    body = json.loads(request.body)
    if request.method != "POST" or "query" not in body:
        return HttpResponse(status=400)

    query = body["query"]
    topk = 1033 # all docs collection
    if "k" in body:
        if type(body["k"]) != int:
            return HttpResponse(status=400)
        topk = body["k"]

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
        postings_encoding = VBEPostings, \
        output_dir = 'index')

    docs = []
    for (_, doc) in BSBI_instance.retrieve_bm25(query, k = topk):
        docs.append(doc)
    
    response = {
        "query": query,
        "k": topk,
        "retrieved": len(docs),
        "docs_id": docs,
    }

    return JsonResponse(response, safe=False)

@csrf_exempt 
def get_docs(request):

    body = json.loads(request.body)
    if request.method != "POST" or "docs_id" not in body:
        return HttpResponse(status=400)

    result = {}
    for doc_id in body["docs_id"]:
        url = staticfiles_storage.url(f'collection/{str(doc_id)}')
        with open(url[1:], 'r') as f:
            result[doc_id] = File(f).read()

    return JsonResponse(result, safe=False)
