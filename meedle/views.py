from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json
from bsbi.views import BSBIIndex, VBEPostings, IdMap
from django.core.files import File
from django.contrib.staticfiles.storage import staticfiles_storage

# Create your views here.
def meedle_view(request):
    return HttpResponse("<h1> Django Deployed meedle</h1>")

def search(request, keyword):
    data = {
        "resp": "ok",
        "keyword": keyword,
    }
    return JsonResponse(data, safe=False)

def search_with_body(request):
    body = json.loads(request.body)
    if "query" not in body:
        return HttpResponse(status=400)
    return JsonResponse(body, safe=False)

def search_query(request):
    body = json.loads(request.body)
    if "query" not in body:
        return HttpResponse(status=400)

    query = body["query"]

    # BSBI_instance = BSBIIndex(data_dir = 'collection', \
    #     postings_encoding = VBEPostings, \
    #     output_dir = 'index')

    docs = []
    # for (_, doc) in BSBI_instance.retrieve_bm25(query, k = 10):
    #     docs.append(doc)
    
    response = {
        "query": query,
        "docs_id": docs,
    }

    return JsonResponse(response, safe=False)

def get_docs(request):

    body = json.loads(request.body)
    if "docs_id" not in body:
        return HttpResponse(status=400)

    result = {}
    for doc_id in body["docs_id"]:
        url = staticfiles_storage.url(f'collection/{str(doc_id)}')
        with open(url[1:], 'r') as f:
            result[doc_id] = File(f).read()

    return JsonResponse(result, safe=False)
