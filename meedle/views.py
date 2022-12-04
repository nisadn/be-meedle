from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def meedle_view(request):
    return HttpResponse("<h1> Django Deployed meedle</h1>")