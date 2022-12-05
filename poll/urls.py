"""poll URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from meedle.views import (meedle_view, search, search_with_body, search_query, get_docs)


urlpatterns = [
    path('' , meedle_view , name="meedle"),
    path('search/<str:keyword>', search, name="search"),
    path('search_with_body', search_with_body, name="search_with_body"),
    path('search_query', search_query, name="search_query"),
    path('get_docs', get_docs, name="get_docs"),
    path('admin/', admin.site.urls),
]
