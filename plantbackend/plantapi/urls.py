from django.contrib import admin
from django.urls import path
from plantapi import views

urlpatterns = [
    path('', view=views.home),
    path('get-prediction/', view=views.make_prediction),
    path('get-seedling/', view=views.make_prediction_for_seedling)
]
