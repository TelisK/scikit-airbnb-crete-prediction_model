from django.contrib import admin
from django.urls import path
from price_prediction import views

urlpatterns = [
    path('analyse_data/', views.analyse_data, name='analyse_data'),
]