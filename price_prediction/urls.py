from django.contrib import admin
from django.urls import path
from price_prediction import views

app_name = 'price_prediction'

urlpatterns = [
    path('analyse_data/', views.analyse_data, name='analyse_data'),
    path('import_data/', views.import_data, name='import_data'),
    path('model_train/', views.model_train, name='model_train')
]