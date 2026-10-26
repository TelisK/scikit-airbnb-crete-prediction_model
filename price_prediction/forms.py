from django import forms
from .models import DataInformation

class DataUploadForm(forms.Form):
    file = forms.FileField()

class DataInfomationForm(forms.ModelForm):
    model = 'DataInformation'
    fields = ['data_country', 'data_area', 'data_town', 'data_date']