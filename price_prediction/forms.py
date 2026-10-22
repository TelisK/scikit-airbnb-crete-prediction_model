from django import forms

class DataUploadForm(forms):
    file = forms.FileField()