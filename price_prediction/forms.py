from django import forms
from .models import DataInformation

class DataUploadForm(forms.Form):
    file = forms.FileField()

class DataInformationForm(forms.ModelForm):
    class Meta:
        model = DataInformation
        fields = ['data_country', 'data_area', 'data_town', 'data_date']
        widgets = {
            'data_date': forms.DateInput(attrs={'type':'date'}, format=['%d/%m/%Y', '%d-%m-%Y']),
        }