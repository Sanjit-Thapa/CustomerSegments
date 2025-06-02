from django import forms
from .models import csv_path

class UploadCSV(forms.ModelForm):

     class Meta:
        model = csv_path
        fields = ['csv_file']

    



