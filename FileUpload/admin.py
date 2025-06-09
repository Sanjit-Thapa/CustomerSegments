from django.contrib import admin
from .models import csv_path
# Register your models here.

class listCsvPath(admin.ModelAdmin):
    list_display = ['csv_file','uploaded_at']

admin.site.register(csv_path,listCsvPath)