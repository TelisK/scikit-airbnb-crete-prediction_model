from django.contrib import admin
from .models import UploadedFile, DataInformation, DataSet

admin.site.register(UploadedFile)
admin.site.register(DataInformation)
admin.site.register(DataSet)

# Register your models here.
