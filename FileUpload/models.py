from django.db import models


# Create your models here.

class csv_path(models.Model):
    csv_file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

# class SignUp()
