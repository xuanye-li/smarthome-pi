from django.db import models

# Create your models here.

class Classification(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    label = models.CharField(max_length=100)
    media_file = models.FileField(upload_to='uploads/', null=True, blank=True)

    def __str__(self):
        return f"{self.label} at {self.timestamp}"

