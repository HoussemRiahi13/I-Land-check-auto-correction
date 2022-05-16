import os
from datetime import datetime

from django.db import models

# Create your models here.

class Check (models.Model):
    Type = models.BooleanField
    Nom = models.CharField(max_length=100)

