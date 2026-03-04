# usuarios/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    # Usamos 'username' para el login/registro, pero lo llamamos 'uvus' en el frontend.
    # Ya que 'uvus' es tu identificador único, lo puedes mapear al campo 'username' por defecto,
    # pero si quieres un campo 'uvus' separado:
    uvus = models.CharField(max_length=150, unique=True)
    grado = models.CharField(max_length=255, blank=True)
    # Email ya existe en AbstractUser, pero a menudo se añade aquí para personalizarlo
    email = models.EmailField(unique=True)
    minutos_conectado = models.PositiveIntegerField(default=0)

    # Opcional: Si quieres que 'uvus' sea el campo de login principal,
    # tendrías que sobrescribir el USERNAME_FIELD y posiblemente el Manager.

    # Si usas 'username' por defecto, simplemente borra la línea de 'uvus' arriba
    # y mapea 'uvus' del frontend a 'username' en el serializer.