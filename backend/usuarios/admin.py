from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    # Añadimos 'minutos_conectado' a la lista para verlo de un vistazo
    list_display = ('username', 'uvus', 'email', 'grado', 'is_staff', 'tiempo_en_web')

    # Añadimos el campo a los detalles del usuario para que puedas editarlo si quieres
    fieldsets = UserAdmin.fieldsets + (
        ('Información Extra', {'fields': ('uvus', 'grado', 'minutos_conectado')}),
    )

    # Creamos una "columna calculada" para que no salgan solo números, sino algo legible
    def tiempo_en_web(self, obj):
        horas = obj.minutos_conectado // 60
        minutos = obj.minutos_conectado % 60
        if horas > 0:
            return f"{horas}h {minutos}min"
        return f"{minutos} min"

    # Le ponemos un nombre bonito a la columna
    tiempo_en_web.short_description = 'Tiempo de Uso'