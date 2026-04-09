from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser
from django.contrib import admin
from django.db.models import Sum
from .models import CustomUser, IntentoEjercicio

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    # Añadimos  la lista para ver los campos
    list_display = ('username', 'uvus', 'email', 'grado',"ejercicios_totales","total_aciertos", 'is_staff', 'tiempo_en_web')

    # Añadimos el campo a los detalles del usuario para que puedas editarlo si quieres
    fieldsets = UserAdmin.fieldsets + (
        ('Información Extra', {'fields': ('uvus', 'grado')}),
    )

    # Creamos una "columna calculada" para que no salgan solo números, sino algo legible
    def tiempo_en_web(self, obj):
        horas = obj.minutos_conectado // 60
        minutos = obj.minutos_conectado % 60
        if horas > 0:
            return f"{horas}h {minutos}min"
        return f"{minutos} min"

    # Otra columna calculada para mostrar el total de ejercicios (aciertos + fallos)
    def ejercicios_totales(self, obj):
        datos = obj.intentos.aggregate(total_a=Sum('aciertos'),total_f=Sum('fallos'))
        total = (datos['total_a'] or 0) + (datos['total_f'] or 0)
        return f"{total}"
    ejercicios_totales.short_description = 'Ejercicios Totales'

    # Otra columna calculada para mostrar los aciertos totales
    def total_aciertos(self, obj):
        datos = obj.intentos.aggregate(total_a=Sum('aciertos'))
        return f"{datos['total_a'] or 0}"
    total_aciertos.short_description = 'Total Aciertos'

@admin.register(IntentoEjercicio)
class IntentoEjercicioAdmin(admin.ModelAdmin):
    # Columnas que verás en la tabla principal
    list_display = ('usuario', 'uvus_del_alumno', 'bloque_id', 'aciertos', 'fallos', 'fecha')

    # Hacemos que la fecha no se pueda editar manualmente
    readonly_fields = ('fecha',)

    # Función extra para ver el UVUS directamente en la tabla
    def uvus_del_alumno(self, obj):
        return obj.usuario.uvus
