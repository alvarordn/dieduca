from django.urls import path
from . import views

urlpatterns = [
    # Captura las peticiones POST procedentes del CircuitosService de Angular
    path('generar_circuito/', views.generar_circuito, name='generar_circuito'),
]