from django.urls import path
from . import views

urlpatterns = [
    path('ravidata_new', views.Value1, name='ravi_baba_new')
]
