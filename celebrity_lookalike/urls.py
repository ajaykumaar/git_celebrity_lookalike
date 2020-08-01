from django.urls import path

from . import views 

urlpatterns=[

    path('',views.index,name='index'),
    path('prediction',views.predict,name='predict'),
    path('actress_prediction',views.actress_predict,name='actress_predict'),
    path('actor_classes',views.actor_classes,name='actors_list'),
    path('actress_classes',views.actress_classes,name='actress_list')
]