from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import json
import fastai2
from fastai2 import *
from fastai2.vision.all import *

# Create your views here.
def index(request):
    load_artifacts()
    return render(request,'index.html')

def predict(request):
    
    if request.method == 'POST':
        up_file=request.FILES['image']
        fs=FileSystemStorage()
        fs.save(up_file.name, up_file)

    load_artifacts()
    image=PILImage.create(up_file)
    cat,idx,prob=actor_model.predict(image)
    max_prob=prob[idx]     
    get_results(cat,idx,max_prob)

    return render(request, 'prediction.html',{'match_name' : match_name,
                                              'match_img' : match_img,
                                              'match_p'  : match_p})

def actress_predict(request):
    if request.method == 'POST':
        up_file=request.FILES['image']
        fs=FileSystemStorage()
        fs.save(up_file.name, up_file)

    load_artifacts()
    image=PILImage.create(up_file)
    cat,idx,prob=actress_model.predict(image)
    max_prob=prob[idx]     
    actress_get_results(cat,idx,max_prob)

    return render(request, 'prediction.html',{'match_name' : match_name,
                                              'match_img' : match_img,
                                              'match_p'  : match_p})


def actor_classes(request):
    load_artifacts()
    classes_dict={
        'ajith':actor_classes[0],
        'aatharva':actor_classes[1],
        'dhanush':actor_classes[2],
        'kamal':actor_classes[3],
        'karthi':actor_classes[4],
        'siva_karthi':actor_classes[5],
        'surya':actor_classes[6],
        'vadivel':actor_classes[7],
        'vijay':actor_classes[8],
        'vjs':actor_classes[9],
        'vikram':actor_classes[10],
        'vishal':actor_classes[11],
    
    }
    return render(request,'actor_list.html',{'classes':classes_dict} )

def actress_classes(request):
    load_artifacts()
    classes_dict={
        'aish':actress_classes[0],
        'amala':actress_classes[1],
        'genelia':actress_classes[2],
        'hansika':actress_classes[3],
        'kajal':actress_classes[4],
        'madonna':actress_classes[5],
        'megha':actress_classes[6],
        'priya':actress_classes[7],
        'sai':actress_classes[8],
        'samantha':actress_classes[9],
        'shalini':actress_classes[10],
        'sneha':actress_classes[11],
        'tamanna':actress_classes[12]
    }

    return render(request, 'actress_list.html',{'classes' : classes_dict})


def load_artifacts():
    
    global actor_classes, actress_classes
    global actor_model, actress_model
    with open('./artifacts/actor_classes.json','r') as f:
        actor_classes=json.load(f)

    actor_model=load_learner('./artifacts/actor_learn.pkl')

    with open('./artifacts/actress_classes.json','r') as f:
        actress_classes=json.load(f)
    actress_model=load_learner('./artifacts/actress_learn.pkl')
        
    
def get_results(cat,idx,max_prob):
    global match_name,match_img,match_p
    
    actor_img={
        actor_classes[0] : 'ajith.jpg', 
        actor_classes[1] : 'aatharva.jpg',
        actor_classes[2] : 'dhanush.jpg',
        actor_classes[3] : 'kamal.jpg',
        actor_classes[4] : 'karthi.jpg',
        actor_classes[5] : 'siva karthi.jpg',
        actor_classes[6] : 'surya.jpg',
        actor_classes[7] : 'vadivel.jpg',
        actor_classes[8] : 'vijay.jpg',
        actor_classes[9] : 'vijay sethupathi.jpg',
        actor_classes[10] : 'vikram.jpg',
        actor_classes[11] : 'vishal.jpg'
        }
    match_img=actor_img.get(cat)
    match_name=cat
    match_p="{:.2%}".format(max_prob)

def actress_get_results(cat,idx,max_prob):
    global match_name,match_img,match_p
    
    actor_img={
        actress_classes[0] : 'aish_rajesh.jpg', 
        actress_classes[1] : 'amala.jpg',
        actress_classes[2] : 'genelia.jpg',
        actress_classes[3] : 'hansika.jpg',
        actress_classes[4] : 'kajal.jpg',
        actress_classes[5] : 'madonna.jpg',
        actress_classes[6] : 'megha.jpg',
        actress_classes[7] : 'priya.jpg',
        actress_classes[8] : 'sai.jpg',
        actress_classes[9] : 'samantha.jpg',
        actress_classes[10] : 'shalini.jpg',
        actress_classes[11] : 'sneha.jpg',
        actress_classes[12] : 'tamanna.jpg'
        }
    match_img=actor_img.get(cat)
    match_name=cat
    match_p="{:.2%}".format(max_prob)
    