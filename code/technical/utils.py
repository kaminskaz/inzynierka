import os
import json
import pandas as pd
from PIL import Image
import torch

#  na razie nie wiem czy bedzie potrzebne wgl utils,
#  bo na ten moment to takie wsm krotkie funkcje mozna rownie dobrze w glownym pliku napisac
#  ale zostawiam na razie

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().strip()
    return data



