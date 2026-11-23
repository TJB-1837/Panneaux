import json
import os
from PIL import Image, ImageDraw, ImageColor, ImageFont
import csv
from collections import defaultdict
import pandas as pd

def load_annotation(image_key):
    try:
        with open(os.path.join('Full_Data/data1/mtsd_v2_fully_annotated/annotations', '{:s}.json'.format(image_key)), 'r') as fid:
            anno = json.load(fid)
        return anno
    except FileNotFoundError as e:
        print(f'Fichier {image_key}.json introuvable; code : {e}')
        return None

def getPannelsFromImage(images_dir, csvfile, output_dir='Full_Data/full_data_cropped'):
    labels = pd.read_csv(csvfile)
    label_to_id = { row['Name'] : row['ClassId'] for col, row in labels.iterrows()}
    os.makedirs(output_dir, exist_ok=True)
    nb_no_anno = 0
    for image_key in os.listdir(images_dir):
        image_key = image_key[:-4]
        anno = load_annotation(image_key)
        if(anno == None):
            nb_no_anno += 1
            continue
        with Image.open(os.path.join('Full_Data/images', '{:s}.jpg'.format(image_key))) as img:
            for i,obj in enumerate(anno['objects']):
                #on ignore les cas où le panneau est peu visible
                if obj["properties"]["ambiguous"]:
                    continue 
                if obj["properties"]["occluded"]:
                    continue
                if obj["label"] == 'other-sign':
                    continue
                #récupération d'une bounding box pour crop
                x1 = obj['bbox']['xmin']
                y1 = obj['bbox']['ymin']
                x2 = obj['bbox']['xmax']
                y2 = obj['bbox']['ymax']

                if((x2-x1)*(y2-y1)>=100):
                    img_cropped =  img.crop((x1,y1,x2,y2))
                    img_resized = img_cropped.resize((64,64))

                    label = obj["label"]
                    if label in label_to_id:
                        class_id = label_to_id[label]
                        class_path = os.path.join(output_dir, str(class_id))
                        os.makedirs(class_path,exist_ok=True)
                        output_path = os.path.join(class_path,f"{image_key}_{class_id}_{i}.jpg")
                        img_resized.save(output_path)
    print(f'images without annotations : {nb_no_anno}')
    return            

    

def occurences(annotations_url,labels):
    try:
        nb_images = 0
        for json_file in os.listdir(annotations_url):
            with open(os.path.join(annotations_url,json_file),'r') as f:
                data = json.load(f)
                
                for obj in data['objects']:
                    label = obj['label']
                    if(label == 'other-sign'):
                        continue
                    labels[label] += 1
                    nb_images += 1
        print(f'nb images : {nb_images}')
        
        print(f"Nombre total de classes avant filtrage : {len(labels)}")
        filtered_labels = {k: v for k, v in labels.items() if v >= 200}
        print(f"Nombre de classes après filtrage : {len(filtered_labels)}")
        return labels
    



    except Exception as e:
        print(f'Erreur dans {json_file} : {e}')
        return None


def csvBuild(label_url):
    if label_url == None : 
        print(f'Erreur dans label_url => None')
        return
    with open(label_url, 'w', newline='') as f:
        writer = csv.writer(f)  
        writer.writerow(["ClassId","Name","Occurences"])
        sorted_labels = sorted(f.items(), key=lambda x: x[1], reverse=True)
        for i, (label,number) in enumerate(sorted_labels):
            if(number>=200):
                writer.writerow([i,label,number])


if __name__ == '__main__':
    #image_key = 'Bh36Ed4HBJatMpSNnFTgTw'
    annotations_url = 'Full_Data/data1/mtsd_v2_fully_annotated/annotations'
    images_url = 'Full_Data/images'
    label_url = 'labels401.csv'
    crop_dir = 'Full_Data/full_data_cropped'
    old_data_path = 'myData'

    

    # comptage des occurence de chaque type
    #labels = defaultdict(int)
    #labels = occurences(annotations_url,labels)

    # création et écriture dans le fichier csv
    #csvBuild(label_url)
    #print(f"Fichier labels401.csv généré avec {len(labels)} classes.")

    # on crop les panneaux dans les images et on les stock dans le bon répertoire
    #getPannelsFromImage(images_url,label_url)
    ctn = 0
    for i in os.listdir(old_data_path):
        ctn += len(os.listdir(os.path.join(old_data_path,i)))
    print(ctn)


