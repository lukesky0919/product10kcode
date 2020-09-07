import csv
import shutil
import os

target_path = 'F:\\product10ktrain'
original_path = 'F:\\train\\train'
with open('train.csv',"rt", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    for row in rows:
        print(row[0],row[1])
        if os.path.exists(target_path+ '\\' +row[1]) :
            full_path = original_path + '\\' + row[0]
            shutil.move(full_path,target_path + '\\' + row[1] )
        else :
            os.makedirs(target_path+ '\\' +row[1])
            full_path = original_path + '\\' +row[0]
            shutil.move(full_path,target_path + '\\' + row[1] )