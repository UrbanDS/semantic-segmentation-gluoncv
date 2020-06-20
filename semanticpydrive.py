# ------- gdrive ------
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import glob
import os
import random
# ------- gdrive -----
# ------- gluon ----
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms.presets.segmentation import test_transform
from matplotlib import pyplot as plt
import glob
import gluoncv
from skimage import io
import mxnet as mx
from mxnet import image
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
import glob
import os

import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms.presets.segmentation import test_transform
from matplotlib import pyplot as plt
import gluoncv
import numpy
from PIL import Image


model = gluoncv.model_zoo.get_model('psp_resnet101_ade', pretrained=True, ctx=mx.cpu(0))
ctx = mx.cpu(0)
import numpy
from PIL import Image
import csv
import glob
from datetime import datetime
from timeit import default_timer as timer

# ------- gluon ----
from database import mydatabase
dbms = mydatabase.MyDatabase(mydatabase.SQLITE, dbname='/Users/divyachandana/Documents/NJIT/work/summertasks/june15-june20/semantic-segmentation-pixel/semanticdb.sqlite')


def main():
    start = timer()
    print('Processing Start time: %.1f' % (start))
    print("current time", datetime.now())
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()

    drive = GoogleDrive(gauth)

    # Auto-iterate through all files that matches this query
    file_list = drive.ListFile({'q': "'root' in parents"}).GetList()
    for file in file_list:
        # print('title: {}, id: {}'.format(file1['title'], file1['id']))
        file_id = None

        if file['title'] == "semanticsegmentation":
            print('Folder Found')
            file_id = file['id']
            break
    if file_id is not None:
        classes = ["wall","building;edifice","sky","floor;flooring","tree","ceiling","road;route","bed","windowpane;window","grass","cabinet","sidewalk;pavement","person;individual;someone;somebody;mortal;soul","earth;ground","door;double;door","table","mountain;mount","plant;flora;plant;life","curtain;drape;drapery;mantle;pall","chair","car;auto;automobile;machine;motorcar","water","painting;picture","sofa;couch;lounge","shelf","house","sea","mirror","rug;carpet;carpeting","field","armchair","seat","fence;fencing","desk","rock;stone","wardrobe;closet;press","lamp","bathtub;bathing;tub;bath;tub","railing;rail","cushion","base;pedestal;stand","box","column;pillar","signboard;sign","chest;of;drawers;chest;bureau;dresser","counter","sand","sink","skyscraper","fireplace;hearth;open;fireplace","refrigerator;icebox","grandstand;covered;stand","path","stairs;steps","runway","case;display;case;showcase;vitrine","pool;table;billiard;table;snooker;table","pillow","screen;door;screen","stairway;staircase","river","bridge;span","bookcase","blind;screen","coffee;table;cocktail;table","toilet;can;commode;crapper;pot;potty;stool;throne","flower","book","hill","bench","countertop","stove;kitchen;stove;range;kitchen;range;cooking;stove","palm;palm;tree","kitchen;island","computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system","swivel;chair","boat","bar","arcade;machine","hovel;hut;hutch;shack;shanty","bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle","towel","light;light;source","truck;motortruck","tower","chandelier;pendant;pendent","awning;sunshade;sunblind","streetlight;street;lamp","booth;cubicle;stall;kiosk","television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box","airplane;aeroplane;plane","dirt;track","apparel;wearing;apparel;dress;clothes","pole","land;ground;soil","bannister;banister;balustrade;balusters;handrail","escalator;moving;staircase;moving;stairway","ottoman;pouf;pouffe;puff;hassock","bottle","buffet;counter;sideboard","poster;posting;placard;notice;bill;card","stage","van","ship","fountain","conveyer;belt;conveyor;belt;conveyer;conveyor;transporter","canopy","washer;automatic;washer;washing;machine","plaything;toy","swimming;pool;swimming;bath;natatorium","stool","barrel;cask","basket;handbasket","waterfall;falls","tent;collapsible;shelter","bag","minibike;motorbike","cradle","oven","ball","food;solid;food","step;stair","tank;storage;tank","trade;name;brand;name;brand;marque","microwave;microwave;oven","pot;flowerpot","animal;animate;being;beast;brute;creature;fauna","bicycle;bike;wheel;cycle","lake","dishwasher;dish;washer;dishwashing;machine","screen;silver;screen;projection;screen","blanket;cover","sculpture","hood;exhaust;hood","sconce","vase","traffic;light;traffic;signal;stoplight","tray","ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin","fan","pier;wharf;wharfage;dock","crt;screen","plate","monitor;monitoring;device","bulletin;board;notice;board","shower","radiator","glass;drinking;glass","clock","flag"]
        files = glob.glob(r'/Users/divyachandana/Documents/NJIT/work/summertasks/jun1-jun5/atlanta/*.jpg')
        print("Total Files",len(files))
        columns = ['filename','class','total_pixel','individual_pixel','ratio','timestamp']

        # ---------- drive code -----
        with open('semantic_results_atlanta.csv','a') as csvfile:
            csvwriter = csv.writer(csvfile,lineterminator='\n')
            # csvwriter.writerow(columns)
            # i=0
            for f in files:
                file_check_query = "select count(*) from {} where filename like '%{}%'".format('semantic_results_atlanta', os.path.basename(f))
                # print(file_check_query)
                # i += 1
                # print(i)
                count = dbms.get_count_result(file_check_query)
                # print(count)
                if count > 0: continue
                # print('resuming',f)
                try:
                    img = image.imread(f)
                    img = image.resize_short(img, 1024)
                    #         img = image.resize_short(img, 100)
                    # print("filename: ", f)
                    #         ctx = mx.gpu(0)
                    img = test_transform(img, ctx)
                    #         print("img: ", img)
                    output = model.predict(img)
                    #         print("output: ", output)
                    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
                    #         print("predict: ", predict)
                    mask = get_color_pallete(predict, 'ade20k')
                    # predict.save('predict.png')
                    # mmask = mpimg.imread('output.png')
                    predict = predict.astype(numpy.uint8)
                    convert_single_array = numpy.array(predict)
                    unique_numbers = numpy.unique(convert_single_array)
                    # print(unique_numbers)
                    new_basename = os.path.basename(f).replace(".jpg", ".png")
                    new_name = os.path.join('output/', new_basename)
                    mask.save(new_name)
                    # color_img = image.imread(new_name)
                    # colors, counts = numpy.unique(color_img.reshape(-1, 3), return_counts=True, axis=0)
                    total_pixel = numpy.sum(predict)
                    d_file = drive.CreateFile({'parents': [{'id': file_id}], 'title': os.path.basename(new_name)})
                    d_file.SetContentFile(new_name)
                    d_file.Upload()
                    # print('Created file %s with mimeType %s' % (d_file['title'], d_file['mimeType']))
                    combile_all_csv_data = []
                    combine_sql_srting_format = []
                    for i in unique_numbers:
                        individual_count = numpy.sum(predict == i)
                        # print(individual_count)
                        csv_data = []
                        csv_data.append(os.path.basename(f))
                        csv_data.append(classes[i])
                        csv_data.append(total_pixel)
                        csv_data.append(individual_count)
                        csv_data.append(round((individual_count/total_pixel),6))
                        time_stamp = datetime.now()
                        csv_data.append(time_stamp)
                        # csv_data = [filename,predict,colors,counts,total_pixel]
                        # print(csv_data)
                        combile_all_csv_data.append(csv_data)
                        sql_srting  = ["NULL" if val == None else "'"+str(val)+"'" for val in csv_data]
                        sql_srting_format = ",".join([str(val) for val in sql_srting])
                        combine_sql_srting_format.append(sql_srting_format)
                    csvwriter.writerows(combile_all_csv_data)
                    dbms.insertmany_sqlite3('semantic_results_atlanta',','.join(columns),combine_sql_srting_format)

                    os.remove(new_name)
                    # if idx % 10 == 0:
                    #     print("Processed: ", idx)
                except Exception as e:
                    print("Error in :", '       ' + f, e)
                    continue
        print('Finished')
        end = timer()
        print('Processing time: %.1f' % (end - start))



if __name__ == '__main__':
    main()





