#!/usr/bin/env python
# coding: utf-8

# In[2]:

#get_ipython().run_line_magic('matplotlib', 'inline')
import brambox as bb
import brambox.io.parser.box as bbb
import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import re
import pandas as pd

shared_path = './def_test_img_folder/rows0375_batch100/clean_factor1/pr_curves_tot/clean/'
json_files_pos_path = './def_test_img_folder/rows0375_batch100/clean_factor1/pr_curves_tot/clean/json_files_ablr05/'
gtpath = shared_path + 'gt_labels/'
map_savepath = './def_test_img_folder/map_saved/rows0375_batch100/clean_factor1/'

def group(l, size):
   return [tuple(l[i:i+size]) for i in range(0, len(l), size)]

# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
#     return sorted(data, key=alphanum_key)

plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('PR curves according to position\nNET: yolov3_tr_baseline, ABLATION: rows, RF=0.375, DATASET: video frames')
ax.set_xlabel('Ablation position')
ax.set_ylabel('Recall')
ax.set_zlabel('Precision')

map_list = []

for pos, clean_r05_json_pos in enumerate(os.listdir(json_files_pos_path)):

    #plt.figure()
    
    print('pos: ' +str(pos))
    pos += 1
    # open annotations for pos i
    gtdir = os.path.join(gtpath, 'pos_' + str(pos))
    #print(gtdir)
    
    annotations = bb.io.load('anno_darknet', gtdir, class_label_map={0: 'person'})

    # open json file for pos i
    json_file = os.path.join(json_files_pos_path, clean_r05_json_pos)
    clean_results_rows05 = bb.io.load('det_coco', json_file, class_label_map={0: 'person'})

    # calculate pr for pos i
    clean = bb.stat.pr(clean_results_rows05, annotations, threshold=0.5)

    clean['recall'] = clean['recall'].to_numpy()
    clean['precision'] = clean['precision'].to_numpy()

    # accumulate results for position i
    single_pos = np.ones(len(clean['recall']))*(pos-1)
    #accum_pr_points.append((single_pos, clean['recall'], clean['precision']))

    # compute mAP
    ap = bb.stat.ap(clean)
    map_list.append(ap)

    
    #print(single_pos)
    #print(clean['recall'].to_numpy())
    #print(clean['precision'])

    ''' Plot PR curves for each position in a 3D graph '''

    # conf difference plot vs ablation position
        
    #ax.scatter3D(single_pos, clean['recall'],clean['precision'], color='blue')
    ax.plot(single_pos, clean['recall'],clean['precision'], color='blue', linewidth = 0.25)
    #plt.show()

plt.savefig(os.path.join(shared_path, "pr_yv3b1_ablrow0375.png"))

######################################################################################################à
''' plot map vs pos scatter + line '''
plt.figure()
x = np.linspace(0,416,416)
map_vec = np.array(map_list)
plt.plot(x, map_vec, color='green', linewidth = 0.1)
plt.scatter(x, map_vec, color='green', marker = ".", edgecolor='black', linewidth = 0.1)

plt.title('mAP at varying position\n NET: yolov3_tr_baseline, ABLATION: rows, RF=0.375, DATASET: video frames')
plt.xlabel('ablation position')
plt.ylabel('mAP')
plt.savefig(os.path.join(shared_path, "map_vs_pos.png"))

#######################################################################################################
''' plot map stat histogram '''
plt.figure()
plt.hist(map_vec, color='green', edgecolor="white")
plt.title('mAP at varying position, compound statistics\n NET: yolov3_tr_baseline, ABLATION: rows, RF=0.5, DATASET: video frames')
plt.xlabel('mAP')
plt.ylabel('mAP')
plt.savefig(os.path.join(shared_path, "map_histogram.png"))

#########################################################################################################
''' plot map stat bar '''
df = pd.DataFrame(map_vec, columns=['mAP'])
colors = ['green']

plt.figure()
plot = df.plot.bar(rot=0, color = colors)#, edgecolor = 'white', width = 0.1)

ax = plt.gca()
plt.title('mAP at varying position, bar plot\n NET: yolov3_tr_baseline, ABLATION: rows, RF=0.375, DATASET: video frames')
plt.setp(ax.get_xticklabels(), visible=False)
ax.tick_params(axis='x', which='both', length=0)

fig = plot.get_figure()
ax.axes.set_xlabel('Ablation position')
plt.ylabel('Dataset images count')
fig.savefig(os.path.join(shared_path, "map_bar.png"))

############################################################################################################
''' save map values '''
txt_name = 'map_yv3_b1ablr0375.txt'
txtfile = open(os.path.join(map_savepath, txt_name), 'w+')

for ap in map_list:
    txtfile.write(f'{ap} ')

txtfile.close()

############################################################################################################
''' plot compound bar plot for each ablation: 

- ablation types on x axis
- for each ablation type, a stackd or unstacked bar plot with each attempt (the network trained on, clean reference and maybe the others)
- for the clean, put the baseline, for the patch let's see
'''

compound = 0
do_average_map = 0

txtname_list_30 = ['map_yv3_r025ablr025.txt', 'map_yv3_b1ablr025.txt', 'map_yv3_r0375ablr0375.txt', 'map_yv3_b1ablr0375.txt', 'map_yv3_r05ablr05.txt', 'map_yv3_b1ablr05.txt', 'map_yv3_r075ablr075.txt', 'map_yv3_b1ablr075.txt']
txtname_list_100 = ['map_yv3_r0375ablr0375.txt', 'map_yv3_b1ablr0375.txt', 'map_yv3_r05ablr05.txt', 'map_yv3_b1ablr05.txt', 'map_yv3_r025ablr025.txt', 'map_yv3_b1ablr025.txt']

ablations = ['abl025', 'abl05', 'abl0375', 'abl075']

acc_list = []
avg_list = []

num_cases_100 = len(txtname_list_100)
num_cases_30 = len(txtname_list_30)

if compound == 1:

    if do_average_map == 1:

        # read map from files
        for i, txt_name in enumerate(txtname_list_100):
            print(i)
            textfile = open(os.path.join(map_savepath, txt_name), 'r')
            if os.path.getsize(os.path.join(map_savepath, txt_name)):  # check to see if label file contains data.
                map = np.loadtxt(textfile)
                # print(label.shape)
            else:
                print('No mAP to load')
                break

            if np.ndim(map) == 1:
                map = np.expand_dims(map, 0)
                acc_list.append(map)

        # calculate average map over positions
        for map_pos in acc_list:
            map_pos_avg = np.mean(map_pos)
            avg_list.append(map_pos_avg)

        group(avg_list, 2) # group by 2 ---> (network trained on that ablation, basilen)

        bar_vec = np.zeros((ablations, 2))
        for i, tuple_for_1_state in zip(range(len(ablations)), avg_list):
            print(i)
            print(tuple_for_1_state)
            bar_vec[i][0] = tuple_for_1_state[0]
            bar_vec[i][1] = tuple_for_1_state[1]

        print(bar_vec)

        # plot bar diagram

        df = pd.DataFrame(bar_vec, columns=['Network trained with ablation', 'Clean baseline'])
        colors = ['yellow', 'black']

        plt.figure()
        plot = df.plot.bar(rot=0, color=colors)  # , edgecolor = 'white', width = 0.1)
        ax = plt.gca()
        plt.title('Average mAP on ablation positions, bar plot\nDATASET: video frames')
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis='x', which='both', length=0)

        fig = plot.get_figure()
        ax.axes.set_xlabel('Ablation position')
        plt.ylabel('Average mAP')
        fig.savefig(os.path.join(shared_path, "avg_map_barplot.png"))


