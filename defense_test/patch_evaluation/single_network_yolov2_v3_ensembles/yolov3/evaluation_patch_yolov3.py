#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import brambox as bb
import brambox.io.parser.box as bbb
import matplotlib.pyplot as plt
from IPython.display import display


# In[5]:


# todo NB ANNOTATIONS COINCIDES WITH CLEAN DETECTIONS! THEY ARE NOT THE GROUND-TRUTH OF INRIA DATASET, SO CLEAN RESULTS ARE 100% ASSUMED AS CORRECT
annotations = bb.io.load('anno_darknet', '../inria/INRIAPerson/Test/labels/', class_label_map={0: 'person'})

clean_results = bb.io.load('det_coco', './json_files/clean_results.json', class_label_map={0: 'person'})
patch_results_obj = bb.io.load('det_coco', './json_files/yolov3_patches/proper_patched_obj.json', class_label_map={0: 'person'})
patch_results_cw = bb.io.load('det_coco', './json_files/yolov3_patches/proper_patched_cw.json', class_label_map={0: 'person'})
#patch_results_cls = bb.io.load('det_coco', './json_files/yolov3_patches/patch_cls_results.json', class_label_map={0: 'person'})
patch_results_cls_obj = bb.io.load('det_coco', './json_files/yolov3_patches/proper_patched_obj_cls.json', class_label_map={0: 'person'})
patch_results_random_noise = bb.io.load('det_coco', './json_files/patch_results_random.json', class_label_map={0: 'person'})
#patch_results_random_image = bb.io.load('det_coco', './json_files/yolov3_patches/random_image_results.json', class_label_map={0: 'person'})

# In[8]:


plt.figure()

clean = bb.stat.pr(clean_results, annotations, threshold=0.5)
obj_only = bb.stat.pr(patch_results_obj, annotations, threshold=0.5)
cw = bb.stat.pr(patch_results_cw, annotations, threshold=0.5)
#cls_only = bb.stat.pr(patch_results_cls, annotations, threshold=0.5)
obj_cls = bb.stat.pr(patch_results_cls_obj, annotations, threshold=0.5)
random_noise = bb.stat.pr(patch_results_random_noise, annotations, threshold=0.5)
#random_image = bb.stat.pr(patch_results_random_image, annotations, threshold=0.5)

#ap = bbb.ap(teddy[0], teddy[1])
#plt.plot(teddy[1], teddy[0], label=f'Teddy: mAP: {round(ap*100, 2)}%')

''' Plot stuffs '''

plt.plot([0, 1.05], [0, 1.05], '--', color='gray')

ap = bb.stat.ap(clean)
plt.plot(clean['recall'], clean['precision'], label=f'CLEAN: AP: {round(ap*100, 2)}%') #, RECALL: {round(clean["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_only)
plt.plot(obj_only['recall'], obj_only['precision'], label=f'OBJ: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_only["recall"].iloc[-1]*100, 2)}%')

#ap = bb.stat.ap(cls_only)
#plt.plot(cls_only['recall'], cls_only['precision'], label=f'CLS: AP: {round(ap*100, 2)}%, RECALL: {round(cls_only["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(cw)
plt.plot(cw['recall'], cw['precision'], label=f'CW: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_only["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_cls)
plt.plot(obj_cls['recall'], obj_cls['precision'], label=f'OBJ-CLS: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_cls["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(random_noise)
plt.plot(random_noise['recall'], random_noise['precision'], label=f'NOISE: AP: {round(ap*100, 2)}%, RECALL: {round(random_noise["recall"].iloc[-1]*100, 2)}%')

#ap = bb.stat.ap(random_image)
#plt.plot(random_image['recall'], random_image['precision'], label=f'RAND_IMG: AP: {round(ap*100, 2)}%, RECALL: {round(random_image["recall"].iloc[-1]*100, 2)}%')

plt.gcf().suptitle('YOLOv3, dataset:INRIA')
plt.gca().set_ylabel('Precision')
plt.gca().set_xlabel('Recall')
plt.gca().set_xlim([0, 1.05])
plt.gca().set_ylim([0, 1.05])
plt.gca().legend(loc=4)
plt.savefig('./pr-curves/single_patches_vs_single_network/yolov3_patch/yolov3_patch.eps')
plt.savefig('./pr-curves/single_patches_vs_single_network/yolov3_patch/yolov3_patch.png')


