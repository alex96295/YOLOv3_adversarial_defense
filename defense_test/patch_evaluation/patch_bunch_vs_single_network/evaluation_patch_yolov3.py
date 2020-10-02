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
patch_yolov2_best = bb.io.load('det_coco', './json_files/yolov2_patches/proper_patched_obj.json', class_label_map={0: 'person'})
patch_yolov3_best = bb.io.load('det_coco', './json_files/yolov3_patches/proper_patched_obj.json', class_label_map={0: 'person'})
patch_yolov4_best = bb.io.load('det_coco', './json_files/yolov4_patches/proper_patched_obj.json', class_label_map={0: 'person'})

patch_ssd_mbntv1 = bb.io.load('det_coco', './json_files/ssd_mbntv1_patches/proper_patched_obj_cls.json', class_label_map={0: 'person'})
patch_ssdlite_mbntv2 = bb.io.load('det_coco', './json_files/ssdlite_mbntv2_patches/proper_patched_obj_cls.json', class_label_map={0: 'person'})
patch_ensemble_yv2_ssd_mbntv1_ssdlitembntv2_best = bb.io.load('det_coco', './json_files/ensemble_yv2_ssdmbntv1_ssdlitembntv2_patches/proper_patched_obj_cls.json', class_label_map={0: 'person'})
patch_ensemble_yv3_ssd_mbntv1_ssdlitembntv2_best = bb.io.load('det_coco', './json_files/ensemble_yv3_ssdmbntv1_ssdlitembntv2_patches/proper_patched_obj_cls.json', class_label_map={0: 'person'})
patch_ensemble_yv2_yv3_ssdlitembntv2_best = bb.io.load('det_coco', './json_files/ensemble_yv2_yv3_ssdlitembntv2_patches/proper_patched_obj_cls.json', class_label_map={0: 'person'})


# patch_results_random_noise = bb.io.load('det_coco', './json_files/yolov3/random_results.json', class_label_map={0: 'person'})
# patch_results_random_image = bb.io.load('det_coco', './json_files/yolov3/random_image_results.json', class_label_map={0: 'person'})

# In[8]:


plt.figure()

clean = bb.stat.pr(clean_results, annotations, threshold=0.5)

yolov2_p = bb.stat.pr(patch_yolov2_best, annotations, threshold=0.5)
yolov3_p = bb.stat.pr(patch_yolov3_best, annotations, threshold=0.5)
yolov4_p = bb.stat.pr(patch_yolov4_best, annotations, threshold=0.5)
ssd_mbntv1_p = bb.stat.pr(patch_ssd_mbntv1, annotations, threshold=0.5)
ssdlite_mbntv2_p = bb.stat.pr(patch_ssdlite_mbntv2, annotations, threshold=0.5)

ensemble1_p = bb.stat.pr(patch_ensemble_yv2_ssd_mbntv1_ssdlitembntv2_best, annotations, threshold=0.5)
ensemble2_p = bb.stat.pr(patch_ensemble_yv3_ssd_mbntv1_ssdlitembntv2_best, annotations, threshold=0.5)
ensemble3_p = bb.stat.pr(patch_ensemble_yv2_yv3_ssdlitembntv2_best, annotations, threshold=0.5)

#random_noise = bb.stat.pr(patch_results_random_noise, annotations, threshold=0.5)
#random_image = bb.stat.pr(patch_results_random_image, annotations, threshold=0.5)

#ap = bbb.ap(teddy[0], teddy[1])
#plt.plot(teddy[1], teddy[0], label=f'Teddy: mAP: {round(ap*100, 2)}%')

''' Plot stuffs '''

plt.plot([0, 1.05], [0, 1.05], '--', color='gray')

ap = bb.stat.ap(clean)
plt.plot(clean['recall'], clean['precision'], label=f'CLEAN: AP: {round(ap*100, 2)}%') #, RECALL: {round(clean["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(yolov2_p)
plt.plot(yolov2_p['recall'], yolov2_p['precision'], label=f'YOLOV2_P: AP: {round(ap*100, 2)}%') #, RECALL: {round(yolov2_p["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(yolov3_p)
plt.plot(yolov3_p['recall'], yolov3_p['precision'], label=f'YOLOV3_P: AP: {round(ap*100, 2)}%') #, RECALL: {round(yolov3_p["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(yolov4_p)
plt.plot(yolov4_p['recall'], yolov4_p['precision'], label=f'YOLOV4_P: AP: {round(ap*100, 2)}%') #, RECALL: {round(yolov3_p["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(ssd_mbntv1_p)
plt.plot(ssd_mbntv1_p['recall'], ssd_mbntv1_p['precision'], label=f'MBNTV1_SSD_P: AP: {round(ap*100, 2)}%') #, RECALL: {round(ssd_mbntv1_p["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(ssdlite_mbntv2_p)
plt.plot(ssdlite_mbntv2_p['recall'], ssdlite_mbntv2_p['precision'], label=f'MBNTV2_SSDLITE_P: AP: {round(ap*100, 2)}%') #, RECALL: {round(ssdlite_mbntv2_p["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(ensemble1_p)
plt.plot(ensemble1_p['recall'], ensemble1_p['precision'], label=f'ENSEMBLE1_P: AP: {round(ap*100, 2)}%') #, RECALL: {round(ensemble1_p["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(ensemble2_p)
plt.plot(ensemble2_p['recall'], ensemble2_p['precision'], label=f'ENSEMBLE2_P: AP: {round(ap*100, 2)}%') #, RECALL: {round(ensemble2_p["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(ensemble3_p)
plt.plot(ensemble3_p['recall'], ensemble3_p['precision'], label=f'ENSEMBLE3_P: AP: {round(ap*100, 2)}%') #, RECALL: {round(ensemble3_p["recall"].iloc[-1]*100, 2)}%')

# ap = bb.stat.ap(random_noise)
# plt.plot(random_noise['recall'], random_noise['precision'], label=f'NOISE: AP: {round(ap*100, 2)}%, RECALL: {round(random_noise["recall"].iloc[-1]*100, 2)}%')

# ap = bb.stat.ap(random_image)
# plt.plot(random_image['recall'], random_image['precision'], label=f'RAND_IMG: AP: {round(ap*100, 2)}%, RECALL: {round(random_image["recall"].iloc[-1]*100, 2)}%')

plt.gcf().suptitle('YOLOv2, dataset = INRIA, all_patches')
plt.gca().set_ylabel('Precision')
plt.gca().set_xlabel('Recall')
plt.gca().set_xlim([0, 1.05])
plt.gca().set_ylim([0, 1.05])
plt.gca().legend(loc=4)
plt.savefig("./pr-curves/patch_bunch_vs_single_network/yolov2_net_patch_bunch.eps")
plt.savefig("./pr-curves/patch_bunch_vs_single_network/yolov2_net_patch_bunch.png")

