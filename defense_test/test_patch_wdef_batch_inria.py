import torch.nn.functional as F
import fnmatch
import argparse
import pickle
from sys import platform
import pandas as pd

from models_wdef import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from torchvision import transforms
from PIL import Image, ImageDraw
import json
import gt_labels_no_loop_export_rows
from torch.utils.data import Dataset
import time
import json
from load_data import *

from mpl_toolkits.mplot3d import Axes3D

cfg = "./cfg/yolov3_6chan.cfg"
weights_f1_orig = "./weights/yolov3_ultralytics.pt"
weights_6ch_f1 = "./weights/best_wdef_column_1_100_correct.pt"
weights_6ch_rows = "./weights/best_wdef_rows_05_100_correct.pt"
names_path = './data/coco.names'

imgdir = "./inria_for_ablation/images/"
clean_label_dir = "./inria_for_ablation/labels_yv3/"

savedir = './def_test_img_folder_inria/rows05_batch100/rows_factor05/patch/'

gt_dir = os.path.join(savedir, 'pr_curves_tot/patch/gt_labels/')

def pad_and_scale(img, lab, common_size):  # this method for taking a non-square image and make it square by filling the difference in w and h with gray

    w, h = img.size
    if w == h:
        padded_img = img
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            padded_img.paste(img, (int(padding), 0))
            lab[:, [1]] = (lab[:, [1]] * w + padding) / h
            lab[:, [3]] = (lab[:, [3]] * w / h)
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            padded_img.paste(img, (0, int(padding)))
            lab[:, [2]] = (lab[:, [2]] * h + padding) / w
            lab[:, [4]] = (lab[:, [4]] * h / w)
    resize = transforms.Resize((common_size, common_size))  # make a square image of dim 416 x 416
    padded_img = resize(padded_img)  # choose here
    return padded_img, lab

def remove_pad(w_orig, h_orig, in_img):

        w = w_orig
        h = h_orig

        img = transforms.ToPILImage('RGB')(in_img)

        dim_to_pad = 1 if w < h else 2

        if dim_to_pad == 1:
            padding = (h - w) / 2
            #padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            #padded_img.paste(img, (int(padding), 0))
            image = Image.Image.resize(img, (h, h))
            image = Image.Image.crop(image, (int(padding), 0, int(padding) + w, h))

        else:
            padding = (w - h) / 2
            # padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            # padded_img.paste(img, (0, int(padding)))
            image = Image.Image.resize(img, (w, w))
            image = Image.Image.crop(image, (0, int(padding), w, int(padding) + h))

        return image

class image_set(Dataset):

    def __init__(self, img_dir, lab_dir, imgsize, adv_patch):
        # imgsize = 416 from yolo
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.adv_patch = adv_patch
        # self.batch_dim_list = batch_dim_list
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        # self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        # self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')

        image = Image.open(img_path).convert('RGB')

        h_orig = image.size[1]  # for opencv it is (height, width), command .shape, while for PIL it is (width, height), command .size
        w_orig = image.size[0]
        # self.batch_dim_list.append((h_orig, w_orig))

        if os.path.getsize(lab_path):  # check to see if label file contains data.
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([6])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        '''
        if label.size()[0] != 1:
            print(label[:, 4])
            conf_max = torch.max(label[:, 4])
            label = label[label[:, 4] == conf_max] '''

        print('original label size:' + str(label.size()))

        label_big = torch.ones((20,6))*(-1)
        for p in range(label.size()[0]):
            label_big[p,0] = label[p,0]
            label_big[p,1] = label[p,1]
            label_big[p,2] = label[p,2]
            label_big[p,3] = label[p,3]
            label_big[p,4] = label[p,4]
            label_big[p,5] = label[p,5]

        print('label with maxlab size: ' +str(label_big.size()))

        # label_patch = torch.ones((label.size()[0], 5))
        cls_id = label_big[:, 5]
        cls_id = cls_id.unsqueeze(1)
        bbox = label_big[:, :4]
        print('cls_id size: ' +str(cls_id.size()))
        print('bbox size: ' +str(bbox.size()))

        # label_patch[:, cls_id] = cls_id
        # label_patch[:, :4] = bbox

        label_patch = torch.cat((cls_id, bbox), 1)
        print('final label size: ' + str(label_patch.size()))

        ''' APPLY PATCH '''
        # put patch digitally if requested, otherwise load patched image

        print('Start image preprocessing')

        image_p, label_patch = pad_and_scale(image, label_patch, common_size=self.imgsize)  # common size = 416

        # convert image back to torch tensor
        image_tens = transforms.ToTensor()(image_p)

        # add fake batch size, fake because it has size = 1, so it's a single image (i.e you don't really need)
        img_fake_batch = torch.unsqueeze(image_tens, 0)
        lab_fake_batch = torch.unsqueeze(label_patch, 0)

        adv_batch_t = PatchTransformer()(self.adv_patch, lab_fake_batch, self.imgsize, do_rotate=True, rand_loc=False)

        # adv_batch_im = transforms.ToPILImage('RGB')(adv_batch_t[0][0])
        # plt.imshow(adv_batch_im)
        # plt.show()

        p_img_batch = PatchApplier()(img_fake_batch, adv_batch_t)

        p_img = torch.squeeze(p_img_batch, 0)

        # come back to original dimensions
        img = remove_pad(w_orig, h_orig, p_img)
        #plt.imshow(img)
        #plt.show()

        print('End image preprocessing')

        img_tens = transforms.ToTensor()(img)
        if img_tens.ndimension() == 3:
            img_tens = img_tens.unsqueeze(0)  # add fake batch size

        # resize image to feed the network
        img_reshaped = F.interpolate(img_tens, (img_size, img_size))

        return img_reshaped, label

# # functions to do random ablation, seed is truly random here
# def random_mask_batch_one_sample_rows(batch, block_size, device, reuse_noise = False):
#     batch = batch.permute(0,2,3,1)
#     out_c1 = torch.zeros(batch.shape).to(device)
#     out_c2 = torch.zeros(batch.shape).to(device)
#
#     if reuse_noise:
#         pos = random.randint(0, batch.shape[1]-1)
#         if (pos+block_size>batch.shape[1]):
#             out_c1[:,pos:,:]=batch[:,pos:,:]
#             out_c2[:,pos:,:]=1.-batch[:,pos:,:]
#
#             out_c1[:,:pos+block_size-batch.shape[1],:]=batch[:,:pos+block_size-batch.shape[1],:]
#             out_c2[:,:pos+block_size-batch.shape[1],:] = 1. - batch[:,:pos+block_size-batch.shape[1],:]
#         else:
#             out_c1[:,pos:pos+block_size,:] = batch[:,pos:pos+block_size,:]
#             out_c2[:,pos:pos+block_size,:] = 1. - batch[:,pos:pos+block_size,:]
#
#     out_c1 = out_c1.permute(0,3,1,2)
#     out_c2 = out_c2.permute(0,3,1,2)
#     out = torch.cat((out_c1,out_c2),1)
#     return out, out_c1
#
# def random_mask_batch_one_sample_cols(batch, block_size, device, reuse_noise = False):
#     batch = batch.permute(0,2,3,1)
#     out_c1 = torch.zeros(batch.shape).to(device)
#     out_c2 = torch.zeros(batch.shape).to(device)
#
#     if reuse_noise:
#         pos = random.randint(0, batch.shape[2]-1)
#         if (pos+block_size>batch.shape[2]):
#             out_c1[:,:,pos:]=batch[:,:,pos:]
#             out_c2[:,:,pos:]=1.-batch[:,:,pos:]
#
#             out_c1[:,:,:pos+block_size-batch.shape[2]]=batch[:,:,:pos+block_size-batch.shape[2]]
#         else:
#             out_c1[:,:,pos:pos+block_size] = batch[:,:,pos:pos+block_size]
#             out_c2[:,:,pos:pos+block_size] = 1. - batch[:,:,pos:pos+block_size]
#
#     out_c1 = out_c1.permute(0,3,1,2)
#     out_c2 = out_c2.permute(0,3,1,2)
#     out = torch.cat((out_c1,out_c2),1)
#     return out, out_c1
#
# def random_mask_batch_one_sample_block(batch, block_size, device, reuse_noise = False):
#     batch = batch.permute(0,2,3,1) #color channel last
#     out_c1 = torch.zeros(batch.shape).to(device)
#     out_c2 = torch.zeros(batch.shape).to(device)
#     if (reuse_noise):
#         xcorner = random.randint(0, batch.shape[1]-1)
#         ycorner = random.randint(0, batch.shape[2]-1)
#         if (xcorner+block_size > batch.shape[1]):
#             if (ycorner+block_size > batch.shape[2]):
#                 out_c1[:,xcorner:,ycorner:] = batch[:,xcorner:,ycorner:]
#                 out_c2[:,xcorner:,ycorner:] = 1. - batch[:,xcorner:,ycorner:]
#
#                 out_c1[:,:xcorner+block_size-batch.shape[1],ycorner:] = batch[:,:xcorner+block_size-batch.shape[1],ycorner:]
#                 out_c2[:,:xcorner+block_size-batch.shape[1],ycorner:] = 1. - batch[:,:xcorner+block_size-batch.shape[1],ycorner:]
#
#                 out_c1[:,xcorner:,:ycorner+block_size-batch.shape[2]] = batch[:,xcorner:,:ycorner+block_size-batch.shape[2]]
#                 out_c2[:,xcorner:,:ycorner+block_size-batch.shape[2]] = 1. - batch[:,xcorner:,:ycorner+block_size-batch.shape[2]]
#
#                 out_c1[:,:xcorner+block_size-batch.shape[1],:ycorner+block_size-batch.shape[2]] = batch[:,:xcorner+block_size-batch.shape[1],:ycorner+block_size-batch.shape[2]]
#                 out_c2[:,:xcorner+block_size-batch.shape[1],:ycorner+block_size-batch.shape[2]] = 1. - batch[:,:xcorner+block_size-batch.shape[1],:ycorner+block_size-batch.shape[2]]
#             else:
#                 out_c1[:,xcorner:,ycorner:ycorner+block_size] = batch[:,xcorner:,ycorner:ycorner+block_size]
#                 out_c2[:,xcorner:,ycorner:ycorner+block_size] = 1. - batch[:,xcorner:,ycorner:ycorner+block_size]
#
#                 out_c1[:,:xcorner+block_size-batch.shape[1],ycorner:ycorner+block_size] = batch[:,:xcorner+block_size-batch.shape[1],ycorner:ycorner+block_size]
#                 out_c2[:,:xcorner+block_size-batch.shape[1],ycorner:ycorner+block_size] = 1. - batch[:,:xcorner+block_size-batch.shape[1],ycorner:ycorner+block_size]
#         else:
#             if  (ycorner+block_size > batch.shape[2]):
#                 out_c1[:,xcorner:xcorner+block_size,ycorner:] = batch[:,xcorner:xcorner+block_size,ycorner:]
#                 out_c2[:,xcorner:xcorner+block_size,ycorner:] = 1. - batch[:,xcorner:xcorner+block_size,ycorner:]
#
#                 out_c1[:,xcorner:xcorner+block_size,:ycorner+block_size-batch.shape[2]] = batch[:,xcorner:xcorner+block_size,:ycorner+block_size-batch.shape[2]]
#                 out_c2[:,xcorner:xcorner+block_size,:ycorner+block_size-batch.shape[2]] = 1. - batch[:,xcorner:xcorner+block_size,:ycorner+block_size-batch.shape[2]]
#             else:
#                 out_c1[:,xcorner:xcorner+block_size,ycorner:ycorner+block_size] = batch[:,xcorner:xcorner+block_size,ycorner:ycorner+block_size]
#                 out_c2[:,xcorner:xcorner+block_size,ycorner:ycorner+block_size] = 1. - batch[:,xcorner:xcorner+block_size,ycorner:ycorner+block_size]
#
#     out_c1 = out_c1.permute(0, 3, 1, 2)
#     out_c2 = out_c2.permute(0, 3, 1, 2)
#     out = torch.cat((out_c1, out_c2), 1)
#     # print(out[14,:,5:10,5:10])
#     return out, out_c1

# functions to do sliding ablation, within a for loop
def sliding_mask_batch_rows(inpt, yolov3_model, block_size, device, fakeabl_res, names):
    batch = inpt.permute(0, 2, 3, 1)  # color channel last

    block_size = int(block_size + 0.5)

    img_h = batch.size()[1]
    img_w = batch.size()[2]

    det_intersect_pos_list = []
    retained_pos_list = []
    det_pos_list = []
    inf_over_thr_pos_list = []

    # number of ablations counter
    abl_cnt = 0

    # for loop for applying sliding ablation
    for pos in range(batch.shape[2]):
        # for pos in range(20):

        det_results = []

        abl_cnt += 1  # count over all sliding positions

        # accumulation lists
        det_list_batch = []
        retained_labels_list_batch = []
        intersect_exist_list2_batch = []
        inf_over_thr_list_batch = []

        print('abl_cnt: ' + str(abl_cnt))

        out_c1 = torch.zeros(batch.shape).to(device)
        out_c2 = torch.zeros(batch.shape).to(device)
        if (pos + block_size > batch.shape[1]):  # along rows (height)
            out_c1[:, pos:, :] = batch[:, pos:, :]
            out_c2[:, pos:, :] = 1. - batch[:, pos:, :]

            out_c1[:, :pos + block_size - batch.shape[1], :] = batch[:, :pos + block_size - batch.shape[1], :]
            out_c2[:, :pos + block_size - batch.shape[1], :] = 1. - batch[:, :pos + block_size - batch.shape[1], :]
        else:
            out_c1[:, pos:pos + block_size, :] = batch[:, pos:pos + block_size, :]
            out_c2[:, pos:pos + block_size, :] = 1. - batch[:, pos:pos + block_size, :]

        out_c1 = out_c1.permute(0, 3, 1, 2)
        out_c2 = out_c2.permute(0, 3, 1, 2)
        out = torch.cat((out_c1, out_c2), 1)  # 6 channels, ablation

        #out = torch.nn.DataParallel(out, device_ids=[0, 1])
        # img_show = transforms.ToPILImage('RGB')(out_c1[0].cpu())
        # plt.imshow(img_show)
        # plt.show
        # img_show.save(savedir + 'rows_abl05_num_' + str(abl_cnt)  + '.png')

        # Inference on single ablated image with 6 channels
        t1 = time.time()
        pred = yolov3_model.forward(out)[0]
        t2 = time.time()
        print('Inference time with batch ' + str(out.size()[0]) + ': ' + str(t2 - t1))

        del out

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.6, multi_label=False, classes=False,
                                   agnostic=False)

        det_single_list = pred
        frame_cnt = 0

        for det in det_single_list:  # for all images

            if det is not None:
                # det = det.cpu()
                # Rescale boxes from img_size to im0 size
                box_widths_half = (det[:, 2] - det[:, 0]) / 2
                box_heights_half = (det[:, 3] - det[:, 1]) / 2
                det[:, 0] = det[:, 2] - box_widths_half  # box x center
                det[:, 1] = det[:, 3] - box_heights_half  # box y center
                det[:, 2] = box_widths_half * 2
                det[:, 3] = box_heights_half * 2
                det[:, :4] = det[:, :4] / 416
                # print(det)  # x_c, y_c, w, h, conf, cls_id; normalized to 416

                # plot_boxes(img_show, det, savedir + 'det/rows_abl05_det_num_' + str(abl_cnt)  + '.png', names)

                ########################################################################################################
                # CHECK IF CONFIDENCE SCORE IS NONZERO
                # condition only for single object in an image right now

                keep_row = []
                # filter for one class and rise conf flag
                for box in det:
                    cls_id = box[5]
                    if cls_id == 0:
                        keep_row.append(box)

                        image_name = 'frame_' + str(frame_cnt)

                        # save json file
                        det_results.append({'image_id': image_name, 'bbox': [(box[0].item() - box[2].item() / 2),
                                                                             (box[1].item() - box[3].item() / 2),
                                                                             box[2].item(),
                                                                             box[3].item()],
                                            'score': box[4].item(),
                                            'category_id': 1})

                if len(keep_row) != 0:
                    det = torch.stack(keep_row, 0)
                    inf_over_thr = 1
                    inf_over_thr_list_batch.append(inf_over_thr)
                else:
                    det = torch.zeros((1, 6))
                    inf_over_thr = 0
                    inf_over_thr_list_batch.append(inf_over_thr)

                det_list_batch.append(det)  # store all the detections

                # print(det_list_batch)
                # print(inf_over_thr_list_batch)
            else:
                det = torch.zeros((1, 6))
                # save json file
                '''
                det_results.append({'image_id': image_name, 'bbox': [(box[0].item() - box[2].item() / 2),
                                                                                (box[1].item() - box[3].item() / 2),
                                                                                box[2].item(),
                                                                                box[3].item()],'score': box[4].item(), 'category_id': 1})'''

                inf_over_thr = 0
                inf_over_thr_list_batch.append(inf_over_thr)
                det_list_batch.append(det)  # store all the detections

            frame_cnt += 1
            with open(
                    './def_test_img_folder/rows05_batch/rows_factor05/pr_curves_tot/patch/json_files_ablr05/det_results_pos_' + str(
                            abl_cnt) + '.json',
                    'w') as fp:
                json.dump(det_results, fp)

        # append for batch
        det_pos_list.append(
            det_list_batch)  # store the det for each image and append in the batch list for the single pos
        inf_over_thr_pos_list.append(inf_over_thr_list_batch)

        ########################################################################################################
        # ABLATE GT FAKEABL BOXES ACCORDING TO ACTUAL ABLATION
        # check for intersection between 'gt' fakeabl boxe(s) and the retained size

        labels_batch = torch.unbind(fakeabl_res, 0)  # fragment the batch

        frame_cnt = 0
        for labels in labels_batch:  # for each image
            # labels = labels.cpu()
            labels = labels[labels[:,5]>-1]

            retained_labels, intersect_exist_list = gt_labels_no_loop_export_rows.labels_ablation(labels, img_h, img_w,
                                                                                                  device, rnd_pos_r=pos,
                                                                                                  retain_rows=block_size)

            gt_name = 'pos_' + str(abl_cnt)
            single_pos_gt = os.path.join(gt_dir, gt_name)
            if os.path.isdir(single_pos_gt) == False:
                os.makedirs(single_pos_gt)
                print('here')

            image_name = 'frame_' + str(frame_cnt)
            textfile_gt_name = os.path.join(single_pos_gt, image_name + '.txt')

            textfile_gt = open(textfile_gt_name, 'w+')
            for box in retained_labels:
                textfile_gt.write(f'{box[5]} {box[0]} {box[1]} {box[2]} {box[3]}\n')

            retained_labels_list_batch.append(retained_labels)
            intersect_exist_list2_batch.append(intersect_exist_list)  # list of 1s and 0s, each associated to 1 box

            # print('intersect_exist_list: ' + str(intersect_exist_list2_batch))
            frame_cnt += 1

        # append for batch
        retained_pos_list.append(retained_labels_list_batch)
        det_intersect_pos_list.append(intersect_exist_list2_batch)
        ########################################################################################################

    # return det_list, retained_labels_list, intersect_exist_list2, inf_over_thr_list, abl_cnt
    # del pred, det_single_list

    return det_pos_list, retained_pos_list, det_intersect_pos_list, inf_over_thr_pos_list, abl_cnt

def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 6 and class_names:
            cls_conf = box[4]
            cls_id = box[5]
            print('%s: %f' % (class_names[int(cls_id)], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[int(cls_id)], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img

def detect():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
                

    img_size = 416

    ablation = 1

    # Initialize model
    yolov3_model = Darknet(cfg, img_size)
    # load weights in .pt format
    # yolov3_model.load_state_dict(torch.load(weights_6ch_f1, map_location=device)['model'])
    #yolov3_model.load_state_dict(torch.load(weights_6ch_rows, map_location=device)['model'])
    yolov3_model.load_state_dict(torch.load(weights_6ch_rows, map_location=device)['model'])
    
    # Eval mode
    
    yolov3_model = torch.nn.DataParallel(yolov3_model)
    yolov3_model.to(device).eval()
    
    # Get names and colors
    names = load_classes(names_path)

    n = 0

    numb_imgs = len(os.listdir(imgdir))
    print(numb_imgs)

    ''' LOAD PATCH '''
    patchfile = "./patches/yolov3_obj.jpg"
    patch_size = 300

    # load patch
    patch_img = Image.open(patchfile).convert('RGB')
    # plt.imshow(patch_img)
    # plt.show()
    patch_img = patch_img.resize((patch_size, patch_size))
    adv_patch = transforms.ToTensor()(patch_img)  # already in range 0,1

    # load dataloader
    dataloader = torch.utils.data.DataLoader(image_set(imgdir, clean_label_dir, img_size, adv_patch),
                                             batch_size=numb_imgs,
                                             shuffle=False,
                                             num_workers=10)

    # print(dataloader)

    # loop over the images you want to test
    for i_batch, (image, gt_label) in enumerate(dataloader):

        print('\nIMAGE #' + str(n))
        # print(n + 1)
        n += 1

        print(image.size())
        print(gt_label.size())

        h_res = image.size()[2]  # batch x chan x height x width
        w_res = image.size()[3]

        num_pos = h_res

        ############################################################################################################################################

        inria = 1

        if ablation == 1:

            # do inteference by applying ablation on the input, sliding over all the 'covering' possibilities

            # ablation parameters
            retain_factor = 0.5
            retain_size_rows = retain_factor * h_res

            det_pos_list, retained_pos_list, det_intersect_pos_list, inf_over_thr_pos_list, abl_cnt = sliding_mask_batch_rows(
                image, yolov3_model, retain_size_rows,
                device, gt_label, names)

            if inria == 1:
                
                # lists of
                print('det_list: ' + str(len(det_pos_list)))
                print('retain_lab: ' + str(len(retained_pos_list)))
                print('inters_exist: ' + str(len(det_intersect_pos_list)))
                print('inf_over_thr: ' + str(len(inf_over_thr_pos_list)))
                print('abl_cnt: ' + str(abl_cnt))

                print('det_list: ' + str(len(det_pos_list[0])))
                print('retain_lab: ' + str(len(retained_pos_list[0])))
                print('inters_exist: ' + str(len(det_intersect_pos_list[0])))
                print('inf_over_thr: ' + str(len(inf_over_thr_pos_list[0])))
                print('abl_cnt: ' + str(abl_cnt))

                # check that lists have all the same length == num. ablations
                assert len(det_pos_list) == len(retained_pos_list) == len(det_intersect_pos_list) == len(
                    inf_over_thr_pos_list) == abl_cnt == num_pos

                '''
                det_list = xc, yc, w, h, conf, cls_id from inference --> "what the network sees"
                retained_labels_list = xc, yc, w, h, conf, cls_id from fake_abl according to abl position --> "what the network should see"
                intersect_exist_list = list of 1s or 0s according to the existence of intersection @ ablation state n
                inf_over_thr_list = list of 1s or 0s according to the existance of the box
                abl_cnt = number of ablations '''

                true_neg_cnt = np.zeros((num_pos * numb_imgs, 3))
                false_neg_cnt = np.zeros((num_pos * numb_imgs, 3))
                false_pos_cnt = np.zeros((num_pos * numb_imgs, 3))
                true_pos_cnt = np.zeros((num_pos * numb_imgs, 3))

                # relevant_cnt = np.ones((num_pos*numb_imgs,3)) *(-10)

                point_cnt = 0

                tp_hist = 0
                fn_hist = 0
                tn_hist = 0
                fp_hist = 0
                hist_list = []

                for i, (gt_pos, inference_pos) in enumerate(zip(det_intersect_pos_list, inf_over_thr_pos_list)):
                    # print('i: ' + str(i))
                    for j, (gt_im, inference_im) in enumerate(zip(gt_pos, inference_pos)):

                        # print('j: ' + str(j))
                        if inference_im == 0 and gt_im[0] == 0:
                            true_neg_cnt[point_cnt][0] = i  # pos
                            true_neg_cnt[point_cnt][1] = j  # img
                            true_neg_cnt[point_cnt][2] = 0.001
                            tn_hist += 1

                        elif inference_im == 0 and gt_im[0] == 1:
                            false_neg_cnt[point_cnt][0] = i
                            false_neg_cnt[point_cnt][1] = j
                            false_neg_cnt[point_cnt][2] = -1
                            fn_hist += 1

                        elif inference_im == 1 and gt_im[0] == 0:
                            false_pos_cnt[point_cnt][0] = i
                            false_pos_cnt[point_cnt][1] = j
                            false_pos_cnt[point_cnt][2] = -2
                            fp_hist += 1
                        elif inference_im == 1 and gt_im[0] == 1:
                            true_pos_cnt[point_cnt][0] = i
                            true_pos_cnt[point_cnt][1] = j
                            true_pos_cnt[point_cnt][2] = 1
                            tp_hist += 1

                        point_cnt += 1

                    hist_list.append((tp_hist, fn_hist, tn_hist, fp_hist))
                    tp_hist = 0
                    fn_hist = 0
                    tn_hist = 0
                    fp_hist = 0

                print(hist_list)

                hist_vec = np.zeros((num_pos, 3))
                for i, tuple in zip(range(num_pos), hist_list):
                    print(i)
                    print(tuple)
                    hist_vec[i][0] = tuple[0]
                    hist_vec[i][1] = tuple[1]
                    hist_vec[i][2] = tuple[2]

                #############################################################################################################
                ''' PLOT tp, fn, fp BAR PLOT'''

                df = pd.DataFrame(hist_vec, columns=['True Positives', 'False Negatives', 'True Negatives'], )
                colors = ['green', 'red', 'blue']

                plt.figure()
                plot = df.plot.bar(rot=0, color=colors)  # , edgecolor = 'white', width = 0.1)

                ax = plt.gca()
                plt.title('Detection evaluation, bar plot\nDATASET: INRIA')
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.tick_params(axis='x', which='both', length=0)

                fig = plot.get_figure()
                ax.axes.set_xlabel('Ablation position')
                plt.ylabel('Dataset images count')
                fig.savefig(os.path.join(savedir, "tp_fn_tn_barplot.png"))

                ##################################################################################################
                ''' PLOT tp, fn, fp BAR PLOT STACKED'''

                plt.figure()
                plot = df.plot.bar(rot=0, stacked=True, color=colors)  # , edgecolor='white',width=0.1)

                ax = plt.gca()
                plt.title('Detection evaluation, bar plot\nDATASET: INRIA')
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.tick_params(axis='x', which='both', length=0)

                fig = plot.get_figure()
                ax.axes.set_xlabel('Ablation position')
                plt.ylabel('Dataset images count')
                fig.savefig(os.path.join(savedir, "tp_fn_tn_barplot_stacked.png"))

                ############################################################################################################
                ''' PLOT tp, fn, fp SCATTER NOT ALIGNED '''

                # filter
                true_pos_cnt = true_pos_cnt[true_pos_cnt[:, 2] > 0]
                true_neg_cnt = true_neg_cnt[true_neg_cnt[:, 2] > 0]
                false_pos_cnt = false_pos_cnt[false_pos_cnt[:, 2] < 0]
                false_neg_cnt = false_neg_cnt[false_neg_cnt[:, 2] < 0]

                # plot relevant elements separated, i.e. true pos and false neg
                plt.figure()
                ax = plt.axes(projection='3d')
                # Data for three-dimensional scattered points

                pos_tp = true_pos_cnt[:, 0]
                imgs_tp = true_pos_cnt[:, 1]
                tp = true_pos_cnt[:, 2]

                pos_fn = false_neg_cnt[:, 0]
                imgs_fn = false_neg_cnt[:, 1]
                fn = false_neg_cnt[:, 2]

                pos_tn = true_neg_cnt[:, 0]
                imgs_tn = true_neg_cnt[:, 1]
                tn = true_neg_cnt[:, 2]

                ax.scatter3D(pos_tp, imgs_tp, tp, label='True positives', color='green', s=2, edgecolor='black',
                             linewidth=0.15)
                ax.scatter3D(pos_fn, imgs_fn, fn + 1, label='False negatives', color='red', s=2, edgecolor='black',
                             linewidth=0.15)
                ax.scatter3D(pos_tn, imgs_tn, tn - 1.001, label='True negatives', color='blue', s=2, edgecolor='black',
                             linewidth=0.15)
                ax.legend()
                ax.set_title('Detection evaluation, scatter plot\nDATASET: INRIA')
                ax.set_xlabel('Ablation position')
                ax.set_ylabel('Frames')
                ax.set_zlabel('True pos/ False neg/ True neg')

                pickle.dump(ax, open(os.path.join(savedir, 'tp_fn_tn_sep.fig.pickle'), 'wb'))
                plt.savefig(os.path.join(savedir, "tp_fn_tn_sep.png"))
                # plt.show()

                ############################################################################################################
                ''' PLOT tp, fn, fp SCATTER ALIGNED'''

                plt.figure()
                ax = plt.axes(projection='3d')

                ax.scatter3D(pos_tp, imgs_tp, tp - 1, label='True positives', color='green', s=2, edgecolor='black',
                             linewidth=0.15)
                ax.scatter3D(pos_fn, imgs_fn, fn + 1, label='False negatives', color='red', s=2, edgecolor='black',
                             linewidth=0.15)
                ax.scatter3D(pos_tn, imgs_tn, tn - 0.001, label='True negatives', color='blue', s=2, edgecolor='black',
                             linewidth=0.15)
                ax.legend()
                ax.set_title('Detection evaluation, scatter plot\nDATASET: INRIA')
                ax.set_xlabel('Ablation position')
                ax.set_ylabel('Frames')
                ax.set_zlabel('True pos/ False neg/ True neg')

                pickle.dump(ax, open(os.path.join(savedir, 'tp_fn_tn_sep_sameplane.fig.pickle'), 'wb'))
                plt.savefig(os.path.join(savedir, "tp_fn_tn_sep_sameplane.png"))

                ################################################################################################################################

                ''' available lists:
                - num_pos lists, each containing num_imgs lists. Each image_list contains what you need for that image, at that position.
                - so you should come back to the original situation where you had all detections for one image 'i' relative to each position
                - once you have this structure, compute conf_diff and iou for each image with external for loop
                - to print in 3d graph, with pos, imgs, conf_diff/iou ---> 
    
                '''

                point_cnt = 0
                # define array for conf_diff and iou as before
                conf_diff_3d = np.zeros((num_pos * numb_imgs, 3))
                iou_3d = np.zeros((num_pos * numb_imgs, 3))

                for i, (det_pos, retained_pos, gt_flag_pos, inference_flag_pos) in enumerate(
                        zip(det_pos_list, retained_pos_list,
                            det_intersect_pos_list, inf_over_thr_pos_list)):
                    # print('i: ' + str(i))

                    for j, (det_im, gt_im, gt_flag_im, inference_flag_im) in enumerate(zip(det_pos, retained_pos,
                                                                                           gt_flag_pos,
                                                                                           inference_flag_pos)):

                        # print('j: ' + str(j))

                        # true positives
                        if gt_flag_im[0] == 1 and inference_flag_im == 1:

                            gt_tmp = gt_im[0].squeeze()

                            gt_tmp = gt_tmp.cpu().numpy()
                            inf_det_tmp = det_im.cpu().numpy()

                            # compute iou between boxes and % distance between conf scores

                            # xcycwh to x1y1x2y2
                            inf_xl = (inf_det_tmp[:, 0] - inf_det_tmp[:, 2] / 2) * w_res
                            inf_xr = (inf_det_tmp[:, 0] + inf_det_tmp[:, 2] / 2) * w_res
                            inf_yt = (inf_det_tmp[:, 1] - inf_det_tmp[:, 3] / 2) * h_res
                            inf_yb = (inf_det_tmp[:, 1] + inf_det_tmp[:, 3] / 2) * h_res

                            gt_xl = (gt_tmp[0] - gt_tmp[2] / 2) * w_res
                            gt_xr = (gt_tmp[0] + gt_tmp[2] / 2) * w_res
                            gt_yt = (gt_tmp[1] - gt_tmp[3] / 2) * h_res
                            gt_yb = (gt_tmp[1] + gt_tmp[3] / 2) * h_res

                            # intersection
                            x_left_int = np.maximum(inf_xl, gt_xl)
                            x_right_int = np.minimum(inf_xr, gt_xr)
                            y_top_int = np.maximum(inf_yt, gt_yt)
                            y_bottom_int = np.minimum(inf_yb, gt_yb)

                            area_inf = abs(inf_xr - inf_xl) * abs(inf_yt - inf_yb)
                            area_inf = np.max(area_inf)

                            area_gt = abs(gt_xr - gt_xl) * abs(gt_yb - gt_yt)

                            # are intersection
                            area_int = abs(x_left_int - x_right_int) * abs(y_top_int - y_bottom_int)
                            # print('Area int:' + str(area_int))
                            area_int = np.max(area_int)
                            # print(area_int)

                            # area union
                            area_union = abs(area_inf + area_gt - area_int)

                            # print('Area union: ' + str(area_union))
                            iou = area_int / area_union

                            # compute % difference in confidence
                            inf_conf = np.max(inf_det_tmp[:, 4])
                            conf_difference = (1 - inf_conf / gt_tmp[4]) * 100

                            conf_diff_3d[point_cnt][0] = i
                            conf_diff_3d[point_cnt][1] = j
                            conf_diff_3d[point_cnt][2] = conf_difference

                            iou_3d[point_cnt][0] = i
                            iou_3d[point_cnt][1] = j
                            iou_3d[point_cnt][2] = iou

                            point_cnt += 1

                        elif gt_flag_im[0] == 1 and inference_flag_im == 0:  # false negative
                            point_cnt += 1

                        elif gt_flag_im[0] == 0 and inference_flag_im == 0:  # true_neg
                            point_cnt += 1

                        else:  # false pos (never here)
                            point_cnt += 1

                #############################################################################################################
                ''' PLOT CONFIDENCE DIFFERENCE'''

                plt.figure()

                ax = plt.axes(projection='3d')
                ax.set_title('Confidence difference, scatter plot\nDATASET: INRIA')
                ax.set_xlabel('Ablation position')
                ax.set_ylabel('Frames')
                ax.set_zlabel('Conf diff')
                ax.scatter3D(conf_diff_3d[:, 0], conf_diff_3d[:, 1], conf_diff_3d[:, 2], s=2, label='True positives',
                             color='green', edgecolor='black', linewidth=0.15)
                ax.scatter3D(pos_fn, imgs_fn, (fn + 1) + 100, label='False negatives', color='red', s=2, edgecolor='black',
                             linewidth=0.15)
                ax.scatter3D(pos_tn, imgs_tn, tn - 0.001, label='True negatives', color='blue', s=2, edgecolor='black',
                             linewidth=0.15)
                ax.legend()
                pickle.dump(ax, open(os.path.join(savedir, 'conf_vs_abl_all.fig.pickle'), 'wb'))

                '''
                # join points with lines
                ni = 0
                for position in range(num_pos):
                    #print(position)
                    #print(conf_diff_3d[ni:ni+numb_imgs,0:1])
                    #print(conf_diff_3d[ni:ni+numb_imgs,1:2])
                    #print(conf_diff_3d[ni:ni+numb_imgs,2:])
                    ax.plot(conf_diff_3d[ni:ni+numb_imgs+1,0:1].flatten(), conf_diff_3d[ni:ni+numb_imgs+1,1:2].flatten(), conf_diff_3d[ni:ni+numb_imgs+1,2:].flatten(), color='red', linewidth=0.25)
                    ni += numb_imgs
                '''

                plt.savefig(os.path.join(savedir, "conf_vs_abl_all.png"))
                # plt.show()

                #############################################################################################################
                ''' PLOT IOU'''

                plt.figure()
                # conf difference plot vs ablation position
                ax = plt.axes(projection='3d')
                ax.set_title('IoU, scatter plot\nDATASET: INRIA')
                ax.set_xlabel('Ablation position')
                ax.set_ylabel('Frames')
                ax.set_zlabel('iou')
                ax.scatter3D(iou_3d[:, 0], iou_3d[:, 1], iou_3d[:, 2], label='True positives', s=2, color='green',
                             edgecolor='black', linewidth=0.15)
                ax.scatter3D(pos_fn, imgs_fn, fn + 1, label='False negatives', color='red', s=2, edgecolor='black',
                             linewidth=0.15)
                ax.scatter3D(pos_tn, imgs_tn, tn - 0.001, label='True negatives', color='blue', s=2, edgecolor='black',
                             linewidth=0.15)
                ax.legend()
                pickle.dump(ax, open(os.path.join(savedir, 'iou_vs_abl_all.fig.pickle'), 'wb'))

                '''
                # join points with lines
                ni = 0
                for position in range(num_pos):
                    #print(position)
                    #print(iou_diff_3d[ni:ni+numb_imgs,0:1])
                    #print(iou_diff_3d[ni:ni+numb_imgs,1:2])
                    #print(iou_3d[ni:ni+numb_imgs,2:])
                    ax.plot(iou_3d[ni:ni+numb_imgs+1,0:1].flatten(), iou_3d[ni:ni+numb_imgs+1,1:2].flatten(), iou_3d[ni:ni+numb_imgs+1,2:].flatten(), color='red', linewidth = 0.25)
                    ni += numb_imgs
                '''

                plt.savefig(os.path.join(savedir, "iou_vs_abl_all.png"))
                # plt.show()


if __name__ == '__main__':
    with torch.no_grad():
        detect()
