import argparse
from sys import platform

from models_wdef import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from torchvision import transforms
from PIL import Image, ImageDraw
import json
from load_data import *
import gt_labels_no_loop_export_rows 

cfg = "./cfg/yolov3_6chan.cfg"
weights = "./weights/yolov3.weights" #pjreddie original weights

# weights trained by ultralitics
weights_f1_orig = "./weights/yolov3_ultralytics.pt"
weights1 = "./weights/best_wdef_column_1_30.pt"
weights_rows05 = "./weights/best_wdef_rows_05_30.pt"
names_path = './data/coco.names'
imgdir = "./def_test_img_folder/clean_imgs/"
clean_label_dir = "./def_test_img_folder/nodef_clean_det_labels/"
savedir = './def_test_img_folder/rows05/clean_factor1/'

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


# functions to do random ablation, seed is truly random here
def random_mask_batch_one_sample_rows(batch, block_size, device, reuse_noise = False):
    batch = batch.permute(0,2,3,1)
    out_c1 = torch.zeros(batch.shape).to(device)
    out_c2 = torch.zeros(batch.shape).to(device)

    if reuse_noise:
        pos = random.randint(0, batch.shape[1]-1)
        if (pos+block_size>batch.shape[1]):
            out_c1[:,pos:,:]=batch[:,pos:,:]
            out_c2[:,pos:,:]=1.-batch[:,pos:,:]

            out_c1[:,:pos+block_size-batch.shape[1],:]=batch[:,:pos+block_size-batch.shape[1],:]
            out_c2[:,:pos+block_size-batch.shape[1],:] = 1. - batch[:,:pos+block_size-batch.shape[1],:]
        else:
            out_c1[:,pos:pos+block_size,:] = batch[:,pos:pos+block_size,:]
            out_c2[:,pos:pos+block_size,:] = 1. - batch[:,pos:pos+block_size,:]

    out_c1 = out_c1.permute(0,3,1,2)
    out_c2 = out_c2.permute(0,3,1,2)
    out = torch.cat((out_c1,out_c2),1)
    return out, out_c1
    
def random_mask_batch_one_sample_cols(batch, block_size, device, reuse_noise = False):
    batch = batch.permute(0,2,3,1)
    out_c1 = torch.zeros(batch.shape).to(device)
    out_c2 = torch.zeros(batch.shape).to(device)

    if reuse_noise:
        pos = random.randint(0, batch.shape[2]-1)
        if (pos+block_size>batch.shape[2]):
            out_c1[:,:,pos:]=batch[:,:,pos:]
            out_c2[:,:,pos:]=1.-batch[:,:,pos:]

            out_c1[:,:,:pos+block_size-batch.shape[2]]=batch[:,:,:pos+block_size-batch.shape[2]]
        else:
            out_c1[:,:,pos:pos+block_size] = batch[:,:,pos:pos+block_size]
            out_c2[:,:,pos:pos+block_size] = 1. - batch[:,:,pos:pos+block_size]

    out_c1 = out_c1.permute(0,3,1,2)
    out_c2 = out_c2.permute(0,3,1,2)
    out = torch.cat((out_c1,out_c2),1)
    return out, out_c1

def random_mask_batch_one_sample_block(batch, block_size, device, reuse_noise = False):
    batch = batch.permute(0,2,3,1) #color channel last
    out_c1 = torch.zeros(batch.shape).to(device)
    out_c2 = torch.zeros(batch.shape).to(device)
    if (reuse_noise):
        xcorner = random.randint(0, batch.shape[1]-1)
        ycorner = random.randint(0, batch.shape[2]-1)
        if (xcorner+block_size > batch.shape[1]):
            if (ycorner+block_size > batch.shape[2]):
                out_c1[:,xcorner:,ycorner:] = batch[:,xcorner:,ycorner:]
                out_c2[:,xcorner:,ycorner:] = 1. - batch[:,xcorner:,ycorner:]

                out_c1[:,:xcorner+block_size-batch.shape[1],ycorner:] = batch[:,:xcorner+block_size-batch.shape[1],ycorner:]
                out_c2[:,:xcorner+block_size-batch.shape[1],ycorner:] = 1. - batch[:,:xcorner+block_size-batch.shape[1],ycorner:]

                out_c1[:,xcorner:,:ycorner+block_size-batch.shape[2]] = batch[:,xcorner:,:ycorner+block_size-batch.shape[2]]
                out_c2[:,xcorner:,:ycorner+block_size-batch.shape[2]] = 1. - batch[:,xcorner:,:ycorner+block_size-batch.shape[2]]

                out_c1[:,:xcorner+block_size-batch.shape[1],:ycorner+block_size-batch.shape[2]] = batch[:,:xcorner+block_size-batch.shape[1],:ycorner+block_size-batch.shape[2]]
                out_c2[:,:xcorner+block_size-batch.shape[1],:ycorner+block_size-batch.shape[2]] = 1. - batch[:,:xcorner+block_size-batch.shape[1],:ycorner+block_size-batch.shape[2]]
            else:
                out_c1[:,xcorner:,ycorner:ycorner+block_size] = batch[:,xcorner:,ycorner:ycorner+block_size]
                out_c2[:,xcorner:,ycorner:ycorner+block_size] = 1. - batch[:,xcorner:,ycorner:ycorner+block_size]

                out_c1[:,:xcorner+block_size-batch.shape[1],ycorner:ycorner+block_size] = batch[:,:xcorner+block_size-batch.shape[1],ycorner:ycorner+block_size]
                out_c2[:,:xcorner+block_size-batch.shape[1],ycorner:ycorner+block_size] = 1. - batch[:,:xcorner+block_size-batch.shape[1],ycorner:ycorner+block_size]
        else:
            if  (ycorner+block_size > batch.shape[2]):
                out_c1[:,xcorner:xcorner+block_size,ycorner:] = batch[:,xcorner:xcorner+block_size,ycorner:]
                out_c2[:,xcorner:xcorner+block_size,ycorner:] = 1. - batch[:,xcorner:xcorner+block_size,ycorner:]

                out_c1[:,xcorner:xcorner+block_size,:ycorner+block_size-batch.shape[2]] = batch[:,xcorner:xcorner+block_size,:ycorner+block_size-batch.shape[2]]
                out_c2[:,xcorner:xcorner+block_size,:ycorner+block_size-batch.shape[2]] = 1. - batch[:,xcorner:xcorner+block_size,:ycorner+block_size-batch.shape[2]]
            else:
                out_c1[:,xcorner:xcorner+block_size,ycorner:ycorner+block_size] = batch[:,xcorner:xcorner+block_size,ycorner:ycorner+block_size]
                out_c2[:,xcorner:xcorner+block_size,ycorner:ycorner+block_size] = 1. - batch[:,xcorner:xcorner+block_size,ycorner:ycorner+block_size]

    out_c1 = out_c1.permute(0, 3, 1, 2)
    out_c2 = out_c2.permute(0, 3, 1, 2)
    out = torch.cat((out_c1, out_c2), 1)
    # print(out[14,:,5:10,5:10])
    return out, out_c1

# functions to do sliding ablation, within a for loop
def sliding_mask_batch_rows(inpt, yolov3_model, block_size, device, fakeabl_res, names):
    #predictions = torch.zeros(inpt.size(0), num_classes).type(torch.int).cuda()
    batch = inpt.permute(0,2,3,1) # color channel last

    block_size = int(block_size+0.5)

    img_h = batch.size()[1]
    img_w = batch.size()[2]

    # accumulation lists
    det_list = []
    retained_labels_list = []
    intersect_exist_list2 = []
    inf_over_thr_list = []
    
    # number of ablations counter
    abl_cnt = 0

    # for loop for applying sliding ablation
    for pos in range(batch.shape[2]):
    #for pos in range(20):
        abl_cnt += 1 # count over all sliding positions

        print('abl_cnt: ' + str(abl_cnt))

        out_c1 = torch.zeros(batch.shape).to(device)
        out_c2 = torch.zeros(batch.shape).to(device)
        if  (pos+block_size > batch.shape[1]): # along rows (height)
            out_c1[:,pos:,:] = batch[:,pos:,:]
            out_c2[:,pos:,:] = 1. - batch[:,pos:,:]

            out_c1[:,:pos+block_size-batch.shape[1],:] = batch[:,:pos+block_size-batch.shape[1],:]
            out_c2[:,:pos+block_size-batch.shape[1],:] = 1. - batch[:,:pos+block_size-batch.shape[1],:]
        else:
            out_c1[:,pos:pos+block_size,:] = batch[:,pos:pos+block_size,:]
            out_c2[:,pos:pos+block_size,:] = 1. - batch[:,pos:pos+block_size,:]

        out_c1 = out_c1.permute(0,3,1,2)
        out_c2 = out_c2.permute(0,3,1,2)

        out = torch.cat((out_c1,out_c2), 1) # 6 channels, ablation

        img_show = transforms.ToPILImage('RGB')(out_c1[0].cpu())
        #plt.imshow(img_show)
        #plt.show
        #img_show.save(savedir + 'rows_abl05_num_' + str(abl_cnt)  + '.png')

        # Inference on single ablated image with 6 channels
        pred = yolov3_model.forward(out)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.6, multi_label=False, classes=False,
                                      agnostic=False)
        det = pred[0]

        if det is not None:
            # Rescale boxes from img_size to im0 size
            box_widths_half = (det[:, 2] - det[:, 0]) / 2
            box_heights_half = (det[:, 3] - det[:, 1]) / 2
            det[:, 0] = det[:, 2] - box_widths_half  # box x center
            det[:, 1] = det[:, 3] - box_heights_half  # box y center
            det[:, 2] = box_widths_half * 2
            det[:, 3] = box_heights_half * 2
            det[:, :4] = det[:, :4] / 416
            print(det)  # x_c, y_c, w, h, conf, cls_id; normalized to 416

            plot_boxes(img_show, det, savedir + 'det_patch/rows_abl05_det_num_' + str(abl_cnt)  + '.png', names)

        ########################################################################################################
        # CHECK IF CONFIDENCE SCORE IS NONZERO
        # condition only for single object in an image right now

            keep_row = []
            # filter for one class and rise conf flag
            for box in det:
                cls_id = box[5]
                if cls_id == 0:
                    keep_row.append(box)

            if len(keep_row) != 0:
                det = torch.stack(keep_row, 0)
                inf_over_thr = 1
                inf_over_thr_list.append(inf_over_thr)
            else:
                det = torch.zeros((1,6))
                inf_over_thr = 0
                inf_over_thr_list.append(inf_over_thr)

            det_list.append(det) # store all the detections

            print(det_list)
            print(inf_over_thr_list)
        else:
            det = torch.zeros((1,6))
            inf_over_thr = 0
            inf_over_thr_list.append(inf_over_thr)
            det_list.append(det) # store all the detections

        ########################################################################################################
        # ABLATE GT FAKEABL BOXES ACCORDING TO ACTUAL ABLATION
        # check for intersection between 'gt' fakeabl boxe(s) and the retained size

        # add batch for labels
        #fakeabl_res = torch.unsqueeze(fakeabl_res, 0) #batch, rows, column

        retained_labels, intersect_exist_list = gt_labels_no_loop_export_rows.labels_ablation(fakeabl_res, img_h, img_w, device, rnd_pos_r=pos, retain_rows=block_size)
        retained_labels_list.append(retained_labels)
        intersect_exist_list2.append(intersect_exist_list) # list of 1s and 0s, each associated to 1 box

        print('intersect_exist_list: ' + str(intersect_exist_list2))
        ########################################################################################################


    return det_list, retained_labels_list, intersect_exist_list2, inf_over_thr_list, abl_cnt


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

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # initialize counters
    true_pos_cnt = 0
    false_pos_cnt = 0
    true_neg_cnt = 0
    false_neg_cnt = 0

    # flags
    conf_over_thr = 0
    intersect_exist = 0

    img_size = 416
    #device = 'cuda:' if torch.cuda.is_available() else 'cpu'
    #print(torch.cuda.device_count())

    ablation = 1


    if ablation == 0:
        
        #from models import *
        
        # Initialize model
        yolov3_model = Darknet(cfg, img_size)
        # load weights in .pt format
        yolov3_model.load_state_dict(torch.load(weights_f1_orig, map_location=device)['model'])
        #load_darknet_weights(yolov3_model, weights)

        # Eval mode
        yolov3_model.to(device).eval()

    else:
        #from models_wdef import *
        
        # Initialize model
        yolov3_model = Darknet(cfg, img_size)
        # load weights in .pt format
        yolov3_model.load_state_dict(torch.load(weights1, map_location=device)['model'])
        # load_darknet_weights(yolov3_model, weights)

        # Eval mode
        yolov3_model.to(device).eval()

    # Get names and colors
    names = load_classes(names_path)
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    n=0

    # loop over the images you want to test
    for imgfile in os.listdir(imgdir):

        print('\nIMAGE #' + str(n) + ': ' + os.path.splitext(imgfile)[0])
        print(n + 1)
        n += 1

        print(imgfile)

        txt_name = os.path.splitext(imgfile)[0] + '.txt'
        clean_label_path = os.path.join(clean_label_dir, txt_name)
        img_path = os.path.join(imgdir, imgfile)

        img = Image.open(img_path).convert('RGB')

        h_orig = img.size[1]  # for opencv it is (height, width), command .shape, while for PIL it is (width, height), command .size
        w_orig = img.size[0]


        # put patch digitally if requested, otherwise load patched image
        apply_patch_digitally = 1
        if apply_patch_digitally:

            patchfile = "./patches/yolov3_obj.jpg"
            patch_size = 300

            # load patch
            patch_img = Image.open(patchfile).convert('RGB')
            # plt.imshow(patch_img)
            # plt.show()
            patch_img = patch_img.resize((patch_size,patch_size))
            adv_patch = transforms.ToTensor()(patch_img) # already in range 0,1

            print('Read predictions done with fake ablation (clean)')
            textfile = open(clean_label_path, 'r')

            if os.path.getsize(clean_label_path):  # check to see if label file contains data.
                fakeabl_label = np.loadtxt(textfile)
                # print(label.shape)
            else:
                fakeabl_label = np.zeros([6])

            if np.ndim(fakeabl_label) == 1:
                fakeabl_label = np.expand_dims(fakeabl_label, 0)

            cls_id = fakeabl_label[:,5]
            conf = fakeabl_label[:,4]
            bbox = fakeabl_label[:,:4]

            fakeabl_label = np.ones((fakeabl_label.shape[0],5))
            fakeabl_label[:,0] = cls_id
            fakeabl_label[:,1:5] = bbox

            print(fakeabl_label)
                       

            fakeabl_label = torch.from_numpy(fakeabl_label).float() # format xc, yc, w, h, conf, cls_id

            print('Fake_ablation_gt for patch application: ' + str(fakeabl_label))  


        print('Start image preprocessing')

        image_p, label = pad_and_scale(img, fakeabl_label, common_size=img_size) # common size = 416

        # convert image back to torch tensor
        image_tens = transforms.ToTensor()(image_p)

        # add fake batch size, fake because it has size = 1, so it's a single image (i.e you don't really need)
        img_fake_batch = torch.unsqueeze(image_tens, 0)
        lab_fake_batch = torch.unsqueeze(label, 0)

        adv_batch_t = PatchTransformer()(adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)

        # adv_batch_im = transforms.ToPILImage('RGB')(adv_batch_t[0][0])
        # plt.imshow(adv_batch_im)
        # plt.show()

        p_img_batch = PatchApplier()(img_fake_batch, adv_batch_t)

        p_img = torch.squeeze(p_img_batch, 0)

        # come back to original dimensions
        img = remove_pad(w_orig, h_orig, p_img)
        plt.imshow(img)
        plt.show()

        print('End image preprocessing')

        img_tens = transforms.ToTensor()(img)
        if img_tens.ndimension() == 3:
            img_tens = img_tens.unsqueeze(0) # add fake batch size

        # resize image to feed the network
        img_reshaped = F.interpolate(img_tens, (img_size, img_size))
        img_reshaped = img_reshaped.to(device)
        
        h_res = img_reshaped.size()[2] # batch x chan x height x width 
        w_res = img_reshaped.size()[3]
 
        ############################################################################################################################################

        if ablation == 0:

            # run inference on the non_ablated image to have the 'ground truth' predicitons w/o ablation (retain_factor = 1)
            # therefore just use a retain_factor = 1

            # non-ablation parameters
            retain_factor_full = 1
            #retain_size_col_full = retain_factor_full*w_res
            #retain_size_col_full = int((retain_size_col_full+0.5))
            retain_size_rows_full = retain_factor_full*h_res
            #retain_size_rows_full = int((retain_size_rows_full+0.5))

            img_out_fakeabl, immagine = random_mask_batch_one_sample_rows(img_reshaped, retain_size_rows_full, device, reuse_noise = True) # 6 channels, fake ablation

            # Inference
            pred_fakeabl = yolov3_model.forward(immagine)[0]

            # Apply NMS
            pred_fakeabl = non_max_suppression(pred_fakeabl, conf_thres=0.3, iou_thres=0.6, multi_label=False, classes=False, agnostic=False)
            det_fakeabl = pred_fakeabl[0]

            # Rescale boxes from img_size to im0 size
            box_widths_half = (det_fakeabl[:, 2] - det_fakeabl[:, 0]) / 2
            box_heights_half = (det_fakeabl[:, 3] - det_fakeabl[:, 1]) / 2
            det_fakeabl[:, 0] = det_fakeabl[:, 2] - box_widths_half  # box x centers
            det_fakeabl[:, 1] = det_fakeabl[:, 3] - box_heights_half  # box y center
            det_fakeabl[:, 2] = box_widths_half * 2
            det_fakeabl[:, 3] = box_heights_half * 2
            det_fakeabl[:, :4] = det_fakeabl[:, :4] / 416
            print(det_fakeabl)  # x_c, y_c, w, h, conf, cls_id; normalized to 416


            # fakeabl_res = []
            # for box in det_fakeabl:
            #     cls_id = box[5]
            #     if cls_id == 0: # person
            #         fakeabl_res.append(box) # store only the boxes of people
            #
            # if fakeabl_res is not None:
            #     fakeabl_res = torch.stack(noabl_res, 0)

            # write the label as a txt file
            '''
            textfile = open(clean_label_path, 'w+')
            for box in det_fakeabl:
                cls_id = box[5]
                if cls_id == 0:
                    textfile.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {cls_id}\n') '''


            # plot_boxes(img, det, os.path.join(clean_label_dir, 'clean/', 'after_detection', imgfile), names)

        ############################################################################################################################################

        else: # do ablation

            # to allow some evaluation, load clean text without ablation (% difference, iou...)
            # evaluation is not doable in 'real time', but useful to see how good/bad the ablation is

            print('Read predictions done with fake ablation (clean)')
            textfile = open(clean_label_path, 'r')
            if os.path.getsize(clean_label_path):  # check to see if label file contains data.
                fakeabl_label = np.loadtxt(textfile)
                # print(label.shape)
            else:
                fakeabl_label = np.zeros([6])

            if np.ndim(fakeabl_label) == 1:
                fakeabl_label = np.expand_dims(fakeabl_label, 0)

            fakeabl_label = torch.from_numpy(fakeabl_label).float() # format xc, yc, w, h, conf, cls_id

            print('Fake_ablation_gt: ' + str(fakeabl_label))
    
            # do inteference by applying ablation on the input, sliding over all the 'covering' possibilities

            # ablation parameters
            retain_factor = 0.5
            retain_size_rows = retain_factor*h_res
            retain_size_col = retain_factor*w_res

            det_list, retained_labels_list, intersec_exist_list2, inf_over_thr_list, abl_cnt = sliding_mask_batch_rows(img_reshaped, yolov3_model, retain_size_rows,
                                                                                                                       device, fakeabl_label, names)

            print('det_list: ' + str(len(det_list)))
            print('retain_lab: ' + str(len(retained_labels_list)))
            print('inters_exist: ' + str(len(intersec_exist_list2)))
            print('inf_over_thr: ' + str(len(inf_over_thr_list)))
            print('abl_cnt: ' + str(abl_cnt))
            
            # check that lists have all the same length == num. ablations
            assert len(det_list) == len(retained_labels_list) == len(intersec_exist_list2) == len(inf_over_thr_list) == abl_cnt

            '''
            det_list = xc, yc, w, h, conf, cls_id from inference --> "what the network sees"
            retained_labels_list = xc, yc, w, h, conf, cls_id from fake_abl according to abl position --> "what the network should see"
            intersect_exist_list = list of 1s or 0s according to the existence of intersection @ ablation state n
            inf_over_thr_list = list of 1s or 0s according to the existance of the box
            abl_cnt = number of ablations '''

            # eval
            for i, (gt, inference) in enumerate(zip(intersec_exist_list2, inf_over_thr_list)):
                if inference == 0 and gt[0] == 0:
                    true_neg_cnt += 1
                elif inference == 0 and gt[0] == 1:
                    false_neg_cnt += 1
                elif inference == 1 and gt[0] == 0:
                    false_pos_cnt += 1
                elif inference == 1 and gt[0] == 1:
                    true_pos_cnt += 1

            print('True positive count: ' + str(true_pos_cnt))
            print('True positive rate: ' + str(true_pos_cnt / abl_cnt))

            print('False positive count: ' + str(false_pos_cnt))
            print('False positive rate: ' + str(false_pos_cnt / abl_cnt))

            print('True negative count: ' + str(true_neg_cnt))
            print('True negative rate: ' + str(true_neg_cnt / abl_cnt))

            print('False negative count: ' + str(false_neg_cnt))
            print('False negative rate: ' + str(false_neg_cnt / abl_cnt))

            plt.figure()
            fig1, ax1 = plt.subplots()
            labels = 'True Positives', 'True Negatives', 'False Positives', 'False Negatives'
            sizes = [true_pos_cnt, true_neg_cnt, false_pos_cnt, false_neg_cnt]
            explode = (0.1, 0, 0, 0)
            ax1.set_title('Num. ablations: ' + str(abl_cnt) + ', 1 obj per each image', fontweight="bold")
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%.2f%%', shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.savefig(os.path.join(savedir, "stat_pie_patch.png"))
            # plt.show()

            conf_difference_list_relevant = []
            iou_list_relevant = []

            conf_difference_list_truepos = []
            iou_list_truepos = []

            conf_difference_list_falseneg = []
            iou_list_falseneg = []

            relevant_cnt = 0

            for (inf_det, gt, gt_flag, inf_det_flag) in zip(det_list, retained_labels_list, intersec_exist_list2,
                                                            inf_over_thr_list):

                # true positives
                if gt_flag[0] == 1 and inf_det_flag == 1:

                    relevant_cnt += 1
                    # print('I am here')
                    # print(inf_det.size())
                    # print(gt[0].size())

                    # inf_det_tmp = inf_det.squeeze()
                    gt_tmp = gt[0].squeeze()

                    inf_det_tmp = inf_det.cpu().numpy()
                    gt_tmp = gt_tmp.cpu().numpy()

                    # compute iou between boxes and % distance between conf scores

                    # print(inf_det_tmp.shape)
                    # print(gt_tmp.shape)

                    # xcycwh to x1y1x2y2
                    inf_xl = (inf_det_tmp[:, 0] - inf_det_tmp[:, 2] / 2) * w_orig
                    inf_xr = (inf_det_tmp[:, 0] + inf_det_tmp[:, 2] / 2) * w_orig
                    inf_yt = (inf_det_tmp[:, 1] - inf_det_tmp[:, 3] / 2) * h_orig
                    inf_yb = (inf_det_tmp[:, 1] + inf_det_tmp[:, 3] / 2) * h_orig

                    gt_xl = (gt_tmp[0] - gt_tmp[2] / 2) * w_orig
                    gt_xr = (gt_tmp[0] + gt_tmp[2] / 2) * w_orig
                    gt_yt = (gt_tmp[1] - gt_tmp[3] / 2) * h_orig
                    gt_yb = (gt_tmp[1] + gt_tmp[3] / 2) * h_orig
                    # print(w_orig, h_orig, gt_xl, gt_xr, gt_yt, gt_yb)

                    # intersection
                    x_left_int = np.maximum(inf_xl, gt_xl)
                    x_right_int = np.minimum(inf_xr, gt_xr)
                    y_top_int = np.maximum(inf_yt, gt_yt)
                    y_bottom_int = np.minimum(inf_yb, gt_yb)

                    area_inf = abs(inf_xr - inf_xl) * abs(inf_yt - inf_yb)
                    area_inf = np.max(area_inf)

                    area_gt = abs(gt_xr - gt_xl) * abs(gt_yb - gt_yt)
                    # print('area gt: ' + str(area_gt))

                    # are intersection
                    area_int = abs(x_left_int - x_right_int) * abs(y_top_int - y_bottom_int)
                    # print('Area int:' + str(area_int))
                    area_int = np.max(area_int)
                    # print(area_int)

                    # area union
                    area_union = abs(area_inf + area_gt - area_int)

                    # print('Area union: ' + str(area_union))
                    iou = area_int / area_union
                    iou_list_relevant.append(iou)
                    iou_list_truepos.append(iou)
                    # print(iou_list)

                    # compute % difference in confidence
                    inf_conf = np.max(inf_det_tmp[:, 4])
                    conf_difference = (1 - inf_conf / gt_tmp[4]) * 100

                    conf_difference_list_relevant.append(conf_difference)
                    conf_difference_list_truepos.append(conf_difference)

                elif gt_flag[0] == 1 and inf_det_flag == 0:  # false negative
                    relevant_cnt += 1
                    iou_list_relevant.append(0)
                    conf_difference_list_relevant.append(100)

                    iou_list_falseneg.append(0)
                    conf_difference_list_falseneg.append(100)

            # plot

            # conf difference for true positives
            if len(conf_difference_list_relevant) != 0:
                conf_difference_vec_relevant = np.stack(conf_difference_list_relevant)
            else:
                conf_difference_vec_relevant = np.array([0])

            if len(conf_difference_list_truepos) != 0:
                conf_difference_vec_truepos = np.stack(conf_difference_list_truepos)
            else:
                conf_difference_vec_truepos = np.array([0])

            if len(conf_difference_list_falseneg) != 0:
                conf_difference_vec_falseneg = np.stack(conf_difference_list_falseneg)
            else:
                conf_difference_vec_falseneg = np.array([0])

            # iou distribution for tot true positives
            if len(iou_list_relevant) != 0:
                iou_vec_relevant = np.stack(iou_list_relevant)
            else:
                iou_vec_relevant = np.array([0])

            if len(iou_list_truepos) != 0:
                iou_vec_truepos = np.stack(iou_list_truepos)
            else:
                iou_vec_truepos = np.array([0])

            if len(iou_list_falseneg) != 0:
                iou_vec_falseneg = np.stack(iou_list_falseneg)
            else:
                iou_vec_falseneg = np.array([0])

            # inference detections distribution with ablation
            inf_over_thr_vec = np.stack(inf_over_thr_list)

            # gt distribution with ablation
            list_tmp = []
            for i, inter in enumerate(intersec_exist_list2):
                list_tmp.append(np.stack(inter))  # first stack

            int_vec = np.stack(list_tmp, 0)

            plt.figure()
            # conf_dec statistics true pos + false neg
            plt.hist(conf_difference_vec_truepos, bins=10, edgecolor='white', color='green', stacked=True)
            # plt.hist(conf_difference_vec_falseneg, bins=2, edgecolor='white', color='red', stacked=True)
            plt.xlabel('confidence score difference [%]')
            plt.ylabel('counts')
            plt.title(
                'conf score difference true pos\nDistribution per ablation position\nTot counts: ' + str(true_pos_cnt))
            plt.savefig(os.path.join(savedir, "conf_diff_patch.png"))
            # plt.show()

            plt.figure()
            # iou statistics true pos + false neg
            plt.hist(iou_vec_truepos * 100, bins=10, edgecolor='white', color='green', stacked=True)
            # plt.hist(iou_vec_falseneg*100, bins=2, edgecolor='white', color='red', stacked=True)
            plt.xlabel('iou [%]')
            plt.ylabel('counts')
            plt.title(
                'iou for true pos\nPredicted_abl vs. Predicted_no_abl\nDistribution per ablation position\nTot counts: ' + str(
                    true_pos_cnt))
            plt.savefig(os.path.join(savedir, "iou_patch.png"))
            # plt.show()

            # abl = np.arange(0,29,1)
            relevant_cnt = np.linspace(1, relevant_cnt, relevant_cnt)
            # relevant_cnt = relevant_cnt*np.ones(len(iou_list_relevant))

            plt.figure()
            # iou plot vs ablation position
            plt.plot(relevant_cnt, iou_vec_relevant)
            plt.xlabel('ablation position from top-left corner')
            plt.ylabel('iou: pred_abl vs. pred_no_abl [%]')
            plt.savefig(os.path.join(savedir, "iou_vs_abl_patch.png"))
            # plt.show()

            plt.figure()
            # conf difference plot vs ablation position
            plt.plot(relevant_cnt, conf_difference_vec_relevant)
            plt.xlabel('ablation position from top-left corner')
            plt.ylabel('conf difference: pred_abl vs. pred_no_abl [%]')
            plt.savefig(os.path.join(savedir, "conf_vs_abl_patch.png"))
            # plt.show()

            # save to txtfile
            iou_data_txt = os.path.join(savedir, 'iou_data_patch.txt')
            conf_diff_data_txt = os.path.join(savedir, 'conf_diff_data_patch.txt')

            ioulist = [iou_list_truepos, iou_list_falseneg, iou_list_relevant, relevant_cnt]
            #ioulist.append(iou_list_truepos, iou_list_falseneg, iou_list_relevant, relevant_cnt)

            confdifflist = [conf_difference_list_truepos, conf_difference_list_truepos, conf_difference_list_relevant, relevant_cnt]
            #confdifflist.append(conf_difference_list_truepos, conf_difference_list_truepos, conf_difference_list_relevant, relevant_cnt)

            iou_file = open(iou_data_txt, 'w+')
            conf_diff_file = open(conf_diff_data_txt, 'w+')

            for item in ioulist:
                iou_file.write("%s\n" % item)

            for item in confdifflist:
                conf_diff_file.write("%s\n" % item)

            '''
            # pos_det statistics
            plt.hist(inf_over_thr_vec, bins=10, edgecolor='white', color='red')
            plt.xlabel('positive detections [%]')
            plt.ylabel('counts')
            plt.title('inference 0/1 flags\nDistribution per ablation position\nTot counts: ' + str(abl_cnt))
            plt.savefig("./def_test_img_folder/rows05/pos_det_distr.png")
            plt.show()

            # intersec statistics
            plt.hist(int_vec, bins=10, edgecolor='white', color='red')
            plt.xlabel('intersections [%]')
            plt.ylabel('counts')
            plt.title('clean gt 0/1 flags\nDistribution per ablation position\nTot counts: ' + str(abl_cnt))
            plt.savefig("./def_test_img_folder/rows05/intersec_distr.png")
            plt.show() '''


if __name__ == '__main__':
    with torch.no_grad():
        detect()

