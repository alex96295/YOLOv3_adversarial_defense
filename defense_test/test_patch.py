import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from torchvision import transforms
from PIL import Image, ImageDraw
import json
from load_data import *

cfg = "./cfg/yolov3.cfg"
weights = "./weights/yolov3.weights"
#weights_pt = "./weights/yolov3_ultralytics.pt"
names_path = './data/coco.names'
imgdir = "../inria/INRIAPerson/Test/pos/"
clean_label_dir = "./test_results_mytrial/"
patchfile = "../master_thesis/saved_patches_mytrial/ensemble/net_ensemble_yv2_yv3_yv4_obj_max_mean.jpg"

def generate_patch(type, patch_size):

        if type == 'gray':
                adv_patch_cpu = torch.full((3, patch_size, patch_size), 0.5)
        elif type == 'random':
                adv_patch_cpu = torch.rand((3, patch_size, patch_size))

        return adv_patch_cpu


def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width # 416 here
    height = img.height # 416 here
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

patch_size = 300

patch_img = Image.open(patchfile).convert('RGB')
# plt.imshow(patch_img)
# plt.show()
patch_img = patch_img.resize((patch_size,patch_size))
adv_patch = transforms.ToTensor()(patch_img) # already in range 0,1
# print(adv_patch.type())

#adv_patch = generate_patch('gray', patch_size)

def detect():
    img_size = 416

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.device_count())

    # Initialize model
    yolov3_model = Darknet(cfg, img_size)
    load_darknet_weights(yolov3_model, weights)
    #yolov3_model.load_state_dict(torch.load(weights_pt, map_location=device)['model'])

    # Eval mode
    yolov3_model.to(device).eval()

    # Get names and colors
    names = load_classes(names_path)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    n=0
    patch_results = []

    for imgfile in os.listdir(imgdir):

        print('\nIMAGE #' + str(n) + ': ' + os.path.splitext(imgfile)[0])
        print(n + 1)
        n += 1

        # print(imgfile)

        txt_name = os.path.splitext(imgfile)[0] + '.txt'
        patch_label_name = os.path.splitext(imgfile)[0] + '_p.txt'
        clean_label_path = os.path.join(clean_label_dir, 'clean/', 'yolov3-labels/', txt_name)
        img_path = os.path.join(imgdir, imgfile)
        txt_clean_path = os.path.join(clean_label_path)
        patch_label_path = os.path.join(clean_label_dir, 'ens_avg', 'yolov3-labels/', patch_label_name)

        print('Start reading generated label file used for patch application')
        # read this label file back as a tensor
        textfile = open(txt_clean_path, 'r')
        if os.path.getsize(txt_clean_path):  # check to see if label file contains data.
            label = np.loadtxt(textfile)
            # print(label.shape)
        else:
            label = np.ones([5])

        if np.ndim(label) == 1:
            # label = label.unsqueeze(0)
            label = np.expand_dims(label, 0)

        label = torch.from_numpy(label).float()
        print('label file used for patch application read correctly')

        img = Image.open(img_path).convert('RGB')

        h_orig = img.size[1]  # for opencv it is (height, width) and .shape, while for PIL it is (width, height) and .size
        w_orig = img.size[0]

        print('Start image preprocessing')

        image_p, label = pad_and_scale(img, label, common_size=img_size) # common size = 416

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
        p_img_orig = remove_pad(w_orig, h_orig, p_img)
        # plt.imshow(p_img_orig)
        # plt.show()

        print('End image preprocessing')

        # Run inference

        img_tens = transforms.ToTensor()(p_img_orig)
        if img_tens.ndimension() == 3:
            img_tens = img_tens.unsqueeze(0) # add fake batch size
        #print(img)

        # img = img.numpy()
        # print(img)
        # print(img.shape)
        # img = np.transpose(img, (1, 2, 0)) # to h_orig x w_orig x 3
        # print(img.shape)

        img_reshaped = F.interpolate(img_tens, (img_size, img_size))
        img_reshaped = img_reshaped.cuda()

        # # Padded resize
        # img_reshaped = letterbox(img, new_shape=img_size)[0] # to 416 x 416 x 3
        # print(img_reshaped/255.0)
        #
        # # Convert
        # img_reshaped = img_reshaped[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img_reshaped = np.ascontiguousarray(img_reshaped)
        # print(img_reshaped.shape)
        #
        # img_reshaped = torch.from_numpy(img_reshaped)
        # if img_reshaped.ndimension() == 3:
        #     img_reshaped = img_reshaped.unsqueeze(0) # add batch size


        # Inference

        pred = yolov3_model.forward(img_reshaped)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.6, multi_label=False, classes=False, agnostic=False)
        det = pred[0]
        # Process detections
        # webcam = 0
        #print(len(pred))

        #for i, det in enumerate(pred):  # detections per image
        #print('i: ' + str(i))
        # print(det.size())
        # print(det)
        
        textfile = open(patch_label_path, 'w+')
        if det is not None:

            # Rescale boxes from img_size to im0 size
            box_widths_half = (det[:,2] - det[:,0])/2
            box_heights_half = (det[:,3] - det[:,1])/2
            det[:,0] = det[:,2] - box_widths_half # box x centers
            det[:,1] = det[:, 3] - box_heights_half # box y center
            det[:,2] = box_widths_half*2
            det[:,3] = box_heights_half * 2
            det[:,:4] = det[:,:4]/416
            # print(det) # x_c, y_c, w, h, conf, cls_id; normalized to 416
    
            for box in det:
                cls_id = box[5]
                if cls_id == 0:
                    textfile.write(f'{cls_id} {box[0]} {box[1]} {box[2]} {box[3]}\n')
    
                    patch_results.append({'image_id': os.path.splitext(imgfile)[0], 'bbox': [(box[0].item() - box[2].item() / 2),
                                                                                             (box[1].item() - box[3].item() / 2),
                                                                                             box[2].item(),
                                                                                             box[3].item()],
                                          'score': box[4].item(),
                                          'category_id': 1})
        else:
            textfile.write(f'\n')
            #patch_results.append({'image_id': , 'bbox': [(box[0].item() - box[2].item() / 2),
            #                                                                                  (box[1].item() - box[3].item() / 2),
            #                                                                                  box[2].item(),
            #                                                                                  box[3].item()],
            #                               'score': box[4].item(),
            #                               'category_id': 1})
            


        #plot_boxes(img, det, os.path.join(clean_label_dir, 'clean/', 'after_detection', imgfile), names)

    with open('./json_files/ens_avg.json', 'w') as fp:
        json.dump(patch_results, fp)


if __name__ == '__main__':
    with torch.no_grad():
        detect()
