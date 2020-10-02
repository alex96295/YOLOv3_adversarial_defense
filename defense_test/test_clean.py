import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from torchvision import transforms
from PIL import Image, ImageDraw
import json

cfg = "./cfg/yolov3.cfg"
#weights = "./weights/yolov3.weights"
weights_pt = "./weights/yolov3_ultralytics_q.pt"
names_path = './data/coco.names'
imgdir = "../inria/INRIAPerson/Test/pos/"
#imgdir = "./inria_for_ablation/images/"
#clean_label_dir = "./inria_for_ablation/labels_yv3/"
clean_label_dir = './test_results_mytrial/clean/yolov3q-labels/'

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
    img_size = 416
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = 'cpu'
    #device = 'cuda:' if torch.cuda.is_available() else 'cpu'
    #print(torch.cuda.device_count())

    # Initialize model
    yolov3_model = Darknet(cfg, img_size)
    #load_darknet_weights(yolov3_model, weights)
    yolov3_model.load_state_dict(torch.load(weights_pt, map_location=device))#['model'])
    #yolov3_model = torch.jit.load(weights_pt)
    
    # Eval mode
    #yolov3_model.cuda().eval()
    yolov3_model.to(device).eval()

    # Get names and colors
    names = load_classes(names_path)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    n=0
    clean_results = []

    for imgfile in os.listdir(imgdir):

        print('\nIMAGE #' + str(n) + ': ' + os.path.splitext(imgfile)[0])
        print(n + 1)
        n += 1

        print(imgfile)

        txt_name = os.path.splitext(imgfile)[0] + '.txt'
        clean_label_path = os.path.join(clean_label_dir, txt_name)
        img_path = os.path.join(imgdir, imgfile)
        txt_clean_path = os.path.join(clean_label_path)

        # Run inference
        #_ = yolov3_model(torch.zeros((1, 3, img_size, img_size), device=device)) if device.type != 'cpu' else None  # run once

        img = Image.open(img_path).convert('RGB')

        # h_orig = img.size[1]  # for opencv it is (height, width) and .shape, while for PIL it is (width, height) and .size
        # w_orig = img.size[0]

        img_tens = transforms.ToTensor()(img)
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
        print(det.size())
        print(det)

        # Rescale boxes from img_size to im0 size
        box_widths_half = (det[:,2] - det[:,0])/2
        box_heights_half = (det[:,3] - det[:,1])/2
        det[:,0] = det[:,2] - box_widths_half # box x centers
        det[:,1] = det[:, 3] - box_heights_half # box y center
        det[:,2] = box_widths_half*2
        det[:,3] = box_heights_half * 2
        det[:,:4] = det[:,:4]/416
        print(det) # x_c, y_c, w, h, conf, cls_id; normalized to 416

        textfile = open(txt_clean_path, 'w+')
        for box in det:
            cls_id = box[5]
            if cls_id == 0:
                textfile.write(f'{cls_id} {box[0]} {box[1]} {box[2]} {box[3]}\n')

                
                clean_results.append({'image_id': os.path.splitext(imgfile)[0], 'bbox': [(box[0].item() - box[2].item() / 2),
                                                                                         (box[1].item() - box[3].item() / 2),
                                                                                         box[2].item(),
                                                                                         box[3].item()],
                                      'score': box[4].item(),
                                      'category_id': 1}) 


        # plot_boxes(img, det, os.path.join(clean_label_dir, 'clean/', 'after_detection', imgfile), names)

    
    with open('./json_files/clean_results_yv3q.json', 'w') as fp:
        json.dump(clean_results, fp) 


if __name__ == '__main__':
    with torch.no_grad():
        detect()
