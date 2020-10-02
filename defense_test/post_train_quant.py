import argparse
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from utils.datasets import *
import torch.quantization

yolov3weights = "./weights/yolov3_ultralytics.pt"
savedir = "./weights/"

def calibrate(model, data_loader, neval_batches):
    model.eval()

    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model.forward(image)[0]

            cnt += 1

            if cnt >= neval_batches:
                 return print('Calibration ended over 10 batches, 30 samples each')

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options

def quant():
    #num_eval_batches = 10
    #eval_batch_size = 30

    train_batch_size = 30
    num_calibration_batches = 10

    # total batches for calibration = 300
    # total batches for evaluation = 300

    # prepare for COCO dataset
    # Dataset
    cfg = opt.cfg
    data = opt.data
    # Configure run
    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    # test_path = data_dict['valid']

    img_size = 416

    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'cls_pw': 1.0,  # cls BCELoss positive_weight
           'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
           'obj_pw': 1.0,  # obj BCELoss positive_weight
           'iou_t': 0.1,  # iou training threshold
           'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.0005,  # final learning rate (with cos scheduler)
           'momentum': 0.937,  # SGD momentum
           'weight_decay': 0.000484,  # optimizer weight decay
           'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98 * 0,  # image rotation (+/- deg)
           'translate': 0.05 * 0,  # image translation (+/- fraction)
           'scale': 0.05 * 0,  # image scale (+/- gain)
           'shear': 0.641 * 0}  # image shear (+/- deg)

    dataset = LoadImagesAndLabels(train_path, img_size, train_batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)

    # Dataloader
    #batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), train_batch_size if train_batch_size > 1 else 0, 8])  # number of workers
    data_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=train_batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # # Testloader
    # testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, img_size, eval_batch_size,
    #                                                              hyp=hyp,
    #                                                              rect=True,
    #                                                              cache_images=opt.cache_images,
    #                                                              single_cls=opt.single_cls),
    #                                          batch_size=eval_batch_size,
    #                                          num_workers=nw,
    #                                          pin_memory=True,
    #                                          collate_fn=dataset.collate_fn)


    # Initialize model
    yolov3_model = Darknet(cfg, img_size)

    # load weights in .pt format
    yolov3_model.load_state_dict(torch.load(yolov3weights, map_location=device)['model'])

    print('Model size before quatization')
    print_size_of_model(yolov3_model)

    # Eval mode
    yolov3_model.to(device).eval()
    yolov3_model.fuse_model()

    yolov3_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(yolov3_model.qconfig)
    torch.quantization.prepare(yolov3_model, inplace=True)

    # calibrate activations
    calibrate(yolov3_model,  data_loader, num_calibration_batches)

    # convert to quantize model
    torch.quantization.convert(yolov3_model, inplace=True)

    print('Model size after quatization')
    print_size_of_model(yolov3_model)

    q_model_name = 'yolov3_ultralytics_q.pt'

    # save quantized model
    yolov3_model.save(os.path.join(savedir, q_model_name))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    quant()





