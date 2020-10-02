import argparse
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from utils.datasets import *
import torch.quantization
import tqdm
import time

yolov3weights = "./weights/yolov3.pt" #by ultralytics
savedir = "./weights/"
mixed_precision = True

def calibrate(model, data_loader, neval_batches, device):
    model.eval()

    cnt = 0
    with torch.no_grad():
        nb = len(data_loader)
        #pbar = tqdm(enumerate(data_loader), total=nb)
        for i, (imgs, targets, paths, _) in enumerate(data_loader): 
            print(imgs.size())
            imgs = imgs.float() / 255.0
            imgs = imgs.to(device)
            output = model.forward(imgs)[0]

            cnt += 1

            if cnt >= neval_batches:
                 return print('Calibration ended over ' + str(neval_batches) + ' batches, ' + str(opt.batch_size) + ' samples each')

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

    train_batch_size = opt.batch_size
    num_calibration_batches = 40

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
    print('Model size before quantization')
    print_size_of_model(yolov3_model)
    # Eval mode
    yolov3_model.to(device).eval()
    #yolov3_model.fuse_model()
    #yolov3_model = torch.nn.parallel.DistributedDataParallel(yolov3_model, find_unused_parameters=True)
    post_training_static_quant = 0
    dynamic_quant = 1

    if post_training_static_quant == 1:
    
        yolov3_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        print(yolov3_model.qconfig)
        torch.quantization.prepare(yolov3_model, inplace=True)

        # calibrate activations
        calibrate(yolov3_model.to(device),  data_loader, num_calibration_batches, device)

        # convert to quantize model
        print('Start conversion to quantized model')
        t1 = time.time()
        yolov3_ptsq = torch.quantization.convert(yolov3_model, inplace=True)
        t2 = time.time()
        print('Conversion ended in: ' + str(t2-t1))
    
        print('Model size after quatization')
        print_size_of_model(yolov3_ptsq)

        #q_model_name = 'yolov3_ultralytics_qscript.pt'
        q_model_name = 'yolov3_ultralytics_q.pt'

        # save quantized model
        #torch.save(yolov3_model.state_dict(), os.path.join(savedir, q_model_name))
        #torch.jit.save(torch.jit.script(yolov3_model), os.path.join(savedir, q_model_name))
        #torch.jit.save(yolov3_model.state_dict(), os.path.join(savedir, q_model_name2))
        torch.save(yolov3_ptsq.state_dict(), os.path.join(savedir, q_model_name))

        # load quantized model
        q_weights = os.path.join(savedir, q_model_name)
        yolov3_model.load_state_dict(torch.load(q_weights, map_location=device))
        print('Model size after quatization')
        print_size_of_model(yolov3_model)

    if dynamic_quant:

        #dynamic quantization
        print('Now dynamic quant trial')
        yolov3_q_dyn = torch.quantization.quantize_dynamic(yolov3_model, dtype=torch.qint8)
        print_size_of_model(yolov3_q_dyn)
        dq_model_name = 'yolov3_ultralytics_dq.pt'
        torch.save(yolov3_q_dyn.state_dict(), os.path.join(savedir, dq_model_name))

        # load quantized model
        dq_weights = os.path.join(savedir, dq_model_name)
        yolov3_model.load_state_dict(torch.load(dq_weights, map_location=device))
        print('Model size after quatization')
        print_size_of_model(yolov3_model)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[512], help='[min_train, max-train, test] img sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    #device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)

    quant()





