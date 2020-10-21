from easydict import EasyDict


_C = EasyDict()
_C.layers_dim = [1, 25, 1]
_C.batch_size = 90
_C.total_epochs = 40000
_C.resume = True  # False means retraining
_C.result_img_path = "D:/project/Pycharm/HJLNet/result.png"
_C.ckpt_path = 'D:/project/Pycharm/HJLNet/ckpt.npy'
