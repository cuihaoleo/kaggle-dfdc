##################################################
# Training Config
##################################################
GPU = '0'                   # GPU
workers = 6                 # number of Dataloader workers
epochs = 60                 # number of epochs
batch_size = 10             # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (320, 320)     # size of training images
net = 'xception'            # feature extractor
num_attentions = 8          # number of attention maps
beta = 5e-2                 # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'dfdc'
pretrained = 'output/dfdc-xception/best.pth'
# saving directory of .ckpt models
save_dir = './output/dfdc-wsdan-xception/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
datapath = "/mnt/ssd0/dfdc"
