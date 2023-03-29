from __future__ import division
import os
import numpy as np
import rawpy
import glob
from PIL import Image
import torch
import torch.nn as nn
from GPUtil import showUtilization as gpu_usage
from numba import cuda



input_dir = './dataset/Sony/short/' # Path to the short exposure images
gt_dir = './dataset/Sony/long/' # Path to the long exposure images
checkpoint_dir = './result_Sony/' # Path to the checkpoint directory
result_dir = './result_Sony/' # Path to the result directory
ckpt = checkpoint_dir + 'model.ckpt' # Path to the model

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]


# Debug mode that only uses 5 images from the dataset
DEBUG = 1
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


# Leaky relu function with slope 0.2
def lrelu(x):
    outt = torch.max(0.2 * x, x)
    return outt


# Unet class
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__() # Call the init function of the parent class
        # Double convolutional layer and one maxpooling layer
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # One upsample layer, double convolutional layer 
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        # One upsample layer, double convolutional layer 
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        # One upsample layer, double convolutional layer 
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(64)

        # One upsample layer, double convolutional layer w
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(32)

        # One convolutional layer
        self.conv10 = nn.Conv2d(32, 12, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv1 = lrelu(self.conv1(x))
        conv1 = lrelu(self.bn1(self.conv1_2(conv1)))
        pool1 = self.pool1(conv1)
        
        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv2 = lrelu(self.conv2(pool1))
        conv2 = lrelu(self.b n2(self.conv2_2(conv2)))
        pool2 = self.pool2(conv2)

        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv3 = lrelu(self.conv3(pool2))
        conv3 = lrelu(self.bn3(self.conv3_2(conv3)))
        pool3 = self.pool3(conv3)

        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv4 = lrelu(self.conv4(pool3))
        conv4 = lrelu(self.bn4(self.conv4_2(conv4)))
        pool4 = self.pool4(conv4)

        # Forward pass through the double convolutional layer leaky relu activation
        conv5 = lrelu(self.conv5(pool4))
        conv5 = lrelu(self.bn5(self.conv5_2(conv5)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up6 = torch.cat([self.up6(conv5), conv4], 1)
        conv6 = lrelu(self.conv6(up6))
        conv6 = lrelu(self.bn6(self.conv6_2(conv6)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up7 = torch.cat([self.up7(conv6), conv3], 1)
        conv7 = lrelu(self.conv7(up7))
        conv7 = lrelu(self.bn7(self.conv7_2(conv7)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up8 = torch.cat([self.up8(conv7), conv2], 1)
        conv8 = lrelu(self.conv8(up8))
        conv8 = lrelu(self.bn8(self.conv8_2(conv8)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up9 = torch.cat([self.up9(conv8), conv1], 1)
        conv9 = lrelu(self.conv9(up9))
        conv9 = lrelu(self.bn9(self.conv9_2(conv9)))

        # Forward pass through the convolutional layer
        conv10 = self.conv10(conv9)

        # Pixel shuffle layer
        out = nn.functional.pixel_shuffle(conv10, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


# Pack the raw image into 4 channels using the bayer pattern
def pack_raw(raw):
    im = raw.raw_image_visible.astype(np.float32)  # Change data to float32
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)  # Add a channel dimension
    img_shape = im.shape  # Get the shape of the image
    H = img_shape[0]  # Get the height of the image
    W = img_shape[1]  # Get the width of the image

    # Channel concatenation for the bayer pattern
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


# loss function using absolute difference between the output and ground truth
def loss_function(out_image, gt_image):
    loss = torch.mean(torch.abs(out_image - gt_image))
    return loss


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
device = torch.device("cpu") # check if GPU is available

unet = UNet() # Initialize the model

unet.load_state_dict(torch.load(ckpt,map_location={'cuda:1':'cuda:0'}))
model = unet.to(device)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

for test_id in test_ids: # Loop through all test_ids
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id) # Get input image files (first image in each sequence) based on the test_id

    for k in range(len(in_files)): # Iterate through all input files
        in_path = in_files[k]
        _, in_fn = os.path.split(in_path)
        print(in_fn)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)  # Get the ground truth files for the current test_id

        _, gt_fn = os.path.split(gt_files[0])
        in_exposure = float(in_fn[9:-5]) # Extract exposure values from input
        gt_exposure = float(gt_fn[9:-5]) # Extract exposure values from ground truth
        ratio = min(gt_exposure / in_exposure, 300) # Calculate the exposure ratio and limit it to 300

        raw = rawpy.imread(in_path) # Read the raw input image
        im = raw.raw_image_visible.astype(np.float32) # Convert it to a visible float32 image
        input_full = np.expand_dims(pack_raw(im), axis=0) * ratio # Multiply image with exposure ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_raw = rawpy.imread(gt_files[0])  # Read the raw ground truth image and post-process it
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full, 1.0) # Clip the input image to the range [0, 1]

        in_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).to(device) # Convert the input image to a PyTorch tensor
        out_img = unet(in_img) # Perform the image enhancement using the UNet model

        # Convert to numpy array and clip between 0 and 1
        output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
        output = np.minimum(np.maximum(output, 0), 1)

        # Remove the batch dimension from the images
        output = output[0, :, :, :]
        gt_full = gt_full[0, :, :, :]
        scale_full = scale_full[0, :, :, :]
        scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)

        Image.fromarray((scale_full * 255).astype('uint8')).save(result_dir + '%5d_00_%d_ori.png' % (test_id, ratio))
        Image.fromarray((output * 255).astype('uint8')).save(result_dir + '%5d_00_%d_out.png' % (test_id, ratio))
        Image.fromarray((scale_full * 255).astype('uint8')).save(result_dir + '%5d_00_%d_scale.png' % (test_id, ratio))
        Image.fromarray((gt_full * 255).astype('uint8')).save(result_dir + '%5d_00_%d_gt.png' % (test_id, ratio))
