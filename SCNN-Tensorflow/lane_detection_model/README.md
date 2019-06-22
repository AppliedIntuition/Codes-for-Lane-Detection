# Clarification of this code base

by Anelise 


## Organization

The code is laid out in a very confusing way and the file names do not accurately describe what is contained in them. 

Despite the file and class names, this code uses no part of the LaneNet model. 

Some important files: 
    - `encoder_decoder_model/vgg_encoder.py`: *This file contains the meat of the model.* Contains an implementation of VGG16 with the message-passing functionality of SCNN built on top. See the function `VGG16Encoder.encode`. 
    - `lanenet_model/lanenet_merge_model.py`: DOES NOT CONTAIN LANENET. Basically a wrapper class for some helper methods, such as performing inference and calculating the loss. Does perform some post-processing, like smoothing, on model outputs. 
    - `encoder_decoder_model/cnn_basenet.py`: a base class that contains some convenience wrappers for common CNN building blocks. 
    - `tools/test_lanenet.py`: performs inference on a test set and writes model outputs to files. 
    - `tools/train_lanenet.py`: code to train and calculate metrics like accuracy, mIoU, etc. 
        - function `train_net`: 1) builds computation graphs for training and testing. 2) matches vgg pretrained weights to the layers in `vgg_encoder` by name and loads them into that part of the network semi-manually. 3) Runs training, displaying stuff periodically, clearing the metrics occasionally, saving stuff occasionally. Looks like its running the val set every time
