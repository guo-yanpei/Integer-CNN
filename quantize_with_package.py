"""
Script for PTQ using pytorch-quantization package
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neural_nets import ConvNet
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules.quant_conv import QuantConv2d
from pytorch_quantization.nn.modules.quant_linear import QuantLinear
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch.utils.data import DataLoader


def collect_stats(model, data_loader, num_bins):
    """Feed data to the network and collect statistic"""
    model.eval()
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
                if isinstance(module._calibrator, calib.HistogramCalibrator):
                    module._calibrator._num_bins = num_bins
            else:
                module.disable()

    for batch, _ in data_loader:
        x = batch.float()
        model(x)

        # Disable calibrators
        for _, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")


def quantize_model_params(model):
    """Quantize layer weights using calculated amax
       and process scale constant for C-code

    Args:
        state_dict (Dict): pytorch model state_dict
        amax (Dict): dictionary containing amax values
    """



    indices = [] 
    scale_factor = 31 # 127 for 8 bits

    
    state_dict = dict()
    for idx in range(len(model.net)):
        if isinstance(model.net[idx],QuantConv2d):
            indices.append((idx,0))
        elif isinstance(model.net[idx],QuantLinear):
            indices.append((idx,1))
    for layer_idx, (idx, linear) in enumerate(indices, start=1):
        # quantize all parameters
        weight = model.state_dict()[f'net.{idx}.weight']
        s_w = model.state_dict()[f'net.{idx}._weight_quantizer._amax'].numpy()
        s_x = model.state_dict()[f'net.{idx}._input_quantizer._amax'].numpy()
        #print(layer_idx,s_x)
        scale = weight * (scale_factor / s_w)
        state_dict[f'layer_{layer_idx}_weight'] = torch.clamp(scale.round(), min=-scale_factor, max=scale_factor).to(int)
        if linear: 
            state_dict[f'layer_{layer_idx}_weight'] = state_dict[f'layer_{layer_idx}_weight'].T # not very clear why
        state_dict[f'layer_{layer_idx}_weight'] = state_dict[f'layer_{layer_idx}_weight'].numpy()
        #print(s_w,s_x,scale_factor / s_x,scale_factor / s_w,s_x / scale_factor,s_w / scale_factor)
        state_dict[f'layer_{layer_idx}_s_x'] = scale_factor / s_x
        state_dict[f'layer_{layer_idx}_s_x_inv'] = s_x / scale_factor
        state_dict[f'layer_{layer_idx}_s_w_inv'] = (s_w / scale_factor).squeeze()        

    return state_dict
        
def eval_epoch(model: nn.Module, data_loader:DataLoader, loss_fn:nn.CrossEntropyLoss):
    """
    Evaluate epoch on validation data
    Args:
        model (nn.Module)
        data_loader (DataLoader)
        loss_fn (nn.CrossEntropyLoss)

    Returns:
        [Float]: average validation loss 
    """
    model.eval()
    num_batches = len(data_loader)

    loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            pred_y = model(x)
            batch_loss = loss_fn(pred_y, y)
            loss += batch_loss.item()
    return loss / num_batches
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='convnet_cifar_vgg16.th')
    parser.add_argument('--num_bins', help='number of bins', type=int, default=128)
    parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='./data')
    parser.add_argument('--save_dir', help='save directory', default='./saved_models/')

    args = parser.parse_args()
    # load model
    saved_stats = torch.load(args.save_dir + args.filename)


    
    state_dict = saved_stats['state_dict']
    
    hidden_sizes = None if 'convnet' in args.filename else saved_stats['hidden_sizes']
    channel_sizes = None if 'mlp' in args.filename else saved_stats['channel_sizes']
    

    quant_nn.QuantLinear.set_default_quant_desc_input(QuantDescriptor(calib_method='histogram', axis=None))
    quant_nn.QuantConv2d.set_default_quant_desc_input(QuantDescriptor(calib_method='histogram', axis=None))
    quant_nn.QuantLinear.set_default_quant_desc_weight(QuantDescriptor(calib_method='histogram', axis=None))
    quant_nn.QuantConv2d.set_default_quant_desc_weight(QuantDescriptor(calib_method='histogram', axis=None))
    quant_modules.initialize()


    model = ConvNet(channel_sizes=channel_sizes, out_dim=10)
    model.load_state_dict(state_dict)
    
    mnist_trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
           (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))

    train_loader = DataLoader(mnist_trainset, batch_size=len(mnist_trainset.data), num_workers=1, shuffle=False)
    
    mnist_test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
       (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))

    test_loader = DataLoader(mnist_test_set , batch_size=len(mnist_trainset.data), num_workers=1, shuffle=False)
    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, train_loader, args.num_bins)
        compute_amax(model, method="entropy")

    state_dict = quantize_model_params(model)
    saved_stats['state_dict'] = state_dict

    name = args.filename.replace('.th', '_quant.th')
    print(f"{args.save_dir}/{name}")
    torch.save(saved_stats, f"{args.save_dir}/{name}")
