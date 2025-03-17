"""
Script for writing param header and source files in C with weights and amax values calculate in python
"""
import argparse
from pathlib import Path

import torch


def get_output_dim(input_dim, kernel_size, stride,pad=0):
            output_dim = (input_dim +2*pad-kernel_size) // stride+1
            return output_dim

FXP_VALUE=6
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename of quantized model', type=str, default='convnet_cifar_vgg16_quant.th')
    parser.add_argument('--save_dir', help='save directory', default='./saved_models/')

    args = parser.parse_args()

    saved_stats = torch.load(args.save_dir + args.filename)
    state_dict = saved_stats['state_dict']
    channel_sizes = saved_stats['channel_sizes']

    layer_num=0
    
    for i in range(1,100):
         if state_dict.get(f"layer_{i}_s_x") is not None:
            layer_num=i
              
    
    # create header file
    with open('convnet_params.h', 'w') as f:
        f.write('/*******************************************************************\n')
        f.write('@file convnet_params.h\n*  @brief variable prototypes for model parameters and amax values\n*\n*\n')
        f.write('*  @author Benjamin Fuhrer\n*\n')
        f.write('*******************************************************************/\n')
        f.write('#ifndef CONVNET_PARAMS\n#define CONVNET_PARAMS\n\n')

        f.write(f'#define INPUT_DIM {32*32*3}\n')
        f.write(f'#define OUTPUT_DIM {10}\n\n')
        f.write('#include <stdint.h>\n\n\n') 


        f.write('// quantization/dequantization constants\n')
        for layer_idx in range(1, layer_num+1):
            name = f'layer_{layer_idx}_s_x'
            f.write(f"extern const int {name};\n")

            name = f'layer_{layer_idx}_s_x_inv'
            f.write(f"extern const int {name};\n")

            name = f'layer_{layer_idx}_s_w_inv'
            value = state_dict[name]
            f.write(f"extern const int {name};\n")

        f.write('// Layer quantized parameters\n')
        for layer_idx in range(1, layer_num+1):
            name = f'layer_{layer_idx}_weight'
            param = state_dict[f'layer_{layer_idx}_weight']
            f.write(f"extern const int {name}[{len(param.flatten())}];\n")

        f.write('\n#endif // end of CONVNET_PARAMS\n')

    # create source file
    with open('convnet_params.cpp', 'w') as f:
        f.write('#include "convnet_params.h"\n\n\n')

        for layer_idx in range(1, layer_num+1):
            name = f'layer_{layer_idx}_s_x'
            print(state_dict[name])
            fxp_value = (state_dict[name] * (2**FXP_VALUE)).round()
            f.write(f"const int {name} = {int(fxp_value)};\n\n")

            name = f'layer_{layer_idx}_s_x_inv'
            print(state_dict[name])
            fxp_value = (state_dict[name] * (2**FXP_VALUE)).round()
            f.write(f"const int {name} = {int(fxp_value)};\n\n")
            name = f'layer_{layer_idx}_s_w_inv'
            print(state_dict[name])
            fxp_value = (state_dict[name] * (2**FXP_VALUE)).round()
            print("write in ",fxp_value)
            f.write(f"const int {name} = {int(fxp_value)};\n\n")

            #for idx in range(len(fxp_value)):
            #    f.write(f"{int(fxp_value[idx])}")
            #    if idx < len(fxp_value) - 1:
            #         f.write(", ")
            #f.write("};\n\n")

        for layer_idx in range(1, layer_num+1):
                name = f'layer_{layer_idx}_weight'
                param = state_dict[f'layer_{layer_idx}_weight']
                param = param.flatten()
                f.write(f"const int {name}[{len(param)}] = {{")
                for idx in range(len(param)):
                    f.write(f"{param[idx]}")
                    if idx < len(param) - 1:
                        f.write(", ")
                f.write("};\n")
