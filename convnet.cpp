
#include "convnet.h"
#include "nn.h"
#include "nn_math.h"
#include <stdlib.h>
#include<iostream>
using namespace std;
const int C0=3;

#define get_output_dim(input_dim,kernel_size,stride,pad) (input_dim +2*pad-kernel_size)/ stride+1

void run_convnet(const int *x, unsigned int *class_indices)
{
    const int LAYER_NUM=19;
    string model[100]={"C64","C64","M","C128","C128","M","C256","C256","C256","M","C512","C512","C512","M","C512","C512","C512","M","L10"};
    const int* layer_w[100]={layer_1_weight,layer_2_weight,layer_3_weight,layer_4_weight,layer_5_weight,layer_6_weight,layer_7_weight,layer_8_weight,layer_9_weight,layer_10_weight,layer_11_weight,layer_12_weight,layer_13_weight,layer_14_weight};
    const int layer_sx[100]={layer_1_s_x,layer_2_s_x,layer_3_s_x,layer_4_s_x,layer_5_s_x,layer_6_s_x,layer_7_s_x,layer_8_s_x,layer_9_s_x,layer_10_s_x,layer_11_s_x,layer_12_s_x,layer_13_s_x,layer_14_s_x};
    const int layer_sx_inv[100]={layer_1_s_x_inv,layer_2_s_x_inv,layer_3_s_x_inv,layer_4_s_x_inv,layer_5_s_x_inv,layer_6_s_x_inv,layer_7_s_x_inv,layer_8_s_x_inv,layer_9_s_x_inv,layer_10_s_x_inv,layer_11_s_x_inv,layer_12_s_x_inv,layer_13_s_x_inv,layer_14_s_x_inv};
    const int layer_sw_inv[100]={layer_1_s_w_inv,layer_2_s_w_inv,layer_3_s_w_inv,layer_4_s_w_inv,layer_5_s_w_inv,layer_6_s_w_inv,layer_7_s_w_inv,layer_8_s_w_inv,layer_9_s_w_inv,layer_10_s_w_inv,layer_11_s_w_inv,layer_12_s_w_inv,layer_13_s_w_inv,layer_14_s_w_inv};
    int* inbuf[100];
    int* outbuf[100];
    int now_in_c=C0;
    
    int now_in=32;
    int *in=(int*)x;
    int pc=0;
    for(int i=0;i<LAYER_NUM;i++)
    {
        int now_out;
        int now_out_c;
        int* out;
        //cout<<i<<" start"<<" "<<model[i]<<" "<<now_in<<" "<<now_in_c<<endl;
        if (model[i][0]=='C')
        {
            sscanf(model[i].c_str(),"C%d",&now_out_c);
            now_out=get_output_dim(now_in,3,1,1);
            
            out=new int[now_out_c*BATCH_SIZE*now_out*now_out];
            //cout<<i<<" start"<<" "<<model[i]<<" "<<now_out<<" "<<now_out_c<<endl;
            conv2d_layer(in, layer_w[pc], out, layer_sx[pc], layer_sx_inv[pc],layer_sw_inv[pc], 
                 BATCH_SIZE, now_in_c, now_out_c, now_in, now_in, now_out, now_out,
                 3, 3,  1, 1,1,1);
            //cout<<"end conv"<<endl;
            relu(out, BATCH_SIZE*now_out_c*now_out*now_out);
            ++pc;
        }
        else if(model[i][0]=='M')
        {
            now_out=get_output_dim(now_in,2,2,0);
            now_out_c=now_in_c;
            out=new int[now_out_c*BATCH_SIZE*now_out*now_out];
            pooling(in, out, BATCH_SIZE, now_in_c, now_in, now_in, now_out, now_out, 2, 2,  2, 2);
        }
        else if(model[i][0]=='L')
        {
            sscanf(model[i].c_str(),"L%d",&now_out_c);
            out=new int[now_out_c*BATCH_SIZE*now_in*now_in];

            linear_layer(in, layer_w[pc], out, layer_sx[pc], layer_sx_inv[pc],layer_sw_inv[pc],
                  BATCH_SIZE, now_in_c*now_in*now_in, OUTPUT_DIM, i!=7);  // not general
            ++pc;
        }
        inbuf[i]=in;
        outbuf[i]=out;
        now_in=now_out;
        now_in_c=now_out_c;
        in=out;
        if(i==LAYER_NUM-1)
        {
            argmax_over_cols(out, class_indices, BATCH_SIZE, OUTPUT_DIM);
        }
        
    }
}
