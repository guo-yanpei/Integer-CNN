#include "nn.h"
#include "nn_math.h"
#include <stdio.h>
#include<iostream>
#include <cassert>
using namespace std;
int max(int a,int b)
{
    return a>b?a:b;
}
int min(int a,int b)
{
    return a<b?a:b;
}
FILE* fp=NULL;
void linear_layer(const int *x, const int *w, int *output, const int x_scale_factor,const int x_scale_factor_inv,
                  const int w_scale_factor_inv, 
                  const unsigned int  N, const unsigned int  K, const unsigned int  M,
                  const unsigned int  hidden_layer)
{
    fprintf(fp,"matrix mult weight %d %d, output:%d  \n",M,K,M);
    int* ww=new int[K*M];
    fprintf(fp,"matrix weight:");
    for(int i=0;i<K*M;i++)
    {
        ww[i]=w[i]*w_scale_factor_inv;   //dump this I think
        fprintf(fp,"%d ",ww[i]);
    }
    fprintf(fp,"\n");
    fprintf(fp,"matrix output:");
    mat_mult(x, ww, output, K, M);
    for(int i=0;i<M;i++)
    {
        fprintf(fp,"%d ",output[i]);
    }
    fprintf(fp,"\n");
    fclose(fp);
}




void conv2d_layer(const int *x, const int *w,int *output, const int x_scale_factor, const int x_scale_factor_inv, const int w_scale_factor_inv, 
                  const unsigned int N, const unsigned int C_in, const unsigned int C_out, const int H, const int W,
                  const int H_conv, const int W_conv, const int k_size_h, const int k_size_w,  const int stride_h, const int stride_w,int pad_h,int pad_w)
{
    static int SUM1=0;
    static int SUM2=0;
    SUM1+=C_in*H*W;
    SUM2+=C_out*C_in*k_size_h*k_size_w;
    int* ww=new int[C_in*C_out*k_size_h*k_size_w];
    for(int i=0;i<C_in*C_out*k_size_h*k_size_w;i++)
        ww[i]=w[i]*w_scale_factor_inv;   //dump this I think

    if(fp==NULL)
        fp=fopen("dat.txt","w");
    fprintf(fp,"Conv input %d %d %d Conv output %d %d %d Conv weight %d %d %d %d\n",C_in,H,W,C_out,H_conv,W_conv, C_in,C_out,k_size_h,k_size_w);
    if(C_in==3)
    {
        fprintf(fp,"image input\n");
        for(int i=0;i<C_in*H*W;i++)
        {
            fprintf(fp,"%d ",x[i]);
        }
        fprintf(fp,"\n");
    }
    fprintf(fp,"conv weight\n");
    for(int i=0;i<C_in*C_out*k_size_h*k_size_w;i++)
    {
        fprintf(fp,"%d ",ww[i]);
    }
    fprintf(fp,"\n");


    conv2d(x, ww, output, N, C_in, C_out, H, W, H_conv, W_conv,
            k_size_h, k_size_w,  stride_h, stride_w,pad_h,pad_w);
    static int mx=-1e9,mn=1e9;
    int oldmx=mx,oldmn=mn;
    fprintf(fp,"conv output\n");

        for (int c = 0; c < C_out; c++)
            for (int k =0; k < H_conv*W_conv; k++)
            {
                assert(output[c* H_conv*W_conv + k]<(1<<30));
                fprintf(fp,"%d ",output[c* H_conv*W_conv + k]);  // I think no need to consider batching
                output[c* H_conv*W_conv + k]>>=FXP_VALUE;
                mx=max(mx,output[c* H_conv*W_conv + k]);
                mn=min(mn,output[c* H_conv*W_conv + k]);
            }
    fprintf(fp,"\n");
    fprintf(fp,"downscale output\n");
        for (int c = 0; c < C_out; c++)
            for (int k =0; k < H_conv*W_conv; k++)
            {
                fprintf(fp,"%d ",output[c* H_conv*W_conv + k]); 
            }
    fprintf(fp,"\n");
}

void relu(int *tensor, const unsigned int size)
{
    fprintf(fp,"relu:%d\n",size);
    fprintf(fp,"relu output:\n");
    for (int i = 0; i < size; i++)
    {
        tensor[i] = MAX(tensor[i], 0);
        fprintf(fp,"%d ",tensor[i]); 
    }
    fprintf(fp,"\n");
}

void pooling(int *x, int *y, int N, int C_out, int H, int W, int H_new, int W_new,
            int k_size_h, int k_size_w,  int stride_h, int stride_w)
        {
            fprintf(fp,"max pool in:%d %d %d  %d %d %d\n",C_out,H,W,C_out,H_new,W_new);
            fprintf(fp,"max pooling output:\n");
            pooling2d(x, y, N, C_out, H,  W,  H_new, W_new,  k_size_h,  k_size_w,   stride_h, stride_w);
            for(int i=0;i<C_out*H_new*W_new;i++)
                fprintf(fp,"%d ",y[i]);
            fprintf(fp,"\n");
        }