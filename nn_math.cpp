
#include "nn_math.h"
#include <stdio.h>
#include<iostream>
#include <cstring>
#include <cassert>
#undef NDEBUG
using namespace std;
void mat_mult(const int *mat_l, const int *mat_r, int *result, const unsigned int K, const unsigned int M)
{       
    unsigned int n, k, m;
    unsigned int row, col;
    int accumulator;

    for (m = 0; m < M; m++)
    {
        
            accumulator = 0;
            for (k = 0; k < K; k++)
            {
                accumulator += mat_l[k] * mat_r[k*M + m];
            }
            result[m] = accumulator;
        
    }
}

void conv2d(const int *x, const int *w, int *y, int N, int C_in, int C_out, int H, int W, int H_new, int W_new,
            int k_size_h, int k_size_w, int stride_h, int stride_w, int pad_h, int pad_w)
{
    static int Y=0;
    ++Y;
    int n_i, c_out_j, c_in_i; /* sample and channels */
    int n, m; /* kernel iterations */
    int i, j; /* output image iteration */
    int *y_new=new int[H_new*W_new];
    memset(y_new,0,sizeof(int)*H_new*W_new);
        for (c_out_j = 0; c_out_j < C_out; c_out_j++)
        {
            for (i = 0; i < H_new; i++)
            {
                for (j = 0; j < W_new; j++)
                {
                    int output_idx_y = i * W_new + j;
                    int S = 0;
                    for (c_in_i = 0; c_in_i < C_in; c_in_i++)
                    {
                        int in_sum=0;
                        for (n = 0; n < k_size_h; n++)
                        {
                            for (m = 0; m < k_size_w; m++)
                            {
                                int x_i = i * stride_h + n - pad_h;
                                int x_j = j * stride_w + m - pad_w;
                                int x_value = 0; // Default to 0 for padding
                                if (x_i >= 0 && x_i < H && x_j >= 0 && x_j < W)
                                {
                                    x_value = x[c_in_i * H * W+ x_i * W + x_j];
                                }
                                //x[c_in_i][i+n-1][j+m-1]
                                //w[c_out_j][c_in_i][n][m]
                                int w_value = w[c_out_j * C_in * k_size_h * k_size_w + c_in_i * k_size_h * k_size_w + n * k_size_w + m];
                                S+= x_value * w_value;
                                if(c_in_i==0 && c_out_j==0)
                                    y_new[i*W_new+j]+=x_value * w_value;
                                //y[c_out_j][i][j]
                            }
                        }
                        assert(S<(1<<30));
                        
                    }

                    y[c_out_j * H_new * W_new + i * W_new + j] = S;
                }
            }
        }
        // for(int i=0;i<H_new;i++)
        // for(int j=0;j<W_new;j++)
        //     cout<<y_new[i*H_new+j]<<" ";
        // exit(0);
}
void pooling2d(int *x, int *y, int N, int C_out, int H, int W, int H_new, int W_new,
            int k_size_h, int k_size_w,  int stride_h, int stride_w)
{
    int n_i, c_out_j; /* sample and channels*/
    int n, m; /* kernel iterations */
    int i, j; /* output image iteration*/
    
    for (n_i = 0; n_i < N; n_i++)
    {
        int N_idx_y = n_i*C_out*H_new*W_new;
        int N_idx_x = n_i*C_out*H*W;
        
        for (c_out_j = 0; c_out_j < C_out; c_out_j++)
        {
            int C_out_idx_y = c_out_j*H_new*W_new;
            int C_out_idx_x = c_out_j*H*W;

            for (i = 0; i < H_new; i++)
            {
                for (j = 0; j < W_new; j++)
                {
                    int output_idx_y = i*W_new + j;
                    int output_idx_x = i*stride_h*W + j*stride_w;
                    
                    int max = x[N_idx_x+ C_out_idx_x + output_idx_x];
                    for (n = 0; n < k_size_w; n++)
                    {
                        for (m = 0; m < k_size_h; m++)
                        {
                            int kernel_idx = n*W + m;
                            
                            int value = x[N_idx_x+ C_out_idx_x + kernel_idx + output_idx_x];
                            if (value > max)
                                max = value;
                        }
                    }
                    y[N_idx_y + C_out_idx_y + output_idx_y] = max;
                }
                
            }
        }
    }
}




void argmax_over_cols(const int *mat_in, unsigned int *indices, const unsigned int N, const unsigned int M)
{

    // calculate max of each row
    unsigned int n, m, max_idx;
    int row_max, value;
    for (n = 0; n < N; n++)
    {
        row_max = mat_in[n*M];
        max_idx = 0;
        for (m = 0; m < M; m++)
        {
            value = mat_in[n*M + m];
            if (value > row_max)
            {
                row_max = value;
                max_idx = m; // return column
            }
        }
        indices[n] = max_idx;
    }
}
