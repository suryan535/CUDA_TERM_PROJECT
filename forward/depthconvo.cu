
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <cstdlib>

__global__ void depthconvo(float *mat, float *col, int K, int channels, int height, int width, int height_col, int width_col, int stride)
{
    
    int tid_j = blockIdx.x*blockDim.x + threadIdx.x;	//column number
    int tid_i = blockIdx.y*blockDim.y + threadIdx.y;	//row number
    int gid = tid_i*(height_col*width_col) + tid_j;    //global_id when reading row major form
    
    if(tid_j < (height_col*width_col))
    {
        int c_im = blockIdx.y;

        int c = gid/(height_col*width_col);//row in which we are working on in the o/p matrix 
        
        int h_offset = (c/K)%K;
        int w_offset = c%K;
        int h =  (gid%(height_col*width_col))/width_col;
        int w = gid%width_col;


        
        int h_pad = h*stride + h_offset;
        int w_pad = w*stride + w_offset;
        
        int index = (c_im * height + h_pad) * width + w_pad;
        
        col[gid] = mat[index];
        	
    }
}

__global__ void rearrange_weights(float* wt_mat, float* out_wt_mat, int K, int channels)
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < channels*K*K)
    {
      int row = gid/(K*K);  //the row in the final output matrix that this thread has to work on 
      int off_set = row*(K*K*channels) + row*(K*K) + gid%(K*K); //Exact position where we have to put the value
      out_wt_mat[off_set] = wt_mat[gid];
      
    }
}


void random_init(float* M, int size)
{
    for(int i=0; i<size; i++)
    {
        M[i] = rand()%size;
    }

    return;
}

//depthwise convolution

void depth_conv(float *mat, float *weights, float *out_mat, int stride, int channels, int K, int height, int width)
{
    int width_col = (width- K)/stride + 1;
    int height_col = (height - K)/stride + 1;
    size_t totalThreads = channels*K*K*height_col*width_col;            //total elements im2col operation
    size_t dim1 = channels*K*K;                                         //size of weight matrix
    size_t dim2 = channels*channels*K*K;                                //size of output weight matrix
    size_t size = channels*height*width;

    cudaError_t error = cudaSuccess;
 
    float* d_mat = NULL;
    error = cudaMalloc((void **)&d_mat, size*sizeof(float));
    if(error != cudaSuccess) {
        fprintf(stderr,"Some Error in cudaMalloc for d_mat %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
 
    cudaMemcpy(d_mat, mat, size*sizeof(float), cudaMemcpyHostToDevice);
 

    float* d_col = NULL;
    error = cudaMalloc((void **)&d_col, totalThreads*sizeof(float));
    if(error != cudaSuccess) {
        fprintf(stderr,"Some Error in cudaMalloc for d_col %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    cudaDeviceProp devp;
    cudaGetDeviceProperties(&devp, 0);
    //printf("Warp Size: %d\n", devp.warpSize);
    //printf("Max number of threads per block: %d\n", devp.maxThreadsPerBlock);

    float num_th = 128.0;
    dim3 gridWeightDim(ceil((channels*K*K)/num_th), 1, 1);
    dim3 blockWeightDim(num_th, 1, 1);
 
 	  dim3 gridDim(ceil((height_col*width_col)/32.0), channels, 1);
    dim3 blockDim(32, K*K, 1);
 
    float* d_wt_mat = NULL;
    error = cudaMalloc((void **)&d_wt_mat, dim1*sizeof(float));
    if(error != cudaSuccess) {
        fprintf(stderr,"Some Error in cudaMalloc for d_wt_mat %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    cudaMemcpy(d_wt_mat, weights, dim1*sizeof(float), cudaMemcpyHostToDevice);
    float* d_out_wt_mat = NULL;
    error = cudaMalloc((void **)&d_out_wt_mat, dim2*sizeof(float));
    if(error != cudaSuccess) {
        fprintf(stderr,"Some Error in cudaMalloc for d_out_wt_mat %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n", gridWeightDim.x, gridWeightDim.y, gridWeightDim.z, blockWeightDim.x, blockWeightDim.y, blockWeightDim.z);

    depthconvo<<<gridDim ,blockDim>>>(d_mat, d_col, K, channels, height, width, height_col, width_col, stride);
 
    float* d_out_mat = NULL;
    error = cudaMalloc((void **)&d_out_mat, channels*width_col*height_col*sizeof(float));        
    if(error != cudaSuccess) {
        fprintf(stderr,"Some Error in cudaMalloc for d_out_mat %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(out_mat, d_out_mat, channels*width_col*height_col*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_out_mat);
    cudaFree(d_mat);
    cudaFree(d_col);
    cudaFree(d_wt_mat);
    cudaFree(d_out_wt_mat);
    cudaFree(d_out_mat);
}



int main()
{
    int K, height, width, stride, channels;	//kernel size , height of image, width of image, stride, number of channels in the input
    printf("Enter kernel size , height of image, width of image, stride, cnumber of channels in the input\n");
    
    //K SHOULD NOT BE LARGER THAN 5, NOT NEEDED IN THIS ARCHITECTURE ANYWAY. OURS IS CONSTRAINED BY BLOCK DIMENSIONS
    // 6*6*32 > 1024
 
    scanf("%d",&K);
    scanf("%d",&height);
    scanf("%d",&width);
    scanf("%d",&stride);
    scanf("%d",&channels);
    
    int group_size = 2;													//number of channels in a group 
    int num = ceil(channels/group_size);								//number of groups
    
    int width_col = (width- K)/stride + 1;			//effective width, that is the number of steps the kernel can be shifted along the width
    int height_col = (height - K)/stride + 1;		//effective height, that is the number of steps the kernel can be shifted along the height
    
    float* wt_mat = (float *)malloc((channels*K*K)*sizeof(float));
 
    random_init(wt_mat, channels*K*K);  //initializing the weight matrix
    
    printf("Weight Matrix \n");
    for(int i = 0; i < channels; i++)
    {
    	for(int j = 0; j < K; j++)
      {
      	for(int k = 0; k < K;k++)
        {
        	printf("%1.1f ",wt_mat[i*K*K + j*K + k]);		//printing the weight matrix 
        }
        	printf("\n");
      }
      printf("\n");
    }
    
 	  size_t size = channels*height*width;
    float* input_mat = (float *)malloc(size*5*sizeof(float));
 
    random_init(input_mat, size*5);//initializing input image matrix
    
    float* out_mat = (float *)malloc(channels*height_col*width_col*sizeof(float));

    int input_offset;						//offset to be provided to input image according to group number
    int weight_offset;						//offset to be provided to weight matrix according to group number
    int output_offset;						//offset to be provided to final output matrix according to group number
    int current_channels = group_size;		//number of channels in the current group, which is different only for possibly the last group
 
    float milliseconds = 0;
 
    cudaEvent_t start, stop;

    cudaEventCreate( & start);
    cudaEventCreate( & stop);

    cudaEventRecord(start);
    printf("Entering\n");
    
    for(int i = 0; i < channels; i+= group_size)
    {
    	input_offset = height*width*i;
	  	weight_offset = K*K*i;
		  output_offset = height_col*width_col*i;
		  if ((channels - i) < group_size)
			current_channels = channels - i;
		  depth_conv(input_mat+input_offset, wt_mat+weight_offset, out_mat+output_offset , stride, current_channels, K,  height, width);
    }
 
    printf("Left\n");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( & milliseconds, start, stop);
 
    printf("Depthwise Conv: The elapsed time in GPU was %f ms\n", milliseconds);
  
    free(wt_mat);
    free(input_mat);
    free(out_mat);
} 