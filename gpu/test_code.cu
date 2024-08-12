#include <cstdio>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cudnn_cnn.h>

struct Tensor4d
{
    cudnnTensorDescriptor_t desc;
    void *data;
    size_t data_size;

    Tensor4d(int n, int c, int h, int w)
    {
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensor4dDescriptor(desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   n, c, h, w);
        data_size = n * c * h * w;
        cudaMalloc((void **)&data, data_size * sizeof(float));
    }
    ~Tensor4d()
    {
        cudaFree(data);
    }
};

struct Bias4d
{
    cudnnTensorDescriptor_t desc;
    void *data;
    size_t data_size;

    Bias4d(int n, int c, int h, int w)
    {
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensor4dDescriptor(desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   n, c, h, w);
        data_size = n * c * h * w;
        float *tmp = (float *)malloc(data_size * sizeof(float));
        for (int i = 0; i < data_size; i++)
        {
            tmp[i] = (float)std::rand() / RAND_MAX / 1000;
        }
        cudaMalloc((void **)&data, data_size * sizeof(float));
        auto code = cudaMemcpy(data, tmp, data_size * sizeof(float),
                               cudaMemcpyHostToDevice);
    }
    ~Bias4d()
    {
        cudaFree(data);
    }
};

struct Filter4d
{
    cudnnFilterDescriptor_t desc;
    void *data;
    size_t data_size;

    Filter4d(int n, int c, int h, int w)
    {
        cudnnCreateFilterDescriptor(&desc);
        cudnnSetFilter4dDescriptor(desc,
                                   CUDNN_DATA_FLOAT,
                                   CUDNN_TENSOR_NCHW,
                                   n, c, h, w);
        data_size = n * c * h * w;
        float *tmp = (float *)malloc(data_size * sizeof(float));
        for (int i = 0; i < data_size; i++)
        {
            tmp[i] = (float)std::rand() / RAND_MAX / 1000;
        }

        cudaMalloc((void **)&data, data_size * sizeof(float));
        auto code = cudaMemcpy(data, tmp, data_size * sizeof(float),
                               cudaMemcpyHostToDevice);
    }
    ~Filter4d()
    {
        cudaFree(data);
    }
};

struct zeros
{
    void *data;
    size_t data_size;
    zeros(std::vector<int> dims)
    {
        data_size = std::accumulate(dims.begin(),
                                    dims.end(),
                                    1,
                                    std::multiplies<int>());
        std::vector<float> host_data(data_size);
        for (int i = 0; i < data_size; i++)
            host_data[i] = 0;

        cudaMalloc((void **)&data, data_size * sizeof(float));
        cudaMemcpy(data, host_data.data(), data_size * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
    ~zeros()
    {
        cudaFree(data);
    }
};



void cuConv2D(float *input, float *output, int w, int h, int c, int n, int k,
              int filter_w, int filter_h, int dilation_w, int dilation_h,
              int pad_w, int pad_h, int wstride, int hstride)
{

    size_t fwd_workspace_size;
    cudnnConvolutionFwdAlgo_t fwd_algo;

    const float alpha = 1.f;
    const float beta = 0.f;

    cudnnHandle_t cudnn_handle;
    // create cudnn handle
    cudnnCreate(&cudnn_handle);

    // datatype
    cudnnDataType_t dataType;
    dataType = CUDNN_DATA_FLOAT;

    // convolution mode
    cudnnConvolutionMode_t mode;
    mode = CUDNN_CONVOLUTION;

    int out_h, out_w, out_c, out_n;
    std::vector<int> output_dims_;

    // create conv desc
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    pad_h,
                                    pad_w,
                                    hstride,
                                    wstride,
                                    dilation_w,
                                    dilation_h,
                                    mode,
                                    dataType);

    // tensor desc
    Tensor4d x_desc(n, c, h, w);

    auto code = cudaMemcpy(x_desc.data, input, x_desc.data_size * sizeof(float),
                           cudaMemcpyHostToDevice);

    // filter desc
    Filter4d w_desc(k, c, filter_w, filter_h);

    // get conv dim
    cudnnGetConvolution2dForwardOutputDim(conv_desc,
                                          x_desc.desc,
                                          w_desc.desc,
                                          &out_n,
                                          &out_c,
                                          &out_h,
                                          &out_w);

    Tensor4d h_desc(out_n, out_c, out_h, out_w);

    // choose forward algorith
    const int requestAlgoCount = 1;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;

    cudnnFindConvolutionForwardAlgorithm(cudnn_handle,
                                         x_desc.desc,
                                         w_desc.desc,
                                         conv_desc,
                                         h_desc.desc,
                                         requestAlgoCount,
                                         &returnedAlgoCount,
                                         &perfResults);

    // what algorithm is choosed
    fwd_algo = perfResults.algo;

    // get workspace size
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                            x_desc.desc,
                                            w_desc.desc,
                                            conv_desc,
                                            h_desc.desc,
                                            fwd_algo,
                                            &fwd_workspace_size);

    std::vector<int> u = std::vector<int>{static_cast<int>(fwd_workspace_size / sizeof(float)), 1};

    // init workspace
    zeros fwd_workspace(u);

    cudnnActivationDescriptor_t activationDesc;
    cudnnCreateActivationDescriptor(&activationDesc);
    auto code3 = cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 100);


    Bias4d bias(k, c, w, h);

    auto start = std::chrono::steady_clock::now();

    // fwd conv
    auto code2 = cudnnConvolutionBiasActivationForward(cudnn_handle,
                                                       &alpha,
                                                       x_desc.desc,
                                                       x_desc.data,
                                                       w_desc.desc,
                                                       w_desc.data,
                                                       conv_desc,
                                                       fwd_algo,
                                                       fwd_workspace.data,
                                                       fwd_workspace_size,
                                                       &beta,
                                                       h_desc.desc,
                                                       h_desc.data,
                                                       bias.desc,
                                                       bias.data,
                                                       activationDesc,
                                                       h_desc.desc,
                                                       h_desc.data);


    code = cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double,
                                                          std::micro>(end - start)
                                        .count());

    std::cout << " " << fwd_time << " ms" << std::endl;

    code = cudaMemcpy(output, h_desc.data, h_desc.data_size * sizeof(float), cudaMemcpyDeviceToHost);

    // destroy conv desc
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return;
}


__global__ void addBias(float *output, float *bias, int output_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size)
    {
        output[idx] += bias[idx];
    }
}
__global__ void conv2d_kernel(float* input, float* output, float* filter, int w, int h, int c, int n, int k, int filter_w, int filter_h, int pad_w, int pad_h, int wstride, int hstride) {
    extern __shared__ float shared_input[];

    int out_w = (w + 2 * pad_w - filter_w) / wstride + 1;
    int out_h = (h + 2 * pad_h - filter_h) / hstride + 1;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * k * out_w * out_h) return;

    int out_x = index % out_w;
    int out_y = (index / out_w) % out_h;
    int out_c = (index / (out_w * out_h)) % k;
    int out_n = index / (k * out_w * out_h);

    // Define shared memory size for each block
    int input_offset = threadIdx.x;
    int shared_mem_size = (filter_w + 2 * pad_w) * (filter_h + 2 * pad_h) * c;

    // Load input data to shared memory
    for (int s = input_offset; s < shared_mem_size; s += blockDim.x) {
        int in_x = (s % (filter_w + 2 * pad_w)) - pad_w + out_x * wstride;
        int in_y = ((s / (filter_w + 2 * pad_w)) % (filter_h + 2 * pad_h)) - pad_h + out_y * hstride;
        int in_c = s / ((filter_w + 2 * pad_w) * (filter_h + 2 * pad_h));

        if (in_x >= 0 && in_x < w && in_y >= 0 && in_y < h) {
            int global_in_index = ((out_n * c + in_c) * h + in_y) * w + in_x;
            shared_input[s] = input[global_in_index];
        } else {
            shared_input[s] = 0.0f;
        }
    }

    __syncthreads();

    // Compute convolution
    float value = 0.0f;
    for (int i = 0; i < filter_h; ++i) {
        for (int j = 0; j < filter_w; ++j) {
            for (int l = 0; l < c; ++l) {
                int shared_index = ((l * (filter_h + 2 * pad_h) + (i + pad_h)) * (filter_w + 2 * pad_w)) + (j + pad_w);
                int filter_index = ((out_c * c + l) * filter_h + i) * filter_w + j;
                value += shared_input[shared_index] * filter[filter_index];
            }
        }
    }

    output[index] = value;
}



void cuConv2D_custom(float *input, float *output, int w, int h, int c, int n, int k,
              int filter_w, int filter_h, int dilation_w, int dilation_h,
              int pad_w, int pad_h, int wstride, int hstride)
{
    float *d_input, *d_output, *d_filter;
    int input_size = n * c * w * h * sizeof(float);
    int output_size = n * k * ((w + 2 * pad_w - filter_w) / wstride + 1) * ((h + 2 * pad_h - filter_h) / hstride + 1) * sizeof(float);
    int filter_size = k * c * filter_w * filter_h * sizeof(float);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_filter, filter_size);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);

    float *h_filter = (float *)malloc(filter_size);
    for (int i = 0; i < k * c * filter_w * filter_h; ++i) {
        h_filter[i] = static_cast<float>(i/ 1000);
    }
    cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);

    // Create filter and bias
    Filter4d w_desc(k, c, filter_w, filter_h);
    Bias4d bias(k, 1, 1, 1); // bias is 1x1x1xk

    // Compute output dimensions
    int out_h = (h + 2 * pad_h - dilation_h * (filter_h - 1) - 1) / hstride + 1;
    int out_w = (w + 2 * pad_w - dilation_w * (filter_w - 1) - 1) / wstride + 1;

    // Allocate device memory for output
    Tensor4d h_desc(n, k, out_h, out_w);

    // Compute grid and block dimensions for convolution kernel
    int threads_per_block = 512;
    int sharedMemSize = (filter_w + 2 * pad_w) * (filter_h + 2 * pad_h) * c * sizeof(float);  // Size of shared memory
    int num_blocks = (n * k * ((w + 2 * pad_w - filter_w) / wstride + 1) * ((h + 2 * pad_h - filter_h) / hstride + 1) + threads_per_block - 1) / threads_per_block;
    auto start = std::chrono::steady_clock::now();
    dim3 DimBlock(4, 4, 64);
    conv2d_kernel<<<num_blocks, DimBlock, sharedMemSize>>>(d_input, d_output, d_filter, w, h, c, n, k, filter_w, filter_h, pad_w, pad_h, wstride, hstride);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    // Add bias
    addBias<<<(n * k + 255) / 256, 256>>>((float*)h_desc.data, (float*)bias.data, n * k);



    cudaDeviceSynchronize();


    cudaMemcpy(output, h_desc.data, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    std::srand(std::time(0));

    float *input;
    float *output;

    int data_size = 224 * 224 * 3 * 1;
    input = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; i++)
    {
        input[i] = i/1000;
    }

    // ===============  1 =====================
    std::cout << "CONV 224x224x64";
    output = (float *)malloc(224 * 224 * 64 * 1 * sizeof(float));
    cuConv2D(input, output, 224, 224, 3, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    // std::swap(input, output);



    std::srand(std::time(0));
    float *input2;
    float *output2;

    data_size = 224 * 224 * 3 * 1;
    input2 = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; i++)
    {
        input2[i] = i/1000;
    }

    // ===============  1 =====================
    std::cout << "CONV 224x224x64";
    output2 = (float *)malloc(224 * 224 * 64 * 1 * sizeof(float));
    cuConv2D_custom(input2, output2, 224, 224, 3, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    // std::swap(input2, output2);


    for(int i=0; i<data_size; i++) {
        if(output[i]!=output2[i]) {
            std::cout<<"not identical "<<i<<std::endl;

        }
    }
    free(output);
    free(output2);



}