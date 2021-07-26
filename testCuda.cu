#include "./common/common.h"
#include <stdio.h>
#include <cuda.h>
#include "cublas_v2.h"

#include <time.h>
#include <stdlib.h>

#include <iostream>
#include <iomanip>

// print Matrix content in row-major
void printArr(float *arr_temp,int M, int N);


__global__ void getLU(float *array, float *const Aarray, int M, bool is_L)
{
    unsigned int i =  blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j =  blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int ix = i * M + j;
    if ( ix < M * M)
    {
        if(is_L)
        {
            if(j>i)
                array[ix] = Aarray[ix];
            else
                array[ix] = 0;
        }
        else
        {
            if(j<i)
                array[ix] = Aarray[ix];
            else
                array[ix] = 0;

        }
    }
}

// get matrix L or U (signal is_L)
void getLU_host(float *array, float *data, int M, bool is_L);
// write matrix contents into a txt file, row-major
void writeMat(FILE *fp, float *array, int M);
// generate the permutation matrix
void getP_host(int *pivot, float *eye, int M);

int main(int argc, char **argv)
{

    // initialize the GPU card
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    dim3 block;
    dim3 grid;

    // create cublas handle
    cublasHandle_t handle_cublas = 0;
    CHECK_CUBLAS(cublasCreate(&handle_cublas));

    // define Matrix size
    int M, N;

    M = 5;
    N = M;

    // define GPU accelerating parameters (threads, blocks, and grids)
    block.x = 32;
    block.y = 32;
    grid.x = (M+block.x-1)/block.x;
    grid.y = (N+block.y-1)/block.y;

    float **arr_A;
    float **arr_B;

    // init 
    arr_A = new float*[1];
    arr_A[0] = new float[M*N];

    arr_B = new float*[1];
    arr_B[0] = new float[M*N];


    // create a matrix

    srand(time(NULL));

    for (int j0=0;j0<M;j0++)
        for (int i0=0;i0<N;i0++)
        {
            int temp = rand();
            int ix = i0*M + j0;
            arr_A[0][ix] = (double) temp / (double) RAND_MAX;
        }

    //
    printf("Matrix A\n");
    printArr(arr_A[0], M, N);
    printf("\n");

    FILE *fp0 = fopen("mat_A.txt","w");
    writeMat(fp0,arr_A[0],M);
    fclose(fp0);

    // for GPU
    float **dev_arr_A = new float*[1];
    float **dev_arr_B;
    int *dev_arr_pivot;
    int *dev_inforArray;

    CHECK(cudaMalloc((void **)&dev_arr_A[0], sizeof(float) * M*N));
    CHECK(cudaMemcpy(dev_arr_A[0], arr_A[0], sizeof(float) * M*N, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    // to store the ** pointer in GPU
    CHECK(cudaMalloc((void **)&dev_arr_B, sizeof(float*) * 1));
    cudaDeviceSynchronize();

    CHECK(cudaMalloc((void **)&dev_arr_pivot,  sizeof(int) * M * 1));
    CHECK(cudaMalloc((void **)&dev_inforArray, sizeof(int) * 1));

    // copy the *pointer
    CHECK(cudaMemcpy(dev_arr_B, dev_arr_A, sizeof(float*) * 1, cudaMemcpyHostToDevice));

    CHECK_CUBLAS(cublasSgetrfBatched(handle_cublas, 
                M, 
                dev_arr_B,
                M,
                dev_arr_pivot,
                dev_inforArray,
                1)
            );
    
    cudaDeviceSynchronize();



    CHECK(cudaMemcpy(arr_B[0], dev_arr_A[0], sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();


    //
    printf("Matrix Aarray\n");
    printArr(arr_B[0], M, N);
    printf("\n");

    // get LU
    float *mat_L = new float[M*N];
    float *mat_U = new float[M*N];
    getLU_host(mat_L, arr_B[0], M, true);
    getLU_host(mat_U, arr_B[0], M, false);


    FILE *fp1 = fopen("mat_L.txt","w");
    FILE *fp2 = fopen("mat_U.txt","w");

    writeMat(fp1,mat_L,M);
    writeMat(fp2,mat_U,M);

    fclose(fp1);
    fclose(fp2);


    int *infoHost = new int[1];
    CHECK(cudaMemcpy(infoHost,dev_inforArray,sizeof(int)*1,cudaMemcpyDeviceToHost));

    std::cout << "info: " << infoHost[0] << std::endl;

    // pivoting
    int *pivotHost = new int [M];
    CHECK(cudaMemcpy(pivotHost, dev_arr_pivot, sizeof(int)*M,cudaMemcpyDeviceToHost));

    std::cout << "the pivoting array" << std::endl;
    for (int i=0;i<M;i++)
    {
        std::cout<<pivotHost[i]<<std::endl;
    }

    float *pivot_matrix = new float[M*M];

    getP_host(pivotHost,pivot_matrix, M);
    fp1 = fopen("mat_P.txt","w");
    writeMat(fp1,pivot_matrix,M);
    fclose(fp1);

    fp1 = fopen("pivot.txt","w");
    for (int i=0;i<M;i++)
    {
        fprintf(fp1,"%d\n",pivotHost[i]);
    }
    fclose(fp1);



    // finish
    CHECK_CUBLAS(cublasDestroy(handle_cublas));



}





void printArr(float *arr_temp,int M, int N)
{

    for (int i0=0;i0<N;i0++)
    {
        for (int j0=0;j0<M;j0++)
        {
            int ix = j0*M + i0;

            if(j0==0)
                std::cout <<"| ";
            else
                NULL;

            std::cout << std::fixed <<  std::setprecision(6) << std::setfill('0') << arr_temp[ix] << " ";

            if(j0==M-1)
                std::cout <<"|";
            else
                NULL;
        }
        std::cout << std::endl;

    }
    std::cout << std::endl;


}


void getLU_host(float *array, float *data, int M, bool is_L)
{
    for(int j=0;j<M;j++)
        for(int i=0;i<M;i++)
        {
            if(is_L)
            {
                int ix = j*M+i;
                if (i>j)
                    array[ix] = data[ix];
                else if (i==j)
                    array[ix] = 1;
                else
                    array[ix] = 0;
            }
            else
            {
                int ix = j*M+i;
                if (i<=j)
                    array[ix] = data[ix];
                else
                    array[ix] = 0;
            }
            

        }

}


void writeMat(FILE *fp, float *array, int M)
{
    for(int i=0;i<M;i++)
        for(int j=0;j<M;j++)
        {
            int ix = j*M+i;

            fprintf(fp,"%.6f",array[ix]);
            if(j==M-1)
                fprintf(fp,"\n");
            else
                fprintf(fp,",");

        }
}


void getP_host(int *pivot, float *eye, int M)
{
    // make an eye matrix
    for(int j=0;j<M;j++)
        for (int i=0;i<M;i++)
        {
            int ix = j*M+i;
            if (i==j)
                eye[ix] = 1;
            else
                eye[ix] = 0;
        }

    for (int i=0;i<M;i++)
    {
        // swap rows
        for (int j=0;j<M;j++) 
        {
            int ix = j*M+i;
            int ix_pivot = pivot[i] - 1 + j*M;
            float temp = eye[ix];
            eye[ix] = eye[ix_pivot];
            eye[ix_pivot] = temp;
        }
    }
}
