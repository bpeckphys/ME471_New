/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM  2048

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia( int x, int y ) {
    const float scale = 2.0;
    float xm = (float) DIM/2.0;
    float jx = scale * (x/xm-1);
    float jy = scale * (1-y/xm);    
 
    cuComplex c(-0.8f, 0.156f);
    cuComplex z(jx, jy);

    int maxiter = 400;
    int i;
    for (i = 0; i < maxiter; i++) {
        z = z * z + c;
        if (z.magnitude2() > 4)
            return 0;  /* color will be black */
    }

    return 1;
}

__global__ void kernel( unsigned char *ptr ) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    int juliaValue = julia( x, y );
    int r = 255;
    int g = 0;
    int b = 0;
    ptr[offset*4 + 0] = r * juliaValue;
    ptr[offset*4 + 1] = g * juliaValue;
    ptr[offset*4 + 2] = b * juliaValue;
    ptr[offset*4 + 3] = 255;    /* Transparency? */
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;

    dim3 grid(DIM,DIM);
    kernel<<<grid,1>>>( dev_bitmap );

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );
                              
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
                              
    FILE *file = fopen("julia_gpu.out","w");
    int dim = DIM;
    fwrite(&dim,1,sizeof(int),file);
    fwrite(bitmap.get_ptr(),4*DIM*DIM,sizeof(unsigned char),file);
    fclose(file);

    cudaDeviceReset();
    
    // bitmap.display_and_exit();

}
