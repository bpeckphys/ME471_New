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

#define DIM 2048

struct cuComplex 
{
    float   r;
    float   i;
    cuComplex( float a, float b ) : r(a), i(b)  {}
    float magnitude2( void ) { return r * r + i * i; }
    cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

int julia( int x, int y ) 
{ 
    const float scale = 2.0;
    float xm = (float) DIM/2.0;
    float jx = scale * (x/xm-1);
    float jy = scale * (y/xm-1);    

    cuComplex c(-0.8, 0.156);
    cuComplex z(jx, jy);

    int i = 0;
    int maxiter = 200;
    for (i = 0; i < maxiter; i++) {
        z = z * z + c;
        if (z.magnitude2() > 4)
            return 0;
    }

    return 1;
}

void kernel( unsigned char *ptr ){
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            ptr[offset*4 + 0] = 255 * juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
 }

int main( void ) 
{
    CPUBitmap bitmap( DIM, DIM );
    unsigned char *ptr = bitmap.get_ptr();

    kernel( ptr );

    FILE *file = fopen("julia_cpu.out","w");
    int dim = DIM;
    fwrite(&dim,1,sizeof(int),file);
    fwrite(bitmap.get_ptr(),4*DIM*DIM,sizeof(unsigned char),file);
    fclose(file);
    // bitmap.display_and_exit();
}
