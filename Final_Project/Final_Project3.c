#include "hmwk3.h"
#include <demo_util.h>
#include <mpi.h>
#include <math.h>
#include <stdio.h>

// Define the function

double foo(double x, double y){
    return exp(x*y)*(x*x+y*y);
    //return -1*(2*M_PI)*(2*M_PI)*cos(2*M_PI*x*y);
}

double exact(double x, double y){
    return exp(x*y);
    //return cos(2*M_PI*x*y);
}

int main(int argc, char** argv){
    // Initialize MPI
    int rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    // Initialize global variables

    int itermax = 5; // quick break for when itermax is not specified
    int p, M, N, err;
    double tol,t,T,I;

    // Read command line

    read_int(argc, argv, "-n", &N, &err);        
    read_int(argc, argv, "--itermax", &itermax, &err);
    read_double(argc, argv, "--tol", &tol, &err);
 

    // Set intervals
    
    if(nprocs==1){
        M=N/nprocs+1;
    }else{
        M=N/nprocs+2;
    }
    double sintvl = 1.0/nprocs;
    double a = sintvl*rank;
    double h = 1.0/N;

    // Initialize vectors

    double u[M][N+1];
    double prevu[M][N+1];
    double f[M-1][N];
    int i,j;
    for(i=1; i<M-1; i++){
        for(j=1; j<N; j++){
            u[i][j] = 0;
            f[i][j] = h*h*foo(a+i*h,j*h);
        }
    }
    // Initialize Boundary Conditions
    if(rank == 0){
        for(i=0; i<M; i++){
            for(j=0; j<N+1; j++){
                t = j;
                I = i;
                u[0][j] = 1;
                u[i][0] = 1;
                u[M][j] = exp(t/N);
                u[i][N] = exp(I/N);
            }
        }
    }else if(rank == nprocs-1){
        for(i=0; i<M; i++){
            for(j=0; j<N+1; j++){
                t = a + i;
                T = j;
                u[0][j] = exp(T/N);
                u[i][0] = 1;
                u[M][j] = exp(T/N);
                u[i][N] = exp(t/N);
            }
        }
    }else{
        for(i=0; i<M; i++){
            for(j=0; j<N+1; j++){
                t = a + j;
                T = j;
                u[0][j] = exp(T/N);
                u[i][0] = 1;
                u[M][j] = exp(T/N);
                u[i][N] = exp(t/N);
            }
        }
    }
//    for(i=0; i<M+1; i++){
//        for(j=0; j<M+1; j++){
//            printf("%.19g\n",u[i][j]);
//        }
//    }
    // Iterate over Jocobi Iterations
    int iter;
    double diff, largest_diff, ri, world_diff;
    for(iter=1; iter<itermax; iter++){
        largest_diff = 0;
        for(i=0; i<M; i++){
            for(j=0; j<N+1; j++){
                prevu[i][j] = u[i][j];
            }
        }
        //perform a Jacobi iteration, keeping track of largest difference
        for(i=1; i<M-1; i++){
            for(j=1; j<N; j++){
                u[i][j] = -0.25*(f[i][j]-prevu[i-1][j]-prevu[i+1][j]-prevu[i][j-1]-prevu[i][j+1]);
                diff = fabs(u[i][j] - prevu[i][j]);
                if(diff>largest_diff){
                    //printf("%s %.19f\n","Diff is: ",diff);
                    largest_diff = diff;
                }
            }
        }
        //find largest difference and break if below tol
        MPI_Allreduce(&largest_diff, &world_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(world_diff<tol){break;}
      
        //Synchronize end conditions
        MPI_Request left_request,right_request;
        if(rank!=0){
            for(j=1; j<N; j++){
                //send left boundary
                MPI_Isend(&(u[2][j]), 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD,&left_request);
                //receive right boundary
                MPI_Recv(&(u[0][j]), 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }else{
            for(j=0; j<N+1; j++){
                u[0][j] = 1;
            }
        }
        if(rank != nprocs-1){
            for(j=1; j<N; j++){
                //send right boundary
                MPI_Isend(&(u[M-2][j]), 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD,&right_request);
                //recieve left boundary
                MPI_Recv(&(u[M][j]), 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }else{
            for(j=0; j<N+1; j++){
                t = j;
                u[M][j] = exp(t/N);
            }
        }
    }
    double max_error=0;
    for(i=1;i<M-1;i++){
        for(j=1;j<N;j++){
            double error = fabs(u[i][j]-exact(a+i*h,j*h));
            if(error>max_error)
                max_error=error;
        }
    }
    double world_error;
    MPI_Allreduce(&max_error, &world_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if(rank==0){
        for(j=0; j<N+1; j++){
            printf("%.19g\n",u[0][j]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for(t=0; t<nprocs; t++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==t){
            for(i=1; i<M-1; i++){
                for(j=0; j<N+1; j++){
                    printf("%.19g\n",u[i][j]);
                }
            }
        }
    }
    if(nprocs>1){
        if(rank==nprocs-1){
            for(i=M; i<M+1; i++){
                for(j=0; j<N+1; j++){
                    printf("%.19g\n",u[i][j]);
                }
            }
        }
    }else{
        for(i=M; i<M+1; i++){
            for(j=0; j<N+1; j++){
                printf("%.19g\n",u[i][j]);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0){
        printf("%d\n",iter);
        printf("%.19g\n",world_diff);
        printf("%.19g\n",world_error);
    }
    MPI_Finalize();
return 0;
}

