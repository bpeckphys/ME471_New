#include "hmwk3.h"
#include <demo_util.h>
#include <mpi.h>
#include <math.h>
#include <stdio.h>

// Define the function

double foo(double x){
    return -1*(2*M_PI)*(2*M_PI)*cos(2*M_PI*x);
}

double exact(double x){
    return cos(2*M_PI*x);
}

int main(int argc, char** argv){
    // Initialize MPI
    int rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    // Initialize global variables

    int itermax = 5; // quick break for when itermax is not specified
    int p, M,N, err;
    double tol;

    // Read command line

    read_int(argc, argv, "-n", &N, &err);        
    read_int(argc, argv, "--itermax", &itermax, &err);
    read_double(argc, argv, "--tol", &tol, &err);
 

    // Set intervals
    
    M=N/nprocs;
    double sintvl = 1.0/nprocs;
    double a = sintvl*rank;
    double b = a + sintvl;
    double h = 1.0/N;

    // Initialize vectors

    double u[M+2];
    double prevu[M+2];
    double f[M+2];
    int i;
    for(i=1; i<M+1; i++){
        u[i] = 0;
        f[i] = h*h*foo(a+i*h-0.5*h);
    }

    // Initialize Boundary Conditions

    if(rank == 0){
        u[0] = 1;
    }
    if(rank == nprocs-1){
        u[M+1] = 1;
    }

    // Iterate over Jocobi Iterations
    int iter;
    double diff, largest_diff, ri, world_diff;
    for(iter=1; iter<itermax; iter++){
        largest_diff = 0;
        for(i=0; i<M+2; i++){
            prevu[i] = u[i];
        }
        //perform a Jacobi iteration, keeping track of largest update difference
        //beginning case
        i=1;
        if(rank==0){
            u[i] = -(1.0/3)*(f[i]-2*prevu[0]-prevu[i+1]);
            diff = fabs(u[i] - prevu[i]);
            if(diff>largest_diff){largest_diff = diff;}
        }else{
            u[i] = -0.5*(f[i] - prevu[i-1]-prevu[i+1]);
            diff = fabs(u[i] - prevu[i]);
            if(diff>largest_diff){largest_diff = diff;}
        }
        //middle case
        for(i=2; i<=M-1; i++){
            u[i] = -0.5*(f[i] - prevu[i-1]-prevu[i+1]);
            diff = fabs(u[i] - prevu[i]);
            if(diff>largest_diff){largest_diff = diff;}
        }
        //end case
        i=M;
        if(rank==nprocs-1){
            u[i] = -(1.0/3)*(f[i] - prevu[i-1]-2*prevu[i+1]);
            diff = fabs(u[i] - prevu[i]);
            if(diff>largest_diff){largest_diff = diff;}
        }else{
            u[i] = -0.5*(f[i] - prevu[i-1]-prevu[i+1]);
            diff = fabs(u[i] - prevu[i]);
            if(diff>largest_diff){largest_diff = diff;}
        }
        //find largest difference and break if below tol
        MPI_Allreduce(&largest_diff, &world_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(world_diff<tol){break;}
        
        //Synchronize end conditions
        MPI_Request left_request,right_request;
        if(rank!=0){
            //send left boundary
            MPI_Isend(&(u[1]), 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD,&left_request);
            //receive right boundary
            MPI_Recv(&(u[0]), 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
        }
        if(rank != nprocs-1){
            //send right boundary
            MPI_Isend(&(u[M]), 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD,&right_request);
            //recieve left boundary
            MPI_Recv(&(u[M+1]), 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
    }

    double max_error=0;
    for(i=1;i<M+1;i++){
        double error = u[i]-exact(a+i*h-0.5*h);
        if(error>max_error)
            max_error=error;
    }
    double world_error;
    MPI_Allreduce(&max_error, &world_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(rank==0){
        for(i=1; i<M+1; i++){
            printf("%.19g\n",u[i]);
        }
        printf("%d\n",iter);
        printf("%.19g\n",world_diff);
        printf("%.19g\n",world_error);
    }


    MPI_Finalize();
return 0;
}
