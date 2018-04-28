#include "hmwk3.h"
#include <demo_util.h>
#include <mpi.h>
#include <math.h>
#include <stdio.h>

// Define the function
void syncvec(double u[], int M);
void matvec(double x[], double b[], int M){
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    syncvec(x,M);
            int i=1;
        if(rank==0){
            b[i] = (2*x[0]-3*x[i]+x[i+1]);
        }else{;
            b[i] = (x[i-1]-2*x[i]+x[i+1]);
        }
        //middle case
        for(i=2; i<=M-1; i++){;
            b[i] = (x[i-1]-2*x[i]+x[i+1]);
        }
        //end case
        i=M;
        if(rank==nprocs-1){;
            b[i] = (x[i-1]-3*x[i]+2*x[i+1]);
        }else{;
            b[i] = (x[i-1]-2*x[i]+x[i+1]);
        }
}
double vecvec(double x[], double y[], int M){
    double sum=0;
    for(int i=1;i<M+1;i++){
        sum+=x[i]*y[i];
    }
    double world_sum;
    MPI_Allreduce(&sum, &world_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return world_sum;
}
void syncvec(double u[], int M){
 int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        //Synchronize end conditions
        MPI_Request left_request,right_request;
        if(rank!=0){
            //send left boundary
            MPI_Isend(&(u[1]), 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD,&left_request);
            //receive left boundary
            MPI_Recv(&(u[0]), 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if(rank != nprocs-1){
            //send right boundary
            MPI_Isend(&(u[M]), 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD,&right_request);
            //recieve left boundary
            MPI_Recv(&(u[M+1]), 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

}
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
    int M,N, err;
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
    //printf("%.19g\n",h);

    // Initialize vectors

    double u[M+2];
    double f[M+2];
    double r[M+2];
    double z[M+2];
    double p[M+2];
    double ap[M+2];
    int i;
    for(i=0; i<M+2; i++){
        u[i] = 0;
        r[i] = 0;
        p[i] = 0;
        ap[i] = 0;
        f[i] = h*h*foo(a+i*h-0.5*h);
    }

    // Initialize Boundary Conditions

    if(rank == 0){
        u[0] = 1;
    }
    if(rank == nprocs-1){
        u[M+1] = 1;
    }

    matvec(u,r,M);
    for(i=1;i<M+1;i++){
        r[i]=f[i]-r[i];
    }
        
        for(i=0;i<M+2;i++){
            p[i] = r[i];
        }


    // Iterate over Jocobi Iterations
    int iter;
    double diff, largest_diff, ri, world_diff, alpha, beta;
    for(iter=0; iter<itermax; iter++){
        largest_diff = 0;
        //perform a Jacobi iteration, keeping track of largest update difference
        //beginning case
        double rr = vecvec(r,r,M);
        

        matvec(p,ap,M);
        double pap=vecvec(p,ap,M);
        alpha = rr/pap;
        for(i=1;i<M+1;i++){
            u[i]=u[i]+alpha*p[i];
            r[i] = r[i] - alpha*ap[i];
        }
        for(i=1;i<M+1;i++){
            double diff = fabs(ap[i]);
            if(diff>largest_diff)
                largest_diff=diff;
        }       
        
        //find largest update difference and break if below tol
        MPI_Allreduce(&largest_diff, &world_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(world_diff<tol){break;}
        double nrr = vecvec(r,r,M);
        beta = nrr/rr;
        for(i=1;i<M+1;i++){
            p[i] = r[i]+beta*p[i];
        }
        //printf("B,A %f, %f\n",beta,alpha);
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
        for(i=1;i<M+1;i++){
        printf("%f\n",u[i]);
        }
        printf("%d\n",iter);
        printf("%.19g\n",world_diff);
        printf("%.19g\n",world_error);
    }

    MPI_Finalize();
return 0;
}
