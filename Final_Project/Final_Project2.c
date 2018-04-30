#include "hmwk3.h"
#include <demo_util.h>
#include <mpi.h>
#include <math.h>
#include <stdio.h>

// Define the function

double foo(double x, double y){
    return exp(x*y)*(x*x+y*y);
}

double exact(double x, double y){
    return exp(x*y);
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
    double tol,t;

    // Read command line

    read_int(argc, argv, "-n", &N, &err);        
    read_int(argc, argv, "--itermax", &itermax, &err);
    read_double(argc, argv, "--tol", &tol, &err);
 

    // Set intervals
    
    M=N/nprocs;
    double sintvl = 1.0/nprocs;
    double a = sintvl*rank;
    double h = 1.0/N;

    // Initialize global vectors

    double u[N][N];
    double f[N-2][N-2];
    int i,j;
    for(i=0; i<N+1; i++){
        for(j=0; j<N+1; j++){
	        if(i==0 || j==0){
	            u[i][j] = 1;
	        }else if(i==N || j==N){
	            u[i][j] = exp(t/N);
	        }else{
	            u[i][j] = 0;
	        }
	        if (i!=0 && j!=0 && i!=N && j!=N){
                f[i][j] = h*h*foo(a+i*h,j*h);
	        }
        }
    }

    int vBound1 = rank*M;
    int vBound2 = (rank+1)*M;
	double v[vBound2+1][M];
    double prevV[vBound2+1][M];
    // Send each processor their initial values
    if(rank == 0){
	    //take u vector from 0 to ..
		    //if nprocs = 1........(rank+1)*M -1
                //if nprocs > 1.......(rank+1)*M + 1
	    if(nprocs ==1){
	        double f[vBound2 -2][M-2];
	        for (int i=0; i< vBound2; i++){
		        for (int j=0; j< M; j++){
		            v[i][j] = u[i][j];
		            prevV[i][j] = 0;
		            if (i!=0 && j!=0 && i!=vBound2-1 && j!=M-1){
		                f[i][j] = 0;
		            }
		        }
	        }
        }else{
	        double f[vBound2 -1][M-2];
	        for (int i=0; i< vBound2 +1; i++){
		        for (int j=0; j< M; j++){
		            v[i][j] = u[i][j];
		            prevV[i][j] = 0;
		            if (i!=0 && j!=0 && i!=vBound2-1 && j!=M-1){
		                f[i][j] = 0;
		            }
		        }
	        }
        }
    }else if(rank == nprocs -1){
        //take u vector from rank*M -1 to (rank+1)*M -1
        double f[M-2][M-2];
        for (int i = vBound1 -1; i< 2*vBound1+M+1; i++){
            for (int j=0; j< M; j++){
	            v[i-vBound1 +1][j] = u[i][j];
	            prevV[i-vBound1 +1][j] = 0;
	            if (i!=vBound1 -1 && j!=0 && i!=2*vBound1+M-1 && j!=M-1){
	                f[i][j] = 0;
	            }
            }
        }
    }else{
        //take u vector from rank*M -1 to (rank+1)*M +1
        double f[M][M-2];
        for (int i = vBound1 -1; i< 2*vBound1+M; i++){
            for (int j=0; j< M; j++){
	            v[i-vBound1 +1][j] = u[i][j];
	            prevV[i][j] = 0;
	            if(i!=vBound1 -1 && j!=0 && i!=2*vBound1+M -1 && j!=M -1){
	                f[i][j] = 0;
	            }
            }
        }
    }


    // Iterate over Jacobi Iterations
    int iter;
    double diff, largest_diff, ri, world_diff;
    for(iter=1; iter<itermax; iter++){
        largest_diff = 0;
        
        if(rank == 0){
	    //copy v vector from 1 to ..
		    //if nprocs = 1........rank*M -1
                    //if nprocs > 1.......(rank+1)*M + 1
	
	        if(nprocs ==1){
	            for (int i=1; i< vBound2 -1; i++){
		            for (int j=1; j< M; j++){
		                prevV[i][j] = v[i][j];
		            }
	            }
            }else{
	            for(int i=1; i< vBound2 +1; i++){
		            for (int j=1; j< M; j++){
		                prevV[i][j] = v[i][j];
		            }
	            }
            }
	    }else if(rank == nprocs -1){
	        //copy v vector from rank*M -1 to (rank+1)*M -2
	        for (int i = vBound1 -1; i< 2*vBound1+M; i++){
	            for (int j=1; j< M; j++){
		            prevV[i][j] = v[i][j];
	            }
	        }
	    }else{
	        double v[M+2][M];
            double prevV[M+2][M];
	        //copy v vector from rank*M -1 to (rank+1)*M +1
	        for (int i = vBound1 -1; i< 2*vBound1+M; i++){
	            for (int j=0; j< M+1; j++){
		            prevV[i][j] = v[i][j];
	            }
	        }
	    }

        //perform a Jacobi iteration, keeping track of largest difference

        if(rank == 0){
	        if(nprocs ==1){
	            for (int i=1; i< vBound2 -1; i++){
		            for (int j=1; j< M; j++){
			            u[i][j] = -0.25*(f[i][j]-prevV[i-1][j]-prevV[i+1][j]-prevV[i][j-1]-prevV[i][j+1]);
                    	diff = fabs(u[i][j] - prevV[i][j]);
                    	if(diff>largest_diff){
                        	    largest_diff = diff;
		                }
		            }
	            }
            }else{
	            for(int i=1; i< vBound2; i++){
		            for (int j=1; j< M; j++){
		        	    u[i][j] = -0.25*(f[i][j]-prevV[i-1][j]-prevV[i+1][j]-prevV[i][j-1]-prevV[i][j+1]);
                    	diff = fabs(v[i][j] - prevV[i][j]);
                    	if(diff>largest_diff){
                        	    largest_diff = diff;
		                }
		            }
	            }
            }
	    }else if(rank == nprocs -1){
	    //copy v vector from rank*M -1 to (rank+1)*M -2
	        for (int i = vBound1 +1; i< 2*vBound1+M; i++){
	            for (int j=1; j< M; j++){
		            u[i][j] = -0.25*(f[i][j]-prevV[i-1][j]-prevV[i+1][j]-prevV[i][j-1]-prevV[i][j+1]);
                    diff = fabs(v[i][j] - prevV[i][j]);
                    if(diff>largest_diff){
                        largest_diff = diff;
		            }
	            }
	        }
	    }else{
	        //copy v vector from rank*M -1 to (rank+1)*M +1
	        for (int i = vBound1; i< 2*vBound1+M -1; i++){
	            for (int j=0; j< M+1; j++){
		            u[i][j] = -0.25*(f[i][j]-prevV[i-1][j]-prevV[i+1][j]-prevV[i][j-1]-prevV[i][j+1]);
                    diff = fabs(v[i][j] - prevV[i][j]);
                    if(diff>largest_diff){
                        largest_diff = diff;
		            }
	            }
	        }
	    }

        //find largest difference and break if below tol
        MPI_Allreduce(&largest_diff, &world_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(world_diff<tol){break;}
      
        //Synchronize end conditions
        MPI_Request left_request,right_request;
	    if (nprocs !=1){
	        if (rank == 0){
	            for(j=1; j< M; j++){
		            //send right boundary
		            MPI_Isend(&(u[(rank+1)*M -1][j]), 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD,&left_request);
                    //receive left boundary
                    MPI_Recv(&(u[(rank+1)*M +1][j]), 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		        }
	        }else if (rank == nprocs -1){
	            for(j=1; j< M; j++){
		        //send left boundary
		        MPI_Isend(&(u[rank*M +1][j]), 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD,&left_request);
                //receive right boundary
                MPI_Recv(&(u[rank*M -1][j]), 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		        }
	        }else{
	            for(j=1; j< M; j++){
		            //send right boundary
		            MPI_Isend(&(u[(rank+1)*M -1][j]), 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD,&left_request);
                    //receive left boundary
                    MPI_Recv(&(u[(rank+1)*M +1][j]), 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		            //send left boundary
		            MPI_Isend(&(u[rank*M +1][j]), 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD,&left_request);
                    //receive right boundary
                    MPI_Recv(&(u[rank*M -1][j]), 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	            }
	        }
	    }
    }

// Send processor vectors back to global vector
    if(rank ==0){
	    if(nprocs ==1){
	        for(int i=1; i< vBound2 -1; i++){
		        for(int j=1; j< M-1; j++){
		            u[i][j] = v[i][j];
		        }
	        }
        }else{
	        for(int i=1; i< vBound2 +1; i++){
		        for (int j=1; j< M-1; j++){
		            u[i][j] = v[i][j];
		        }
	        }
        }
    }else if(rank == nprocs -1){
        //take u vector from rank*M -1 to (rank+1)*M -1
        for(int i = vBound1 +1; i< 2*vBound1+M; i++){
            for(int j=0; j< M; j++){
                u[i][j] = v[i-vBound1 -1][j];
            }
        }
    }else{
        //take u vector from rank*M -1 to (rank+1)*M +1
        for(int i = vBound1 -1; i< 2*vBound1+M +2; i++){
            for(int j=0; j< M; j++){
                u[i][j] = v[i-vBound1 -1][j];
            }
        }
    }


    double max_error=0;
    for(i=1;i<M;i++){
        for(j=0;j<M+1;j++){
            double error = fabs(u[i][j]-exact(a+i*h,j*h));
            if(error>max_error)
                max_error=error;
        }
    }
    double world_error;
    MPI_Allreduce(&max_error, &world_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(rank==0){
        for(i=0; i<M+1; i++){
            for(j=0; j<M+1; j++){
                printf("%.19g\n",u[i][j]);
            }
        }
        printf("%d\n",iter);
        printf("%.19g\n",world_diff);
        printf("%.19g\n",world_error);
    }

    MPI_Finalize();
return 0;
}
