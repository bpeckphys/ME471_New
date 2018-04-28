#include "hmwk3.h"
#include <demo_util.h>

#include <mpi.h>
#include <math.h>

#define PI 3.14159265358979323846264338327

static
void fill_ghost(double *u, int m, int my_rank);

double utrue(double x)
{
    double u;
    double pi2;
    pi2 = 2*PI;
    u = cos(pi2*x);
    return u;
}

double rhs(double x)
{
    double fx;
    double pi2;
    pi2 = 2*PI;
    fx = -(pi2)*(pi2)*cos(pi2*x);
    return fx;
}


void main(int argc, char** argv)
{
    /* Data arrays */
    double a,b;
    int n_global;
    double range[2];
    double *x, *F, *B;

    /* Iterative variables */
    double tol;
    int kmax;

    /* Misc variables */
    int i,j,k;

    /* MPI variables */
    int my_rank, nprocs;

    /* ----------------------------------------------------------------
       Set up MPI
     ---------------------------------------------------------------- */

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    set_rank(my_rank);  /* Used in printing */
    read_loglevel(argc,argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ----------------------------------------------------------------
       Read parameters from the command line
     ---------------------------------------------------------------- */
    if (my_rank == 0)
    {        
        int m,err,loglevel;
        read_int(argc,argv, "-m", &m, &err);
        if (err > 0)
        {
            print_global("Command line argument '-m' not found\n");
            exit(0);
        }        

        read_int(argc,argv, "--kmax", &kmax, &err);
        if (err > 0)
        {
            print_global("Command line argument '--kmax' not found\n");
            exit(0);
        }

        read_double(argc,argv, "--tol", &tol, &err);
        if (err > 0)
        {
            print_global("Command line argument '--tol' not found\n");
            exit(0);
        }

        n_global = pow2(m);     

        /* Hardwire domain values values */
        a = 0;
        b = 1;  
    }

    /* ---------------------------------------------------------------
       Broadcast global information : kmax, tol, n_global
         -- compute range, h, and number of intervals for each
            processor
    --------------------------------------------------------------- */
        
    /* TODO : .... */
    int root = 0;
    MPI_Bcast(&tol,1,MPI_DOUBLE,root,MPI_COMM_WORLD);
    MPI_Bcast(&a,1,MPI_DOUBLE,root,MPI_COMM_WORLD);
    MPI_Bcast(&b,1,MPI_DOUBLE,root,MPI_COMM_WORLD);
    MPI_Bcast(&n_global,1,MPI_DOUBLE,root,MPI_COMM_WORLD);
    MPI_Bcast(&kmax,1,MPI_DOUBLE,root,MPI_COMM_WORLD);

    double w = (b-a)/nprocs;
    double h = (b-a)/n_global;
    int m = n_global/nprocs; // m is number of subintervals

    range[0] = my_rank*w;
    range[1] = range[0]+w;

    /* ---------------------------------------------------------------
       Set up right hand side and any inhomogenous boundary conditions
    --------------------------------------------------------------- */
    zeros_array(10,&B);  /* Change 10 to proper size */
    zeros_array(m+2,&F);
    linspace_array(range[0]-h,range[1]+h,m+3,&x);

    for(j = 0; j < m+1; j++){
        F[j] = rhs(x[j]);  // Include boundary Conditions
    }

    /* TODO : .... */

    /* ----------------------------------------------------------------
       Set up arrays and other vars needed for iterative method
    ---------------------------------------------------------------- */

    /* TODO : .... */


    /* ----------------------------------------------------------------
       Start iterations
    ---------------------------------------------------------------- */

    /* TODO : .... */

    /* ----------------------------------------------------------------
       Calculate error and report results
    ---------------------------------------------------------------- */

    /* TODO : .... */

    /* ----------------------------------------------------------------
       Clean up
    ---------------------------------------------------------------- */
    delete_array(&B);
    /* etc */

    MPI_Finalize();
}


/* ----------------------------------------------------------------
   Fill ghost cell values
---------------------------------------------------------------- */
static
void fill_ghost(double *u, int m, int my_rank)
{
    int tag, source, dest;
    int nprocs;
    MPI_Status status;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* Fill ghost cells in u[0], u[m+2] */

    /* ...... */

}
