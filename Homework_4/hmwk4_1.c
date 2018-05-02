#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <demo_util.h>
#include <string.h>

typedef struct{
   char suits[2][10]; /* "spades","hearts","diamonds","clubs"            */
   char values[2][3]; /* "A", "2", "3", ... ,"10", "J", "Q", "K"        */
   int total;         /* Sum of two cards                               */
   float bet;         /* between $1.00 and $100.00, in inc. of $0.50$   */
} struct_player_t;

void build_domain_type(MPI_Datatype *player_t){
    int block_lengths[4] = {1,1,1,1};

    /* Set up types */
    MPI_Datatype typelist[4];
    typelist[0] = MPI_CHAR;
    typelist[1] = MPI_CHAR;
    typelist[2] = MPI_INT;
    typelist[3] = MPI_FLOAT;

    /* Set up displacements */
    MPI_Aint disp[4];
    disp[0] = offsetof(struct_player_t,suits);
    disp[1] = offsetof(struct_player_t,values);
    disp[2] = offsetof(struct_player_t,total);
    disp[3] = offsetof(struct_player_t,bet);

    MPI_Type_create_struct(4,block_lengths, disp, typelist, player_t);
    MPI_Type_commit(player_t);
}

void main(int argc, char** argv){
    /* File I/O */
    MPI_File   file;
    MPI_Status status;
    
    /* Data type */
    MPI_Datatype player_t;
    MPI_Datatype card_as_string;
    MPI_Datatype localarray;
    
    // Initialize MPI
    int my_rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    set_rank(my_rank);  /* Used in printing */
    read_loglevel(argc,argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int i, loglevel;
    int nc[2];
    int c[52];
    char ca[3];
    char cb[3];
    char s[3];
    char nc1, nc2;
    
    build_domain_type(&player_t);
    
    // Node 0 creates shuffled deck
    for(i=0;i<=1;i++){
        nc[i] = 0;
    }
    if(my_rank==0){
        srand(time(0));
        for(i=0;i<=51;i++){
            c[i] = (rand()%51+1);
            //printf("%i\n", c[i]);
        }
    }
    
    // Node 0 deals 2 cards to each processor using MPI_Scatter
    MPI_Scatter(&c,2,MPI_INT,&nc,2,MPI_INT,0,MPI_COMM_WORLD);
    print_debug("Node %i, First card %i\n", my_rank, nc[0]);
    print_debug("Node %i, Second card %i\n", my_rank, nc[1]);
    
    struct_player_t bets;
    
    nc1 = nc[0]%13;
    nc2 = nc[1]%13;
    
    // Each processor determines suit and card value. (Ace=1,face cards=10,others=face value)
    // Each processor then provides information to fill out the provided C-struct.
    if((nc[0]%13)==1){
        print_debug("Node %i has an ace for first card\n",my_rank);
        if(nc[0]==1){
            strcpy(ca, "AC");
            print_debug("Node %i has an ace of Clubs ca = %s\n",my_rank,ca);
        }else if(nc[0]==14){
            strcpy(ca, "AD");
            print_debug("Node %i has an ace of Diamonds ca = %s\n",my_rank,ca);
        }else if(nc[0]==27){
            strcpy(ca, "AH");
            print_debug("Node %i has an ace of Hearts ca = %s\n",my_rank,ca);
        }else if(nc[0]==40){
            strcpy(ca, "AS");
            print_debug("Node %i has an ace of Spades ca = %s\n",my_rank,ca);
        }
    }else if((nc[0]%13)>1 & (nc[0]%13)<11){
        print_debug("Node %i has a %i for first card\n",my_rank,nc[0]);
        if(nc[0]>1 & nc[0]<11){
            sprintf(s,"%i", nc1);
            strcpy(ca, strcat(s,"C"));
            print_debug("Node %i has a %i of Clubs ca = %s\n",my_rank,nc[0],ca);
        }else if(nc[0]>14 & nc[0]<24){
            sprintf(s,"%i", nc1);
            strcpy(ca, strcat(s,"D"));
            print_debug("Node %i has a %i of Diamonds ca = %s\n",my_rank,nc[0]%13,ca);
        }else if(nc[0]>27 & nc[0]<37){
            sprintf(s,"%i", nc1);
            strcpy(ca, strcat(s,"H"));
            print_debug("Node %i has a %i of Hearts ca = %s\n",my_rank,nc[0]%13,ca);
        }else if(nc[0]>40 & nc[0]<50){
            sprintf(s,"%i", nc1);
            strcpy(ca, strcat(s,"S"));
            print_debug("Node %i has a %i of Spades ca = %s\n",my_rank,nc[0]%13,ca);
        }
    }else if((nc[0]%13)==11){
        print_debug("Node %i has a jack for first card\n",my_rank);
        if(nc[0]==11){
            strcpy(ca, "JC");
            print_debug("Node %i has a jack of Clubs ca = %s\n",my_rank,ca);
        }else if(nc[0]==24){
            strcpy(ca, "JD");
            print_debug("Node %i has a jack of Diamonds ca = %s\n",my_rank,ca);
        }else if(nc[0]==37){
            strcpy(ca, "JH");
            print_debug("Node %i has a jack of Hearts ca = %s\n",my_rank,ca);
        }else if(nc[0]==50){
            strcpy(ca, "JS");
            print_debug("Node %i has a jack of Spades ca = %s\n",my_rank,ca);
        }
    }else if(((nc[0]%13)==12)){
        print_debug("Node %i has a queen for first card\n",my_rank);
        if(nc[0]==12){
            strcpy(ca, "QC");
            print_debug("Node %i has a queen of Clubs ca = %s\n",my_rank,ca);
        }else if(nc[0]==25){
            strcpy(ca, "QD");
            print_debug("Node %i has a queen of Diamonds ca = %s\n",my_rank,ca);
        }else if(nc[0]==38){
            strcpy(ca, "QH");
            print_debug("Node %i has a queen of Hearts ca = %s\n",my_rank,ca);
        }else if(nc[0]==51){
            strcpy(ca, "QS");
            print_debug("Node %i has a queen of Spades ca = %s\n",my_rank,ca);
        }
    }else if(((nc[0]%13)==0)){
        print_debug("Node %i has a king for first card\n",my_rank);
        if(nc[0]==13){
            strcpy(ca, "KC");
            print_debug("Node %i has a king of Clubs ca = %s\n",my_rank,ca);
        }else if(nc[0]==26){
            strcpy(ca, "KD");
            print_debug("Node %i has a king of Diamonds ca = %s\n",my_rank,ca);
        }else if(nc[0]==39){
            strcpy(ca, "KH");
            print_debug("Node %i has a king of Hearts ca = %s\n",my_rank,ca);
        }else if(nc[0]==52){
            strcpy(ca, "KS");
            print_debug("Node %i has a king of Spades ca = %s\n",my_rank,ca);
        }
    }
    
    if(((nc[1]%13)==1)){
        print_debug("Node %i has an ace for second card\n",my_rank);
        if(nc[1]==1){
            strcpy(cb, "AC");
            print_debug("Node %i has an ace of Clubs cb = %s\n",my_rank,cb);
        }else if(nc[1]==14){
            strcpy(cb, "AD");
            print_debug("Node %i has an ace of Diamonds cb = %s\n",my_rank,cb);
        }else if(nc[1]==27){
            strcpy(cb, "AH");
            print_debug("Node %i has an ace of Hearts cb = %s\n",my_rank,cb);
        }else if(nc[1]==40){
            strcpy(cb, "AS");
            print_debug("Node %i has an ace of Spades cb = %s\n",my_rank,cb);
        }
    }else if(((nc[1]%13)>1 & (nc[1]%13)<11)){
        print_debug("Node %i has a %i for second card\n",my_rank,nc[1]);
        if(nc[1]>1 & nc[1]<11){
            sprintf(s,"%i", nc2);
            strcpy(cb, strcat(s,"C"));
            print_debug("Node %i has a %i of Clubs cb = %s\n",my_rank,nc[1],cb);
        }else if(nc[1]>14 & nc[1]<24){
            sprintf(s,"%i", nc2);
            strcpy(cb, strcat(s,"D"));
            print_debug("Node %i has a %i of Diamonds cb = %s\n",my_rank,nc[1]%13,cb);
        }else if(nc[1]>27 & nc[1]<37){
            sprintf(s,"%i", nc2);
            strcpy(cb, strcat(s,"H"));
            print_debug("Node %i has a %i of Hearts cb = %s\n",my_rank,nc[1]%13,cb);
        }else if(nc[1]>40 & nc[1]<50){
            sprintf(s,"%i", nc2);
            strcpy(cb, strcat(s,"S"));
            print_debug("Node %i has a %i of Spades cb = %s\n",my_rank,nc[1]%13,cb);
        }
    }else if(((nc[1]%13)==11)){
        print_debug("Node %i has a jack for second card\n",my_rank);
        if(nc[1]==11){
            strcpy(cb, "JC");
            print_debug("Node %i has a jack of Clubs cb = %s\n",my_rank,cb);
        }else if(nc[1]==24){
            strcpy(cb, "JD");
            print_debug("Node %i has a jack of Diamonds cb = %s\n",my_rank,cb);
        }else if(nc[1]==37){
            strcpy(cb, "JH");
            print_debug("Node %i has a jack of Hearts cb = %s\n",my_rank,cb);
        }else if(nc[1]==50){
            strcpy(cb, "JS");
            print_debug("Node %i has a jack of Spades cb = %s\n",my_rank,cb);
        }
    }else if(((nc[1]%13)==12)){
        print_debug("Node %i has a queen for second card\n",my_rank);
        if(nc[1]==12){
            strcpy(cb, "QC");
            print_debug("Node %i has a queen of Clubs cb = %s\n",my_rank,cb);
        }else if(nc[1]==25){
            strcpy(cb, "QD");
            print_debug("Node %i has a queen of Diamonds cb = %s\n",my_rank,cb);
        }else if(nc[1]==38){
            strcpy(cb, "QH");
            print_debug("Node %i has a queen of Hearts cb = %s\n",my_rank,cb);
        }else if(nc[1]==51){
            strcpy(cb, "QS");
            print_debug("Node %i has a queen of Spades cb = %s\n",my_rank,cb);
        }
    }else if(((nc[1]%13)==0)){
        print_debug("Node %i has a king for second card\n",my_rank);
        if(nc[1]==13){
            strcpy(cb, "KC");
            print_debug("Node %i has a king of Clubs cb = %s\n",my_rank,cb);
        }else if(nc[1]==26){
            strcpy(cb, "KD");
            print_debug("Node %i has a king of Diamonds cb = %s\n",my_rank,cb);
        }else if(nc[1]==39){
            strcpy(cb, "KH");
            print_debug("Node %i has a king of Hearts cb = %s\n",my_rank,cb);
        }else if(nc[1]==52){
            strcpy(cb, "KS");
            print_debug("Node %i has a king of Spades cb = %s\n",my_rank,cb);
        }
    }
    MPI_Type_contiguous(21, MPI_CHAR, &card_as_string); 
    MPI_Type_commit(&card_as_string);
    int globalsize = nprocs;
    int localsize = 1;
    int starts = my_rank;
    int order = MPI_ORDER_C;
    int nsize = my_rank < nprocs-1 ? 4 : 5;
    char *text;
    char_array(nsize,&text);
    MPI_Type_create_subarray(1, &globalsize, &localsize, &starts, order, 
                             card_as_string, &localarray);
    MPI_Type_commit(&localarray);
    MPI_File_open(MPI_COMM_WORLD, "text.out", 
                  MPI_MODE_CREATE|MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_Offset offset = 0;
    MPI_File_set_view(file, offset,  MPI_CHAR, localarray, 
                           "native", MPI_INFO_NULL);

    MPI_File_write_all(file, text, localsize, card_as_string,MPI_STATUS_IGNORE);
    MPI_File_close(&file);

    MPI_Type_free(&localarray);
    MPI_Type_free(&card_as_string);
    
    
    MPI_Finalize();
}
