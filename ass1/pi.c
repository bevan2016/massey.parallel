#include "mpi.h"
#include <stdio.h>

//2**15
#define SHORT_1_VALUE 32768
// 2**16
#define SHORT_VALUE 65536     
// 2**32
#define INT_VALUE   4294967296 

#define RANDOM_SEED 31

/*
 * coefficient for calculate random number.
 */
unsigned long AC_Table[32][2] = {  //see http://stream.massey.ac.nz/mod/resource/view.php?id=2370031
	{1664525, 1013904223},    //1 
	{389569705, 1196435762},  //2
	{0, 0},                   //3 placeholder, not used
	{158984081, 2868466484},  //4
	{0, 0},                   //5 
	{0, 0},                   //6 
	{0, 0},                   //7 
	{3934847009, 2748932008}, //8
    {0, 0},                   //9
	{0, 0},                   //10
	{0, 0},                   //11
	{0, 0},                   //12
	{0, 0},                   //13
	{0, 0},                   //14
	{0, 0},                   //15
	{2001863745, 2210837584}, //16
    {0, 0},                   //17
	{0, 0},                   //18
	{0, 0},                   //19
	{0, 0},                   //20
	{0, 0},                   //21
	{0, 0},                   //22
	{0, 0},                   //23
    {0, 0},                   //24
	{0, 0},                   //25
	{0, 0},                   //26
	{0, 0},                   //27
	{0, 0},                   //28
	{0, 0},                   //29
	{0, 0},                   //30
    {0, 0},                   //31
	{666245249, 95141024}     //32
};

/* 
 * test if the coordinate represented by n falls within the circle.
 * return 1 if in the circle, 0 if not
 */
int hit_circle (n)unsigned long n;
{
    //calculate (x, y) for n
    double x, y;
    unsigned long ix, iy;
    ix = n % SHORT_VALUE;
    iy = n / SHORT_VALUE;
    x = ix*1.0 / SHORT_1_VALUE - 1;
    y = iy*1.0 / SHORT_1_VALUE - 1;
    return  (x*x + y*y <= 1) ? 1:0;
}

/*
 * calculate the next random number from current n.
 */
unsigned long random_nk(n, A, C)unsigned long n;unsigned long A;unsigned long C;
{
    return (A*n+C) % INT_VALUE;
}

/*
 * test how many points starting from n0 falls in the circle.
 */
unsigned long hit_number(n0, count, A, C)unsigned long n0;unsigned long count;unsigned long A;unsigned long C;
{
    unsigned long hit, n1,  i;
    if (count < 1 )
    {
        return 0L;
    }

    hit = hit_circle(n0);
    for(i=1; i<count; i++){
         n1 = random_nk(n0, A, C);
         hit += hit_circle(n1);
         n0 = n1;
    } 
    return hit;
}

void test_comm_time(myid, numproc)int myid;int numproc;
{
    int i;
    unsigned long data_send, data_recv;
    double time_start, time_elapsed;
    MPI_Status Stat;

    if (myid == 0)
    {
        data_send = 3934847009;
           
        //time_start = MPI_Wtime();   
        for (i=1; i<numproc; i++)
	    {  
             time_start = MPI_Wtime();
             MPI_Send(&data_send, 1, MPI_LONG, i, 0, MPI_COMM_WORLD);
	         MPI_Recv(&data_recv, 1, MPI_LONG, i, 0, MPI_COMM_WORLD, &Stat);
             time_elapsed = MPI_Wtime() - time_start;
             fprintf(stdout, "It takes %f to communicate with %d\n", time_elapsed, i);
	    }
    }
    else
    {
        time_start = MPI_Wtime();   
        MPI_Recv(&data_recv, 1, MPI_LONG, i, 0, MPI_COMM_WORLD, &Stat);
        MPI_Send(&data_recv, 1, MPI_LONG, i, 0, MPI_COMM_WORLD);
        time_elapsed = MPI_Wtime() - time_start;
        //fprintf(stdout, "Slave %d spent %f.\n", myid, time_elapsed);
    }

}
//change the number of the processors (1, 2, 4, 8)
//change the value of the N (1m, 2m, 4m, 8m, 16m)
int main(argc,argv)int argc;char *argv[]; 
{
    int numproc, myid, namelen;
    unsigned long total, hit, n0, n1, i;
    unsigned long N, A, C, task_n;
    double work_time_start, work_time, total_time_start, total_time;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status Stat; //status variable, so operations can be checked
  
    MPI_Init(&argc,&argv);//INITIALIZE
    MPI_Comm_size(MPI_COMM_WORLD, &numproc); //how many processors

    //check if the process number is supported for leap-frog algorithm
    if (numproc < 1 || numproc > 32 || AC_Table[numproc-1][0] == 0) {
        fprintf(stdout, "Processor number %d is not supported. Exiting...\n", numproc);
        MPI_Finalize();
  	return 0;
    }
  
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);    //what is THIS processor-ID 
    MPI_Get_processor_name(processor_name, &namelen); //what is THIS processor name (hostname)
   
    A = AC_Table[numproc-1][0];
    C = AC_Table[numproc-1][1];
    N = atoi(argv[1]);
    //if N is zero, just test the communication time
    if (N == 0)
    {
        test_comm_time(myid, numproc);
        MPI_Finalize();
        return 0;
    }  

    task_n = N / numproc; //ignore the remainder if there is as N is to be very big

    total = 0; //the total number that falls in the circle
    if (myid == 0) // if master
    {   
         fprintf(stdout, "*************Test for %d processor*************\n", numproc); 
         fprintf(stdout, "Processor: %s\n", processor_name); 
         fprintf(stdout, "N=%ld, A=%ld, C=%ld\n", N, A, C);
         total_time_start = MPI_Wtime();   
         n0 = RANDOM_SEED; 
         // Master sends N to all the slave processes
         for  (i=1; i<numproc; i++)
         {
             n1 = random_nk(n0, AC_Table[0][0], AC_Table[0][1]); //k is 1
             MPI_Send(&n1, 1, MPI_LONG, i, 0, MPI_COMM_WORLD);
             n0 = n1;
         }	

         // Master does its own task
         work_time_start = MPI_Wtime();
         total = hit_number(RANDOM_SEED, task_n, A, C);
         work_time = MPI_Wtime() - work_time_start;

         //check all slaves
         for (i=1;i<numproc;i++)
         {
             MPI_Recv(&hit, 1, MPI_LONG, i, 0, MPI_COMM_WORLD, &Stat);
             total += hit;
         }
         total_time = MPI_Wtime() - total_time_start;
         fprintf(stdout,"Master time: [total, work, comm] = [%f, %f, %f]\n", total_time, work_time, total_time-work_time);

         //result
         fprintf(stdout,"\n");
         fprintf(stdout,"The %ld of %ld points falls into the circle\n", total, N);
         fprintf(stdout,"PI is %f\n", total*4.0/N);
         fprintf(stdout,"\n");

     } 
     else //this is slave
     {
         total_time_start = MPI_Wtime();
         MPI_Recv(&n0, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD, &Stat);

         work_time_start = MPI_Wtime();
         hit = hit_number(n0, task_n, A, C);
         work_time = MPI_Wtime() - work_time_start;

         MPI_Send(&hit, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
         total_time = MPI_Wtime() - total_time_start;
         fprintf(stdout, "\n");
         fprintf(stdout, "Slave %ld: hit/total is %ld/%ld\n", myid, hit, task_n);
         fprintf(stdout, "Slave %ld time: [total, work] = [%f, %f]\n", myid, total_time, work_time);
         fprintf(stdout, "\n");
     }

     MPI_Finalize();
}
