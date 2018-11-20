/* 159.735 Semester 2, 2016.  Ian Bond, 3/10/2016

 Sequential version of the N-sphere counting problem for Assignment
 5. Two alternative algorithms are presented.

 Note: a rethink will be needed when implementing a GPU version of
 this. You can't just cut and paste code.

 To compile: g++ -O3 -o nsphere nsphere.cpp
 (you will get slightly better performance with the O3 optimization flag)
*/
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <string>
#include <cuda.h>
#include <vector>

const long MAXDIM = 10;
const double RMIN = 2.0;
const double RMAX = 8.0;
const int MAX_POINTS_PER_THREAD = 500;
const int MAX_BPG_ONE_DIM = 1024;
const int MAX_TPB = 1024;

double diffclock(clock_t clock1, clock_t clock2)
{
	double diffticks = clock1 - clock2;
	double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
	return diffms; // Time difference in milliseconds
}

/*
 * Evaluate n**k where both are long integers
 */
long powlong(long n, long k)
{
	long p = 1;
	for (long i = 0; i < k; ++i) p *= n;
	return p;
}

/*
 * Convert a decimal number into another base system - the individual
 * digits in the new base are stored in the index array.
 */
void convert(long num, long base, std::vector<long>& index)
{
	const long ndim = index.size();
	for (long i = 0; i < ndim; ++i) index[i] = 0;
	long idx = 0;
	while (num != 0) {
		long rem = num % base;
		num = num / base;
		index[idx] = rem;
		++idx;
	}
}

long count_in_v1(long ndim, double radius)
{
	const long halfb = static_cast<long>(floor(radius));
	const long base = 2 * halfb + 1;
	const double rsquare = radius * radius;

	// This is the total number of points we will need to test.
	const long ntotal = powlong(base, ndim);
	std::cout << "Points need to be test " << ntotal << std::endl;
	long count = 0;

	// Indices in x,y,z,.... 
	std::vector<long> index(ndim, 0);

	// Loop over the total number of points. For each visit of the loop,
	// we covert n to its equivalent in a number system of given "base".
	for (long n = 0; n < ntotal; ++n) {
		convert(n, base, index);
		double rtestsq = 0;
		for (long k = 0; k < ndim; ++k) {
			double xk = index[k] - halfb;
			rtestsq += xk * xk;
		}

		if (rtestsq < rsquare) ++count;
	}

	return count;
}

// kernel
__global__ void cuda_count(int ndim, double radius, long nfrom, long nto, long nthreads, int* counter)
{
	long id = blockIdx.x * blockDim.x + threadIdx.x;
	counter[id] = 0;
	if (id >= nto)
		return;

	const long halfb = static_cast<long>(floor(radius));
	const long base = 2 * halfb + 1;
	const double rsquare = radius*radius;
	
	long index = 0;
	long num = nfrom + id;
	//a thread might test more than one numbers
	while (num < nto)
	{
		double rtestsq = 0;
		
		for (int i=0; i<ndim; i++)
		{
			long rem = num % base;
			num = num / base;
			double xk = rem - halfb;
			rtestsq += xk * xk;
		}
		
		if (rtestsq < rsquare )
		{
			atomicAdd(&counter[id], 1);
		}
		
		index++;
		num = nfrom + id + nthreads*index;	
	}
}

long count_in_cuda(long ndim, double radius)
{
	const long halfb = static_cast<long>(floor(radius));
	const long base = 2 * halfb + 1;
	// This is the total number of points we will need to test.
	const long ntotal = powlong(base, ndim);
	const int tpb_x = (ntotal<MAX_TPB)?ntotal:MAX_TPB;

	//use maximum MAX_BPG_ONE_DIM x 1024 threads
	int blocks = ntotal / MAX_TPB + 1;
	if (blocks >  MAX_BPG_ONE_DIM)
	{
		blocks = MAX_BPG_ONE_DIM;
	}
	const long nthreads = tpb_x*blocks;
	int* counters = new int[nthreads];
	memset(counters, 0, sizeof(int)*nthreads);
	int* d_counters;
	cudaMalloc(&d_counters, sizeof(int)*nthreads);

	long total_count = 0;
	//invoke the kernel
	//std::cout << "Launching a grid of " << nthreads << " threads" << std::endl;
	const long points_for_each_call = MAX_POINTS_PER_THREAD*nthreads;
	long nfrom = 0; 
	long nto = points_for_each_call;
	do
	{
		if (nto > ntotal)
			nto = ntotal;

		//std::cout << "will handle [" << nfrom << ", " << nto << "]\n";

		cuda_count <<<blocks, tpb_x>>>(ndim, radius, nfrom, nto, nthreads, d_counters); 
		cudaError err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cout << "CUDA kernel error:\n"<<cudaGetErrorString(err)<<std::endl;
			break;
		}
		//copy the counters to host
		cudaMemcpy(counters, d_counters, sizeof(int)*nthreads, cudaMemcpyDeviceToHost);
		//sum all counters
		for (long i = 0; i < nthreads; i++)
		{
			total_count += counters[i];
		}

		nfrom = nto;
		nto += points_for_each_call;
	}while (nfrom < ntotal);

	cudaFree(d_counters);
	delete[] counters;

	return total_count;
}

int main(int argc, char* argv[])
{
	// You can make this larger if you want
	const long ntrials = 20;
	std::cout <<"r  nd   Seq Count   Seq Time   cuda Count   tcuda Time"<<std::endl;
	for (long n = 0; n < ntrials; ++n) 
	{
		// Get a random value for the hypersphere radius between the two limits
		const double r = drand48() * (RMAX - RMIN) + RMIN;

		// Get a random value for the number of dimensions between 1 and MAXDIM inclusive
		const long  nd = lrand48() % (MAXDIM - 1) + 1;

		clock_t tstart = clock();
		const long count_s = count_in_v1(nd, r);
		double ts = diffclock(clock(), tstart);
		//std::cout << "Counted by sequential is "<< count_s << std::endl;
		
		tstart = clock();
		const long count_cuda = count_in_cuda(nd, r);
		double tp = diffclock(clock(), tstart);
		//std::cout << "Counted by CUDA is " << count_cuda << std::endl<<std::endl;
		
		std::cout << r << "\t " << nd << "\t" << count_s << "\t" << ts <<"\t"<< count_cuda << "\t"<< tp <<std::endl;	
	}

}

