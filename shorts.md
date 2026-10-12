CUDA is not the moat.
You can compile CUDA to AMD with Spectral Compute 's scale-lang.com to run your C/C++ workloads.

AI models can run on AMD using vllm/sglang/llama.cpp with minimal effort.

You can train and fine tune models on AMD using TensorWave , Unsloth AI , ScalarLM and others....

You can access AI models via OpenAI endpoints that don't run on Nvidia 's hardware.

Code can be ported from CUDA to another vendor's stack in under a month with coding agents.

It's no longer the hardware or the software that is the bottleneck. 
So, what's the moat if all these are possible today?

The moat is the idea built-in to the population that CUDA is the moat. 

Good luck fixing that one...

PyTorch uses reshape and view to resize a tensor.
__Flash attention__
Since we are doing matrix multiplication and in matrix multiplication first row of first matrix is multiplied with the first column of the second matrix, then first row of first matrix is multiplied with the second column of the second matrix and so on. So, q1 is the first row of the query matrix and k1 is the first row of the K matrix. We are multiplying without transposing K, so row of Q will get multiplied with row of K. 
When you take a vector from a matrix, like first row of a matrix, in linear algebra the usual notation is we always pretend that a vector is a column vector. So, q1 and K1 both are column vector. So, to keep the notation same as linear algebra, we need to get the result of QKT, so we do for notation q1Tk1 to its transpose.
S=QKT :[q1Tk1 q1Tk2 …. q1Tkn
	q2Tk1 q2Tk2 ……q2Tkn
q2Tk1 q2Tk2 ……q2Tkn]
P=softmax(s) = [a11 a12 … a1n
		      a21 a22 .. .a2n
      an1 an2 .. .ann]
O=PV
The problem is: accessing the GPU global memory (HBM) is very slow compared to its shared memory. However, the shared memory is much smaller than the HBM.
Flash attention paper claims the operation of the attention is I/O bound. It means if we keep accessing the global memory, the overall operation of computing the attention is not because of computing of all these operations, it is because we keep accessing the global memory which is slow. We call this kind of operations I/O bound. 
The only way to improve this situation is to compute the attention inside the shared memory (which is much closer to the core that actually do the computation) of the GPU which is much smaller. So, we need to kind of split attention computation into smaller blocks that can reside in the shared memory. This is possible through block matrix multiplication. In the paper they called it “tiling”. It is a technique used at the time of writing GPU kernels. 
So, flash attention is trying to solve, we do not need to access the GPU global memory (HBM) when computing attention but copying only a part of each matrix inside the local memory i.e., shared memory of the GPU which is closer to the cores. Computing a part of the output matrix there, then copying that part to the HBM and keep doing it for all the blocks in which we can divide these Q, K and V matrices.
The biggest problem in computing this block computation is the softmax computation, which needs to access all the elements of each row of the S matrix to apply the softmax. If you remember, softmax uses a normalization factor which is the sum of exponentials of each values on which it is applied (row-wise). We solve this problem with the help of “Online Softmax”. 

Verify this info and use “
How FlashAttention Handles Block Computation Memory
•	Tiling via SRAM: The GPU loads small blocks of Q, K, and V from the slow HBM into the fast on-chip SRAM. [1]
•	Kernel Fusion: All intermediate operations—such as matrix multiplication (\(QK^{T}\)), scaling, masking, softmax calculation, and the final multiplication by V—happen entirely on-chip inside a single fused GPU kernel. [1]
•	Avoiding HBM Hops: The massive intermediate matrices (S and P) are never written out to HBM. [1]
•	Final Output Only: Only the final accumulated output block (O) is scaled, updated using online softmax statistics, and written back to the HBM once. [1, 2]
__Safe softmax:__
The Problem with Standard Softmax
The standard softmax function turns raw numbers (logits) into probabilities by exponentiating each value and dividing by the sum of all exponentiated values. [1, 2]
•	Numerical instability: Large numbers cause computer overflow errors in standard float formats.
Given a vector x, softmax transforms the vector keeping the dimension of the vector same such that the sum of elements of vector becomes 1. So, each element is calculated as below: so the ith element of the output vector will be exponential of the ith vector divided by sum of exponentials of all the elements of the vector. softmax = e^xi/summation of e^xj
The Dr is called the normalization factor which normalizez all the element of the vector such that their values will become a range bound [0,1]. Why we use softmax? because we want the output distribution to be positive and not negative.
Output of dot product is scalar. Softmax transform each row of the P such that each row of P becomes kind of probability distribution row-wise. It means sum of elements of each row becomes 1.
But there is a problem with softmax. Output of e^100 will not fil fp 32 or fp 16 or an integer of 32-bit. So, we cannot compute as it will overflow our variable that is storing this value during computation. This is called “Numerical instability” in computer science. 
Numerical instability in computer science is a number cannot be represented in fixed representation with the bits we have available. We have 16-bit or 32-bit. We also have 64-bit, but it would be very expensive to use. 
So, to make the softmax computable and numericalyy stable we use safe softmax.
Safe softmax: To fix this, algorithms subtract the maximum value of the vector from each element.
Graph of e^x when -inf < x =< 0, lies between 0 and 1.
Pseudocode:
m0 = -inf
for i=1 to n:
mi = max(mi-1,xi)
l0 = 0
for j=1 to n:
lj = lj-1 + e^(xj – mn) 
for k=1 to n:
xk   e^(xk – mn)/ln
Add a small vector example of [4, 1, 7, 2] to demonstrate safe softmax operation.


__Online SoftMax__:
Problem with Safe softmax:
Memory bottleneck: Safe softmax traditionally requires three separate passes over the data in memory (one to find the max, one to sum the exponents, and one to compute the final values), which slows down GPU processing. [1, 2, 3, 4]
To apply the softmax to a vector in a numerically safe way, we need to find a k constant which is the maximum value of the vector.
Then we need to subtract this maximum value from elements of the input each vector and compute the exponential sum.
Given a n*n matrix, for each row:
1.	Find the max value among all elements. Time complexity: O(N), Memory reads: O(N)
2.	Calculate the normalization factor (we cannot do it before the step 1 as we need to have the max element to compute the denominator summation): Time complexity: O(N), Memory reads: O(N)
3.	Apply the softmax to each element of the vector (also, we cannot do the step 3, before calculating the normalization factor because we need to divide each number by the normalization factor): Time complexity: O(N), Memory reads: O(N)

Pseudocode:
m0 = -inf
for i=1 to n:
mi = max(mi-1,xi)
l0 = 0
for j=1 to n:
lj = lj-1 + e^(xj – mn) 
for k=1 to n:
xk   e^(xk – mn)/ln
So, to apply softmax to a matrix of n*n, we need to load each of its elements three times and it must be done sequentially. This is very slow!
Is there any better way to not go through the vector 3 times and also prevent the exponential from exploding? Yes, Online Softmax
In online softmax, we will try to fuse the first two for loops into a single for loop. See how we can do it, while using block maximum value.
New Pseudocode:
m0 = -inf
l0 = 0
for i=1 to n:
maxi = max(maxi-1,xi) -> it gives max so far i.e. Local maxima (maximum until 0th element to ith element)
li = (li-1)*e^[ previous max i.e. maxi-1 (local max) - current max i.e. maxi (global max)] + e^(xj – maxi) -> previously computed li is fixed by the correction factor e^[ previous max i.e. maxi-1 (local max) - current max i.e. maxi (global max)] to calc the correct normalization factor
for k=1 to n:
xk   e^(xk – mn)/ln
Every time the global max changes, we multiply with the correction factor. If the global max doesn’t change it is e to the power of 0 i.e. 1. So, no correction is required and done the same with e to the power of zero that is 1.
Now, we transform the three passes through the array to two passes through the array to get the softmax.
Add a small vector example of [4, 1, 7, 2] to demonstrate online softmax operation and it’s correction factor calculation.
How Online Softmax Works
Online softmax is an optimized algorithm that computes the softmax function in two reduced memory passes by tracking running maximums and scaling sums dynamically, instead of requiring multiple full reads of the input data.
Introduced by researchers Maxim Milakov and Natalia Gimelshein in 2018, online softmax calculates the required values in a streaming or single-loop fashion. [1, 2]
•	Running maximum (m): It tracks the highest value seen so far as it scans the data.
•	Running sum (d): It updates the denominator on the fly.
•	Rescaling: When a new, larger maximum appears, it multiplies the old sum by a correction factor (\(\exp(prev\_max - new\_max)\)) instead of restarting the pass.
•	Efficiency: This reduces memory operations from three passes down to two, matching the speed of naive softmax methods while keeping numerical stability. You can read more about the mechanics in this Online Softmax Overview. [1, 2, 3]

__Block Matrix Multiplication:__
If you have a smaller number of cores (e.g., 8 CPU cores) and you like to multiply two matrices of large size to get an output matrix.
For simplicity, let’s take an example of a smaller matrix: give example you are multiplying a 8*4 matrix to a 4*8 matrix to get an 8*8 matrix so it has 64 elements, and you can compute these 64 elements parallelly as they do not depends on each other, but you have only 8 cores. Ho will you do iT?
You will use block matrix multiplication.
Here we divide original matrix into smaller blocks of elements, and then we do operation of matrix multiplication between these blocks. Like in the example we divide first matrix into 4 blocks (2,2) and second matrix into 8 blocks (2, 4) to get an output block matrix of (2, 4).


__GPU and CUDA__
The NVIDIA A100 Tensor Core GPU comes in two main memory configurations: 40 GB and 80 GB. 
Memory Specifications
•	A100 40GB: Uses high-bandwidth memory (HBM2) with a memory bandwidth of about 1.55 to 1.6 TB/s.
A100 80GB: Uses advanced high-bandwidth memory (HBM2e) that doubles the capacity and delivers over 2.0 TB/s of memory bandwidth.
GPU is the hardware unit that we buy and CUDA is the software stack made by NVIDIA to write the software for this GPU. AMD has their own software stach and the other manufacturer has their own.
In this particular video I will be showing you the example of CUDA kernels but the knowledge that you will get can be applied on the other GPUs.
CPU
The first difference between a CPU and a GPU is its purpose. My computer is now running on CPU, and the OS is interfacing with the CPU using the Scheduler. 
Suppose you are running a browser and running other software on the computer., scheduler is tasked with switching between these tasks very fast on the CPU in such a way that it looks like the processes are running concurrently. This actually is a fake kind of parallelism, unless your CPU has multiple cores. Nowdays, CPU ha smultiple core like dual core, quad core, or eight core. Each of these cores can execute instruction in  parallel.  The main purpose of the CPU is to execute many different task and switching between them very fast. You may have a browser open, playing some game, movie player and you have a word processor and then you have some utility to download files etc. Most of these tasks are not compute intensive or actually IO bound (meaning most of the time they are waiting for the network or for the disk). They are very different from each other in the purpose, so a browser is completely different from a movie player and it is completely different from a word processor. So, the job of the CPU is to reduce the latencies of processing all of these operations, and it is highly optimized to process each of these execution units called cores. Which means each core of a CPU has a part that is tasked to run first of all what is the next instruction to run or to predict the branch what the next operation maybe on the conditions that you are running. For example, if you have an if condition, the branch predictor can understand what the most likely next instruction would be and can do some optimizations. Also, the CPU has a lot of cache to reduce the latencies in loading the data from all the devices it can interface with. It can interface with RAM for sure, but it can interface with disk it can also interface with some peripheral like the key, the mouse, the printer etc. 

__GPU__
On the other hand, GPU is not tasked to do many different things at the same time. It is tasked to do with one thing or few things but with massive amount of data. Operations that we do on the GPU requires a lot of computation. For this reason, most of the area the physical area on the GPU is dedicated to compute units. The green stuff in the image and the are called as cores. You can see the part that is given to control area for the next instruction prediction and optimization is very little.
On a simple addition task of two vectors, on CPU we can use a for loop:
for i = 0 to n-1:
C[i] = A[i] + B[i]
we can ask each core of the GPU to add each element of vector A and vector B and store it as a element of vector C. Multiple cores/threads work together on the similar job on a GPU.
Each core in parallel calculate sum of each elements:
Core1:				Core 2: 				Core n:
C[0] = A[0] + B[0]            C[0] = A[0] + B[0]                                C[n-1] = A[n-1] + B[n-1]
How to do a vector summation: First of all, usually a GPU is interfaced with a CPU and CPU tells the GPU what the data is it is going to work with. So, the CPU needs to have this data and transfer it to the GPU and then the GPU do the summation and then the CPU copy back the summation result form the GPU and then make it available to the program. 

Launching a kernel means we launch a program that the GPU should execute in parallel on multiple threads (or multiple cores). Each Warp contains 32 threads and each of these threads shares same control unit. It means these group of threads will always execute same statement at any time. They will always work synchronously, they will always work on the same instructions, but the data could be different, will be their own data because they will be working on their own registers. This programming model is called single instruction multiple data (SIMD). CUDA like sto call it Single Instruction Multiple Thread (SIMT). You want to minimize the control divergence by thinking, why does not give their own control unit to each core. Because control unit is expensive to add in the chip area of the GPU, it is not done. It is much more efficient to add much more worker than to add the control unit for each worker. So, this is a design choice of the GPU and it works fine!
One thread work can be dependent on other thread, we use synchronization for this purpose.
Cuda code to calc sum of two numbers:
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) { //__global__ is kind of superset of C language where we have cuda sepcific keywords.
    //when we ask cuda to allocate N parallel threads, it allocates 
    // N number of therads and assign a unique identifiers to each of these threads.
    // The item taht each thread should process is equal to its thread index. So thread 0 process a[o]
    // b[0], c[0]; So thread 1 process a[1], b[1], c[1] and so on.
    //int i = threadIdx.x; // here we are saying which item each thread will process
    // 256 threads per block
    // N = numnber of blocks
    // every position is handled by a different thread index
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(idx < N){ // since we know that we are going to launch N threads so i should be automatically will be less than N; so why we are having this if condition?
    // this condition is because when cuda launches a thread it is in the multiple of 32 which is in case
    // so if we have 34 items in our vector and we ask cuda to launch 34 threads it will not launch 34, it 
    // launch 64 threads (multiple of 32, which is the warp size)!
    // with this if statement we are asking thread to not do anything whose id is greater than equal to the elements, sine all the threads in a warp shares the same control unit, they will enter into the if statement even our if condition fails (when we do not have any elements left to add). How cuda manges this control divergence? All the instructions for which if statement is True, will execute the addition instruction. All the thtreads which have this if statement equal to False, they will not do any operations, they will just sit idle! This is called “Control divergence” and it could reduce the throughput of your program. 
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    //this is kernal launch
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N); // it says launch the blocks number: blocksPerGrid; and number of threads: threadsPerBlock
    cudaDeviceSynchronize(); //C is out, A and B are input as in example they show, N is number of vector elemenmts 
}

Now, we seen how a kernel works, let’s move to another example:






