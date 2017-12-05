/** @file     main.c
 *  @brief    Traveling Salesman Problem.
 *  @author   Marcelo Pinto (xmrcl0@gmail.com)
 *  @date     09/26/2017
 *  @version  0.3
 */

#include "tsp.h"
#include "print.h"
#include "utils.h"
#include "graphviz.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_timer.h>

void
help (void)
{
  printf ("usage: tsp [-h] [-g] -n <ITER> [-m <MODE>] [-b <BNUM>] [-t <TNUM>] [-d <DEVN>] -f <FILE>\n");
  printf ("Find best path to Traveling Salesman Problem using Monte Carlo Method\n\n");
  printf ("Options:\n");
  printf ("  -n <ITER>    Number of paths to simulate per thread\n");
  printf ("  -m <MODE>    Exibition mode 0, 1 or 2 (silent = 0)\n");
  printf ("  -d <DEVN>    Number of the device to be used\n");
  printf ("  -b <BNUM>    Number of blocks in the grid\n");
  printf ("  -t <TNUM>    Number of threads per block(must be power of two)\n");
  printf ("  -f <FILE>    Cities coordinates file\n");
  printf ("  -g           Generate city coordinates + shortest path graph in graphviz's dot format\n");
  printf ("  -s <SEED>    Seed for RNG, time(NULL) by default\n");
  printf ("  -h           Show this help message and exit\n\n");
  printf ("Example:\n");
  printf ("  tspcuda -b 64 -t 256 -n 2000 -m 0 -f data/grid15_xy.txt   # Using 64 blocks of 256 threads, simulate 2000 paths in each thread for 15 cities data file\n");
}

int
parse_cmdline(int argc, char **argv, long double *num_iter, int *num_cities,
	      float ***coord, int *mode, int *gendot, int *threadsPerBlock,
	      int *numBlocks, unsigned int *seed, int * device, struct cudaDeviceProp *devProp)
{
  char c;
  long double i;
  int nflag = 0, mflag = 0, fflag = 0, gflag = 0, bflag = 0, tflag = 0, sflag = 0, dflag = 0;
  float len = 0, min_len = FLT_MAX;
  FILE *file;
  int candidate = 0;
  cudaError_t result = cudaSuccess;


  // Read and parse command line arguments
  opterr = 0;
  while ((c = getopt (argc, argv, "n:m:f:b:t:s:d:gh::")) != -1)
    switch (c)
    {
    case 'n':
      nflag = 1;
      if (!is_integer (optarg))
      {
        fprintf (stderr, "%s: error: number of simulations must be an integer\n", argv[0]);
        exit (EXIT_FAILURE);
      }
      else
      {
        *num_iter = strtold (optarg, NULL);
      }
      break;
    case 'm':
      mflag = 1;
      if (!is_positive_number (optarg))
      {
        fprintf (stderr, "%s: error: invalid mode, choose 0, 1 or 2\n", argv[0]);
        exit (EXIT_FAILURE);
      }
      else
      {
        *mode = atoi (optarg);
      }
      if (*mode > 2)
      {
        fprintf (stderr, "%s: error: invalid mode, choose 0, 1 or 2\n", argv[0]);
        exit (EXIT_FAILURE);
      }
      break;
    case 'f':
      fflag = 1;
      *num_cities = read_file (optarg, coord);
      if (num_cities == 0)
      {
        fprintf (stderr, "%s: error: no such file or directory\n", argv[0]);
        exit (EXIT_FAILURE);
      }
      else
      if (*num_cities == -1)
      {
        fprintf (stderr, "%s: error: incompatible data file\n", argv[0]);
        exit (EXIT_FAILURE);
      }
      break;
    case 'g':
      gflag = 1;
      *gendot = 1;
      break;
    case 't':
      tflag = 1;
      candidate = atoi(optarg);
      if (!is_integer (optarg))
      {
        fprintf (stderr, "%s: error: number of threads per block must be an integer\n", argv[0]);
        exit (EXIT_FAILURE);
      }
      else if (candidate <= 0)
      {
	fprintf (stderr, "%s: error: number of threads per block must be greater than zero.\n", argv[0]);
	exit (EXIT_FAILURE);
      }
      else if ((candidate & (candidate - 1)) != 0)
      {
	fprintf (stderr, "%s: error: number of threads per block must be a power of two(for efficient reduction).\n", argv[0]);
	exit (EXIT_FAILURE);
      }
      else
      {
        *threadsPerBlock = strtold (optarg, NULL);
      }
      break;
    case 'b':
      candidate = atoi(optarg);
      bflag = 1;
      if (!is_integer (optarg))
      {
        fprintf (stderr, "%s: error: number of blocks in the grid must be an integer\n", argv[0]);
        exit (EXIT_FAILURE);
      }
      else if (candidate <= 0)
      {
	fprintf(stderr, "%s: error: number of blocks must be greater than zero\n", argv[0]);
	exit (EXIT_FAILURE);
      }
      else
      {
	*numBlocks = candidate;
      }
      break;
    case 's':
	sflag = 1;
	if(!is_integer (optarg))
	{
	  fprintf (stderr, "%s: error: rng seed must be an integer\n", argv[0]);
	  exit(EXIT_FAILURE);
	}
	else
	{
	  *seed = strtold (optarg, NULL);
	}
	break;
    case 'd':
      candidate = atoi(optarg);
      result = cudaSetDevice(candidate);
      if (result != cudaSuccess) {
	fprintf (stderr, "%s: error: invalid device requested\n", argv[0]);
	exit (EXIT_FAILURE);
      }
      *device = candidate;
      dflag = 1;
      break;
    case 'h':
      help ();
      exit (EXIT_SUCCESS);
      break;
    case '?':
      fprintf (stderr, "%s: error: invalid option\n", argv[0]);
      return 1;
    default:
      fprintf (stderr, "usage: tsp [-h] [-g] -n <ITER> [-m <MODE>] [-b <BNUM>] [-t <TNUM>] [-d <DEVN>] -f <FILE>\n");
      abort ();
    }

  for (i = optind; i < argc; i++)
  {
    fprintf (stderr, "%s: error: too many or too few arguments\n", argv[0]);
    exit (EXIT_FAILURE);
  }

  if (num_iter + 1 < num_iter)
  {
    fprintf (stderr, "%s: error: number of simulations must be less than %Lf \n", argv[0], LDBL_MAX);
    exit (EXIT_FAILURE);
  }

  
  // Check if obligatory argumets were given
  if (nflag == 0 || fflag == 0)
  {
    fprintf (stderr, "%s: error: too few parameters\n", argv[0]);
    fprintf (stderr, "usage: tsp [-h] [-g] -n <ITER> [-m <MODE>] [-b <BNUM>] [-t <TNUM>] [-d <DEVN>] -f <FILE>\n");
    exit (EXIT_FAILURE);
  }

  if(!dflag)
  {
    handleCudaErrors (cudaSetDevice(*device));
  }
    
  handleCudaErrors (cudaGetDeviceProperties(devProp, *device));
  
  if (*threadsPerBlock > devProp->maxThreadsDim[0])
  {
    fprintf (stderr, "O número de threads por bloco excede as capacidades do dispositivo\n");
    exit (EXIT_FAILURE);
  }

  if (*numBlocks > devProp->maxGridSize[0])
  {
    fprintf (stderr, "O número de blocos na grid excede as capacidades do dispositivo\n");
    exit (EXIT_FAILURE);
  }
  
}

int
main (int argc, char **argv)
{
  long double i, num_iter;
  int num_cities, mode = 0, gendot = 0, threadsPerBlock = 64, numBlocks = 16;
  float **coord, **distance;
  int *min_path;
  float len = 0, min_len = FLT_MAX;
  unsigned int rngSeed = time(NULL);
  int device = 0;
  struct cudaDeviceProp deviceProp;
  //Block and grid
  dim3 block;
  dim3 grid;

  // Parse command line
  parse_cmdline(argc, argv, &num_iter, &num_cities, &coord, &mode, &gendot, &threadsPerBlock, &numBlocks, &rngSeed, &device, &deviceProp);
  
  block.x = threadsPerBlock;
  grid.x = numBlocks;
  StopWatchInterface *timer = NULL;

  // Create distance matrix
  distance_matrix (&coord, &distance, num_cities);

  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  // Allocate memory for RNG states
  curandState *d_rngStates = 0;
  handleCudaErrors (cudaMalloc((void **)&d_rngStates, grid.x * block.x * sizeof(curandState)));

  
  //Initialize RNG
  initRNG<<<grid, block>>>(d_rngStates, rngSeed);
  
  //Sadly CUDA doesn't like arrays-of-pointers matrices very much, flattened coord and
  //distance matrices are thus needed.
  float *fdistance;
  distance_vector (&coord, &fdistance, num_cities);
  
  //Allocate and copy distance matrix to device
  float * d_distance;

  handleCudaErrors (cudaMalloc( (void **) &d_distance, num_cities * num_cities * sizeof(float)));

  handleCudaErrors (cudaMemcpy(d_distance, fdistance, num_cities * num_cities * sizeof(float)
			       , cudaMemcpyHostToDevice));

  //Free the host-side flattened distance matrix
  free(fdistance);

  //Allocate memory in device for computation results
  int * d_minpaths;

  handleCudaErrors (cudaMalloc( (void **) &d_minpaths, grid.x * num_cities * sizeof(int)));

  float * d_mindists;

  handleCudaErrors (cudaMalloc( (void **) &d_mindists, grid.x * sizeof(float)));
  
  /* Shared memory setup:
   * - One float for each thread in a block to store the minimum distance computed
   * - Two num_cities-long int array. Threads alternate between using one for storing
   *   the best path and the other for storing the next path to be computed. That way
   *   one is able to avoid the horridly expensive memory copying I was doing earlier
   *   and possibly obtain better memory locality in warps.
   */
  kernel<<<grid, block,
    block.x * sizeof(float) + 2 * block.x * sizeof(int) * num_cities>>>
    (d_mindists, d_minpaths, d_distance, d_rngStates,  num_cities, num_iter);

  // Copy results back to device
  float *mindists = (float *) malloc(grid.x * sizeof(float));
  handleCudaErrors (cudaMemcpy(mindists, d_mindists,
			       grid.x * sizeof(float), cudaMemcpyDeviceToHost));
  
  // Finish reduction on host
  int min_idx = 0;
  min_len = mindists[0];
  for (int i = 1; i < grid.x; i++)
  {
    if (mindists[i] < min_len)
    {
      min_len = mindists[i];
      min_idx = i;
    }
  }
  free(mindists);

  min_path = (int *) malloc(num_cities * sizeof(int));

  handleCudaErrors (cudaMemcpy(min_path, &d_minpaths[min_idx * num_cities],
			       num_cities * sizeof(int), cudaMemcpyDeviceToHost));

  // Clean up device variables
  cudaFree(d_rngStates);
  cudaFree(d_distance);
  cudaFree(d_minpaths);
  cudaFree(d_mindists);
  sdkStopTimer(&timer);
  
  float elapsedTime = sdkGetAverageTimerValue(&timer)/1000.0f;


  // Print report 
  print_repo (coord, distance, min_path, num_cities, min_len, num_iter, mode, grid.x, block.x,
	      elapsedTime);

  // Generate dot file
  if(gendot) {
      gen_graphviz (coord, min_path, num_cities);
  }
  
  free (min_path);
  free (coord);
  free (distance);

  return 0;
}
