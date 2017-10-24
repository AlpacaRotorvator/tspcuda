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

void
help (void)
{
  printf ("usage: tsp [-h] [-g] [-n <ITER>] -m <MODE> -f <FILE>\n");
  printf ("Find best path to Traveling Salesman Problem using Monte Carlo Method\n\n");
  printf ("Options:\n");
  printf ("  -n <ITER>    Number of paths to simulate\n");
  printf ("  -m <MODE>    Exibition mode 0, 1 or 2 (silent = 0)\n");
  printf ("  -f <FILE>    Cities coordinates file\n");
  printf ("  -g           Generate city coordinates + shortest path graph in graphviz's dot format\n");
  printf ("  -h           Show this help message and exit\n\n");
  printf ("Example:\n");
  printf ("  tsp -n 5 -m 0 -f data/grid04_xy.txt   # Simulates 5 paths for 4 cities data file\n");
}


int
parse_cmdline(int argc, char **argv, long double *num_iter, int *num_cities, float ***coord, int *mode, int *gendot)
{
  char c;
  long double i;
  int nflag = 0, mflag = 0, fflag = 0, gflag = 0;
  float len = 0, min_len = FLT_MAX;
  FILE *file;


  // Read and parse command line arguments
  opterr = 0;
  while ((c = getopt (argc, argv, "n:m:f:gh::")) != -1)
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
    case 'h':
      help ();
      exit (EXIT_SUCCESS);
      break;
    case '?':
      fprintf (stderr, "%s: error: invalid option\n", argv[0]);
      return 1;
    default:
      fprintf (stderr, "usage: tsp [-h] [-g] [-n <ITER>] -m <MODE> -f <FILE>\n");
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

  // Check if obrigatory argumets were given
  if (nflag == 0 || mflag == 0 || fflag == 0)
  {
    fprintf (stderr, "%s: error: too few parameters\n", argv[0]);
    fprintf (stderr, "usage: tsp [-h] [-n <ITER>] -m <MODE> -f <FILE>\n");
    exit (EXIT_FAILURE);
  }
}

void
setupGPU (unsigned int device, struct cudaDeviceProp *deviceProp, unsigned int *blocksize, unsigned int *gridsize)
{
  cudaError_t cudaResult = cudaSuccess;
  cudaResult = cudaGetDeviceProperties(deviceProp, device);

  if (cudaResult != cudaSuccess)
  {
    fprintf (stderr, "Não foi possível obter as propriedades da GPU.\n");
    fprintf (stderr, cudaGetErrorString(cudaResult));
    exit (EXIT_FAILURE);
  }

  cudaResult = cudaSetDevice(device);

  if (cudaResult != cudaSuccess)
  {
    fprintf (stderr, "Não foi possível conectar à GPU.\n");
    fprintf (stderr, cudaGetErrorString(cudaResult));
    exit (EXIT_FAILURE);
  }

  if (*blocksize > (unsigned int)deviceProp->maxThreadsDim[0])
  {
    fprintf (stderr, "O número de threads por bloco excede as capacidades do dispositivo");
    exit (EXIT_FAILURE);
  }

  if (*gridsize > (unsigned int)deviceProp->maxGridSize[0])
  {
    fprintf (stderr, "O número de blocos na grid excede as capacidades do dispositivo");
    exit (EXIT_FAILURE);
  }
}

int
main (int argc, char **argv)
{
  long double i, num_iter;
  int num_cities, mode, gendot = 0;
  float **coord, **distance;
  int *path, *min_path;
  float len = 0, min_len = FLT_MAX;

  // Parse command line
  parse_cmdline(argc, argv, &num_iter, &num_cities, &coord, &mode, &gendot);
 

  // Create distance matrix
  distance_matrix (&coord, &distance, num_cities);

  //Hardcoded device for now
  unsigned int device = 0;
  cudaError_t cudaResult = cudaSuccess;
  
  //Block and grid
  dim3 block;
  dim3 grid;
  //Hardcoded for now
  block.x = 64;
  grid.x = 64;
  
  //Initalize device, perform basic checks
  struct cudaDeviceProp deviceProp;
  setupGPU (device, &deviceProp, &block.x, &grid.x);

  // Allocate memory for RNG states
  curandState *d_rngStates = 0;
  cudaResult = cudaMalloc((void **)&d_rngStates, grid.x * block.x * sizeof(curandState));

  if (cudaResult != cudaSuccess)
  {
    fprintf(stderr, "Erro: não foi possível alocar memóra na GPU para os estados do RNG\n");
    fprintf(stderr, cudaGetErrorString(cudaResult));
    exit(EXIT_FAILURE);
  }
  
  //Initialize RNG
  //initRNG<<<grid, block>>>(d_rngStates, time (NULL));

  //Sadly CUDA doesn't like arrays-of-pointers matrices very much, flattened coord and
  //distance matrices are thus needed.
  float *fdistance;
  distance_vector (&coord, &fdistance, num_cities);
  
  //Allocate and copy distance matrix to device
  float * d_distance;

  cudaResult = cudaMalloc( (void **) &d_distance, num_cities * num_cities * sizeof(float));

  if (cudaResult != cudaSuccess)
  {
    fprintf(stderr, "Erro: não foi possível alocar memóra na GPU para a matriz de distâncias\n");
    fprintf(stderr, cudaGetErrorString(cudaResult));
    exit(EXIT_FAILURE);
  }

  cudaResult = cudaMemcpy(d_distance, fdistance, num_cities * num_cities * sizeof(float), cudaMemcpyHostToDevice);

  if (cudaResult != cudaSuccess)
  {
    fprintf(stderr, "Erro: não foi possível copiar a matriz de distâncias para a GPU.\n");
    fprintf(stderr, cudaGetErrorString(cudaResult));
    exit(EXIT_FAILURE);
  }

  //Free the flattened distance matrix
  free(fdistance);
  
  //kernel<<<grid, block>>> (num_iter, &d_distance, num_cities);
  
  // Simulates n round trips
  if (!mode == 0)
  {
    printf ("POSSIBLE PATHS:\n");
  }
  for (i = 0; i < num_iter; i++)
  {
    create_path (num_cities, &path);
    len = measure_path (&distance, num_cities, &path);

    // Print path if option 
    if (!mode == 0)
    {
      print_path (&distance, &path, num_cities, len, mode);
    }

    // Save the path if better 
    if (len < min_len)
    {
      min_len = len;
      array_copy (&path, &min_path, num_cities);
    }
    free (path);
  }

  if (!mode == 0)
  {
    printf ("\n");
  }

  // Print report 
  print_repo (coord, distance, min_path, num_cities, min_len, num_iter, mode);

  // Generate dot file
  if(gendot) {
      gen_graphviz (coord, min_path, num_cities);
  }

  free (min_path);
  free (coord);
  free (distance);

  return 0;
}
