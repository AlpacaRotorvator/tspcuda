/** @file     tsp.c
 *  @brief    Traveling Salesman Problem functions.
 *  @author   Marcelo Pinto (xmrcl0@gmail.com)
 *  @date     09/12/2017
 *  @updated  09/21/2017
 */

#include "tsp.h"
#include "print.h"
#include "utils.h"


void
distance_matrix (float ***coord, float ***distance, int num_city)
{
  int i, j, nrows, ncols;

  ncols = num_city;
  nrows = num_city;

  *distance = (float **) malloc (nrows * sizeof (float *));
  for (i = 0; i < nrows; i++)
    (*distance)[i] = (float *) malloc (ncols * sizeof (float));

  for (i = 0; i < num_city; i++)
    for (j = 0; j < num_city; j++)
      (*distance)[i][j] = sqrt (pow ((*coord)[i][0] - (*coord)[j][0], 2) + pow ((*coord)[i][1] - (*coord)[j][1], 2));
}

void
distance_vector (float ***coord, float **distance, int num_city)
{
  int i, j, nrows, ncols;

  ncols = num_city;
  nrows = num_city;

  *distance = (float *) malloc (num_city * num_city  * sizeof (float));

  for (i = 0; i < num_city; i++)
  {
    for (j = 0; j < num_city; j++)
    {
      (*distance)[i + j * num_city] = sqrt (pow ((*coord)[i][0] - (*coord)[j][0], 2) + pow ((*coord)[i][1] - (*coord)[j][1], 2));
    }
  }
}

__global__ void
initRNG(curandState *const rngStates, const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}

__device__ void
create_path (int num_city, int *coord, curandState localState)
{
  randperm (num_city, coord, localState);
}


__device__ float
measure_path (float *distance, int num_city, int *path)
{
  int i;
  float l = 0;

  for (i = 0; i < num_city - 1; i++)
  {
    int j = path[i];
    int k = path[i + 1];
    l = l + distance[j + num_city * k];
  }
  return l;
}

__global__ void
kernel (float *const mindists, int *const minpaths, float *const distance,
	curandState *const rngStates, const int n_cities, const int n_iter)
{
  // Determine thread ID
  unsigned int bid = blockIdx.x;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int ltid = threadIdx.x;

  // Sort out shared memory
  extern __shared__ float sdata[];
  float *threadsMinDists = sdata;
  int *minPathMatrix = (int *) &threadsMinDists[blockDim.x];
  int *computePathMatrix = (int *) &minPathMatrix[n_cities * blockDim.x];

  // Sort out local(ie this thread's) variables
  float *curThreadMinDist = &threadsMinDists[ltid];
  int *curThreadMinPath = &minPathMatrix[ltid * n_cities];
  int *curThreadCptPath = &computePathMatrix[ltid * n_cities];
  curandState localState = rngStates[tid];

  //Run everything at least once to initialize a sane minimum path
  create_path (n_cities, curThreadMinPath, localState);
  *curThreadMinDist =  measure_path (distance, n_cities, curThreadMinPath);

  float curThreadCptDist = 0;
  for (int i = 1; i < n_iter; i++)
  {
    create_path (n_cities, curThreadCptPath, localState);
    curThreadCptDist = measure_path (distance, n_cities, curThreadMinPath);

    if (curThreadCptDist < *curThreadMinDist)
    {
      *curThreadMinDist = curThreadCptDist;
      // This is not great at all for performance but guess who cares?
      memcpy (curThreadMinPath, curThreadCptPath, sizeof(int) * n_cities);
    }
  }
  unsigned int minDistTid = reduce_dists(threadsMinDists);
  if(ltid == 0)
  {
    printf("\n\t%d\n", minDistTid);
  }
  
  if (ltid == minDistTid)
  {
    mindists[bid] = threadsMinDists[0];
    if(bid == 2)
    {
      printf("\n(%d,%d) here! My distance is %f, and my path:\n\t[", bid, ltid, mindists[bid]);
      for(int i = 0; i < n_cities - 1; i++)
      {
	printf("%d, ", curThreadMinPath[i]);
      }
      printf("%d]\n", curThreadMinPath[n_cities - 1]);
    }
    memcpy(curThreadMinPath, &minpaths[bid], sizeof(int) * n_cities);
  }
}

__device__ unsigned int
reduce_dists(float *const threadsMinDists)
{
  unsigned int ltid = threadIdx.x;
  
  __syncthreads();
  
  // Do reduction in shared mem
  for (unsigned int s = blockDim.x / 2 ; s > 0 ; s >>= 1)
  {
    if (ltid < s)
    {
      if (threadsMinDists[s] < threadsMinDists[ltid])
      {
	threadsMinDists[ltid] = threadsMinDists[ltid + s];
	threadsMinDists[ltid + s] = ltid + s;
      }
      else {
	threadsMinDists[ltid + s] = ltid;
      }
    }
    __syncthreads();
  }

    return threadsMinDists[1];
}

int
read_file (char *file, float ***array)
{
  int i, j, nrows = 0, ncols = 2;
  char c;
  char *line = NULL;
  size_t len=0;
  FILE *fp;

  fp = fopen (file, "r");
  if (fp == NULL)
    return 0;

  while ((getline(&line, &len, fp) != -1))
  { 
    if (!is_coordinate (line))
      return -1;
    nrows++;
  }
  free(line);

  // Allocate memory for coordinates matrix 
  *array = (float **) malloc (nrows * sizeof (float *));
  for (i = 0; i < nrows; i++)
    (*array)[i] = (float *) malloc (ncols * sizeof (float));

  // Read coordinates from file to coordinates matrix
  fseek (fp, 0, SEEK_SET);
  for (i = 0; i < nrows; i++)
    for (j = 0; j < ncols; j++)
      if (!fscanf (fp, "%f", &(*array)[i][j]))
	break;
  fclose (fp);

  return nrows;
}

__device__ void
randperm (int n, int perm[], curandState localState)
{
  int i, j, t;

  for (i = 0; i < n; i++)
  {
    perm[i] = i;
  }
  for (i = 0; i < n; i++)
  {
    j = curand (&localState) % (n - i) + i;
    t = perm[j];
    perm[j] = perm[i];
    perm[i] = t;
  }
}
