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

void
create_path (int num_city, int **coord)
{
  int i;

  (*coord) = (int *) malloc ((num_city + 1) * sizeof (int));

  randperm (num_city, *coord);
  (*coord)[num_city] = (*coord)[0];
}


float
measure_path (float ***distance, int num_city, int **path)
{
  int i;
  float l = 0;

  for (i = 0; i < num_city; i++)
    l = l + (*distance)[(*path)[i]][(*path)[i + 1]];
  return l;
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
