/** @file     print.h
 *  @brief    Function prototypes for print.
 *  @author   Marcelo Pinto (xmrcl0@gmail.com)
 *  @date     09/12/2017
 */

#include <stdio.h>
#include <stdlib.h>

/** @brief Print report.
 *
 *  @param[in] d  Distance matrix 
 *  @param[in] p  Path array 
 *  @param[in] n  Number of cities 
 *  @param[in] l  Path's length
 *  @param[in] pm Print mode
 *  @return void 
 */
void print_path(float ***d, int **p, size_t n, float l, int pm);

/** @brief Print distance matrix.
 *
 *  @param[in] d Distance matrix 
 *  @param[in] n Number of cities
 *  @return void 
 */
void print_dist(float ***d, size_t n);

/** @brief Print coordinates matrix.
 *
 *  @param[in] c Coordinate matrix 
 *  @param[in] n Matrix size 
 *  @return void 
 */
void print_cord(float ***c, size_t n);

/** @brief Print report.
 *
 *  @param[in] c  Coordinate matrix
 *  @param[in] d  Distance matrix
 *  @param[in] m  Minimal path array
 *  @param[in] n  Number of cities
 *  @param[in] ml Minimal length 
 *  @param[in] i  Number of simulations
 *  @param[in] pm Print mode
 *  @param[in] grid Number of blocks in the grid
 *  @param[in] block Number of threads in a block
 *  @return void 
 */
void print_repo(float **c, float **d, int *m, int n, float ml, long double i, int pm, int grid, int block);
