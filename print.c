#include "print.h"


/*
 * Print a path
 */
void
print_path(float ***d, int **p, size_t n, float l, int m)
{
    int i;

    if (m == 0)
    {
        for (i = 0; i < n; i++)
            printf("d[%d][%d]=%.2f, ", (*p)[i], (*p)[i+1], (*d)[(*p)[i]][(*p)[i+1]]);
        printf("distance=%.2f\n", l); 
    }

    else
    if (m == 1)
    {
        for (i = 0; i <= n; i++)
            printf("%d,", (*p)[i]); 
        printf("%.2f\n", l); 
    }

    else
    if (m == 2)
    {
        for (i = 0; i <= n; i++)
            printf("%4d", (*p)[i]);
        printf("\t%4.2f\n", l); 
    }   
}


/*
 * Print distance matrix 
 */
void
print_dist(float ***d, size_t n)
{
    int i, j;

    printf("Distance matrix:\n");
    for (i = 0; i < n; i++)
    {   
        for (j = 0; j < n; j++)
            printf("%6.2f", (*d)[i][j]);
        printf("\n");
    }   
    printf("\n");
}

/*
 * Print coordinate matrix 
 */
void
print_cord(float ***v, size_t n)
{
    int i, j;

    printf("\nCities coordinates:\n");
    for (i = 0; i < n; i++)
    {   
        for (j = 0; j < 2; j++)
            printf("%.4f ", (*v)[i][j]); 
        printf("\n");
    }
    printf("\n");
}