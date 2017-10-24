/** @file     utils.h
 *  @brief    Function prototypes for utils.
 *  @author   Marcelo Pinto (xmrcl0@gmail.com)
 *  @date     09/12/2017
 */

#include <stddef.h>
#include <regex.h>
#include <stdlib.h>
#include <stdio.h>

/** @brief Verify if a line contains a valid xy coordinate.
 *
 *  @param [in] l  Line to be validated
 *  @return Validation status 
 *
 *  @retval  0 Success
 *  @retval  1 Failure
 *  @retval -1 Internal error
 */
int is_coordinate(char *l);


/** @brief Verify if a string is a valid integer.
 *
 *  @param[in] n  Number to be validated
 *  @return Validation status 
 *
 *  @retval  0 Success
 *  @retval  1 Failure
 *  @retval -1 Internal error
 */
int is_integer(char *n);


/** @brief Verify if a string is a valid number.
 *
 *  @param[in] number Number to be validated
 *  @return Validation status 
 *
 *  @retval  0 Success
 *  @retval  1 Failure
 *  @retval -1 Internal error
 */
int is_positive_number(char *n);


/** @brief Copy an array.
 *
 * @param[in]  src Source array
 * @param[out] dst Destination array
 * @param[in]  n   Number of bytes to copy
 * @return void
 */
void array_copy(int **src, int **dst, size_t n);


/** @brief Calculates n!.
 *
 *  @param[in] n Number
 *  @return The factorial of n
 */
long double factorial(int n);
