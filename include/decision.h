#ifndef _DECISION_H
#define _DECISION_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COMM_SIZE_MIN  4
#define COMM_SIZE_MAX  16
#define COMM_SIZE_STEP 1

#define MESS_SIZE_MIN  1024
#define MESS_SIZE_MAX  (16 * 1024)
#define MESS_SIZE_STEP 1024

#define TRIALS_NO 8

unsigned char **decision_matrix;

void decision_matrix_construct(int verbose);

#endif
