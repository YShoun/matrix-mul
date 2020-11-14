/*
    Code written by Yong Shoun PHO
*/

#include <iostream>
#include <time.h>
#include <omp.h>

using namespace std;

double ALLOCATION_TIME = 0;
double MATRIX_SIZE = 1024;

/*--------------------------------------------Functions------------------------------------------------------
------------------------------------------------------------------------------------------------------------*/

double** matrixGenerator(int size, bool isC){

	double** m = (double**)malloc(size * sizeof(double*)); // rows
	for(int i=0; i<size; i++){
		m[i] = (double*)malloc(size * sizeof(double)); //cols
	}

	if(!isC){
        srand(1); //seed
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i=0; i<size; i++){
            for(int j=0; j<size; j++){
                m[i][j] = rand() % 9 + 1; // max + min
            }
        }
	}
	return m;
}


void checkResult(double**A, double**B, double**C, int size){
    double **checker = matrixGenerator(size, true);
    for(int i=0; i<size; i++){
		for(int j=0; j<size; j++){
			for(int k=0; k<size; k++){
				checker[i][j] += A[i][k] * B[k][j];
			}
			if(checker[i][j] != C[i][j]){ //check value of strassen compared to traditional
				printf("Matrix checker: False\n");
				return; // end the function if found different
			}
		}
	}
	printf("Matrix checker: True\n");
}


void matrixViewer(double** m, int size){
    for(int i=0; i<size; i++){
		for(int j=0; j<size; j++){
            cout << m[i][j] << "   ";
		}
		cout << endl << endl;
	}

	printf("----------------------------------------------\n\n\n");
}


void matrixAddition(double**A, double**B, double**C, int size){
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i=0; i < size; i++){
        for(int j=0; j < size; j++){
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}


void matrixSubstration(double** A, double** B, double** C, int size){
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i=0; i < size; i++){
        for(int j=0; j < size; j++){
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}


void traditionalMulti(double** A, double** B, double** C, int size){
    // no openMP optimization here as we just need traditional multiplication
	for(int i=0; i<size; i++){
		for(int j=0; j<size; j++){
			for(int k=0; k<size; k++){
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}


void strassenMulti(double** A, double** B, double** C, int size){

    if (size == 1){ // if matrix size == 1
        C[0][0] = A[0][0] * B[0][0];
        return;
    }
    else if (size < MATRIX_SIZE/2){ // when reaching size threshold = traditional faster than strassen
        traditionalMulti(A,B,C,size);
        return;
    }

    /*
        Divide and Conquer method
    */

    int reduced_size = (int) size/2;

    clock_t extra_start = clock();
    // sub Matrix of A init
    double **a11, **a12, **a21, **a22;
    a11 = matrixGenerator(reduced_size, true);
    a12 = matrixGenerator(reduced_size, true);
    a21 = matrixGenerator(reduced_size, true);
    a22 = matrixGenerator(reduced_size, true);

    // sub Matrix of B init
    double **b11, **b12, **b21, **b22;
    b11 = matrixGenerator(reduced_size, true);
    b12 = matrixGenerator(reduced_size, true);
    b21 = matrixGenerator(reduced_size, true);
    b22 = matrixGenerator(reduced_size, true);

    // sub Matrix of C init
    double **c11, **c12, **c21, **c22;
    c11 = matrixGenerator(reduced_size, true);
    c12 = matrixGenerator(reduced_size, true);
    c21 = matrixGenerator(reduced_size, true);
    c22 = matrixGenerator(reduced_size, true);

    // Strassen 7 formulas init

    double **s1, **s2, **s3, **s4, **s5, **s6, **s7;
    s1 = matrixGenerator(reduced_size, true);
    s2 = matrixGenerator(reduced_size, true);
    s3 = matrixGenerator(reduced_size, true);
    s4 = matrixGenerator(reduced_size, true);
    s5 = matrixGenerator(reduced_size, true);
    s6 = matrixGenerator(reduced_size, true);
    s7 = matrixGenerator(reduced_size, true);

    // SubMatrix to store sub results from strassen calculation and from applying multiple additions/substrations
    double **subA = matrixGenerator(reduced_size, true);
    double **subB = matrixGenerator(reduced_size, true);

    //SubMatrix to store sub results
    double **temp0 = matrixGenerator(reduced_size, true);
    double **temp1 = matrixGenerator(reduced_size, true);

    // Fill sub-matrix of A and B
    for (int i = 0; i < reduced_size; i++){
        for (int j = 0; j < reduced_size; j++){
            a11[i][j] = A[i][j];
            a12[i][j] = A[i][j + reduced_size];
            a21[i][j] = A[i + reduced_size][j];
            a22[i][j] = A[i + reduced_size][j + reduced_size];

            b11[i][j] = B[i][j];
            b12[i][j] = B[i][j + reduced_size];
            b21[i][j] = B[i + reduced_size][j];
            b22[i][j] = B[i + reduced_size][j + reduced_size];
        }
    }

    clock_t extra_end = clock();
    ALLOCATION_TIME += (extra_end - extra_start); // put the allocation time of the matrixes in the global variable

    /*
        Strassen's formula calculation
    */
    #pragma omp task
    {
        // S1 = (a11 + a22) (b11 + b22) = subA * subB
        matrixAddition(a11, a22, subA, reduced_size);
        matrixAddition(b11, b22, subB, reduced_size);
        strassenMulti(subA, subB, s1, reduced_size);
    }
    #pragma omp taskwait
    {
        // S2 = (a21 + a22 ) b11
        matrixAddition(a21 ,a22, subA, reduced_size);
        strassenMulti(subA, b11, s2, reduced_size);
    }
    #pragma omp taskwait
    {
        // S3 = a11 (b12 - b22)
        matrixSubstration(b12, b22, subB, reduced_size);
        strassenMulti(a11, subB, s3, reduced_size);
    }
    #pragma omp taskwait
    {
        // S4 = a22 (b21 - b11)
        matrixSubstration(b21, b11, subB, reduced_size);
        strassenMulti(a22, subB, s4, reduced_size);
    }
    #pragma omp taskwait
    {
        // S5 = (a11 + a12) b22
        matrixAddition(a11, a12, subA, reduced_size);
        strassenMulti(subA, b22, s5, reduced_size);
    }
    #pragma omp taskwait
    {
        // S6 = (a21 - a11) (b11 + b12)
        matrixSubstration(a21, a11, subA, reduced_size);
        matrixAddition(b11, b12, subB, reduced_size);
        strassenMulti(subA, subB, s6, reduced_size);
    }
    #pragma omp taskwait
    {
        // S7 = (a12 - a22) (b21 + b22)
        matrixSubstration(a12, a22, subA, reduced_size);
        matrixAddition(b21, b22, subB, reduced_size);
        strassenMulti(subA, subB, s7, reduced_size);
    }

    /*
        Application of Strassen's formulas
    */
    #pragma omp task
    {
        // C12 = S3 + S5
        matrixAddition(s3, s5, c12, reduced_size);
    }
    #pragma omp task
    {
        // C21 = S2 + S4
        matrixAddition(s2, s4, c21, reduced_size);
    }
    #pragma omp task
    {
        // C11 = S1 + S4 - S5 + S7
        matrixAddition(s1, s4, temp0, reduced_size);
        matrixSubstration(temp0, s5, temp1, reduced_size);
        matrixAddition(temp1, s7, c11, reduced_size);
    }
    #pragma omp taskwait
    {
        // C22 = S1 - S2 + S3 + S6
        matrixSubstration(s1, s2, temp0, reduced_size);
        matrixAddition(temp0, s3, temp1, reduced_size);
        matrixAddition(temp1, s6, c22, reduced_size);
    }


    // fill the result in matrix C
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < reduced_size ; i++)
    {
        for (int j = 0 ; j < reduced_size ; j++)
        {
            C[i][j] = c11[i][j];
            C[i][j + reduced_size] = c12[i][j];
            C[i + reduced_size][j] = c21[i][j];
            C[i + reduced_size][j + reduced_size] = c22[i][j];
        }
    }
}

/*--------------------------------------------Main-----------------------------------------------------------
------------------------------------------------------------------------------------------------------------*/

int main()
{
    /*
        System Info
    */
    printf("Code executed on:\n");
    printf("CPU name:      Intel Core i5 8250U CPU @1.60GHz\n");
    printf("Threading:     1 CPU - 4 Core - 8 Threads\n");
    printf("Memory (RAM):  8192 MB Dual Channel\n");
    printf("-------------------------------------------------\n\n");


    /*
        Matrix Init
    */


    int size = MATRIX_SIZE;
    double exe_time, exe_time_temp;
    clock_t start0, end0, start1, end1;

    if (size%2 != 0){
        printf("Size must be a power of 2.\n");
        return 0;
    }

    printf("Program started. Matrix of size: %d x %d\n\n", size, size);

    double **A = matrixGenerator(size, false);
    double **B = matrixGenerator(size, false);
    double **C = matrixGenerator(size, true);

    /*
        Traditional Multiplication Matrix
    */

    start0 = clock();
    traditionalMulti(A, B, C, size);
    end0 = clock();

    exe_time = (double)(end0 - start0) / CLOCKS_PER_SEC;
    printf("traditionalMulti execution time: %f\n\n", exe_time);
    //matrixViewer(C, size);

    C = matrixGenerator(size, true);

    /*
        Strassen Multiplication Matrix with OMP
    */

    #pragma omp parallel
    {
        #pragma omp single
        {
            start1 = clock();
            strassenMulti(A, B, C, size);
            end1 = clock();
        }
    }

    //matrixViewer(C, size);
    exe_time = (double)(end1 - start1) / CLOCKS_PER_SEC;
    exe_time_temp = (double) ALLOCATION_TIME / CLOCKS_PER_SEC;

    printf("strassenMulti execution time with allocation time:     %f\n", exe_time);
    printf("                             without allocation time:  %f\n\n", exe_time - exe_time_temp);

    checkResult(A,B,C,size);

    return 0;
}
