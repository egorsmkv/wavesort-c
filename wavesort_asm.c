// nasm -f elf64 -O3 wavesort.asm -o wavesort.o
// clang -std=c11 -mavx2 -mfma -lm -O3 -Wall -Wextra -o wavesort_asm wavesort_asm.c wavesort.o

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

// Declare external assembly function
extern void wave_sort(int32_t *arr, size_t n);

// Compare function for qsort
int compare_ints(const void *a, const void *b) {
    int32_t arg1 = *(const int32_t *)a;
    int32_t arg2 = *(const int32_t *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

// Validation function
bool is_sorted(int32_t *arr, size_t n) {
    for (size_t i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return false;
    }
    return true;
}

int main(void) {
    printf("Initializing Wave Sort Test...\n");
    
    const size_t ARR_SIZE = 100000000;
    int32_t *arr = (int32_t*)malloc(ARR_SIZE * sizeof(int32_t));
    int32_t *arr_qsort = (int32_t*)malloc(ARR_SIZE * sizeof(int32_t));
    
    if (!arr || !arr_qsort) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    srand((unsigned int)time(NULL));
    
    printf("Generating %zu random integers...\n", ARR_SIZE);
    for (size_t i = 0; i < ARR_SIZE; i++) {
        int32_t r = (rand() % 2000000) - 1000000;
        arr[i] = r;
        arr_qsort[i] = r;
    }

    // Benchmark QSort
    printf("Running qsort...\n");
    clock_t start = clock();
    qsort(arr_qsort, ARR_SIZE, sizeof(int32_t), compare_ints);
    clock_t end = clock();
    double qsort_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("qsort time: %.4f seconds\n", qsort_time);

    // Benchmark Wave Sort
    printf("Running wave_sort (ASM)...\n");
    start = clock();
    wave_sort(arr, ARR_SIZE);
    end = clock();
    double wave_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("wave_sort time: %.4f seconds\n", wave_time);

    // Validation
    if (is_sorted(arr, ARR_SIZE)) {
        printf("SUCCESS: Array is sorted correctly.\n");
    } else {
        printf("FAILURE: Array is NOT sorted.\n");
        // Check for mismatch
        for(size_t i=0; i<ARR_SIZE; i++) {
            if (arr[i] != arr_qsort[i]) {
                printf("Mismatch at index %zu: Expected %d, Got %d\n", i, arr_qsort[i], arr[i]);
                break;
            }
        }
    }

    free(arr);
    free(arr_qsort);
    return 0;
}
