/*
nasm -f elf64 -O3 wavesort_parallel.asm -o wavesort_parallel.o

clang -std=c17 -fopenmp -mavx2 -mfma -lm -O3 -Wall -Wextra -Wpedantic \
  -o wavesort_asm_parallel wavesort_asm_parallel.c wavesort_parallel.o
*/

#include <inttypes.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ============================================================================
// External ASM Kernels
// ============================================================================
// Assumed external linkage with C ABI
extern void wave_sort(int32_t *arr, size_t n);
extern void block_swap(int32_t *arr, size_t m, size_t r, size_t p);
extern size_t partition(int32_t *arr, size_t l, size_t r, size_t p_idx);

// ============================================================================
// Parallel Logic (C + ASM Kernels)
// ============================================================================
#define TASK_CUTOFF 8192

static void upwave_par(int32_t *restrict arr, size_t start, size_t end);

static void downwave_par(int32_t *restrict arr, size_t start,
                         size_t sorted_start, size_t end) {
  if (sorted_start == start) {
    return;
  }

  // Serial fallback for small chunks
  if (end - start < TASK_CUTOFF) {
    // Fallthrough to the recursive logic below (running on current thread)
    // unless a dedicated serial non-recursive kernel is available.
  }

  size_t p = sorted_start + (end - sorted_start) / 2;
  size_t m = partition(arr, start, sorted_start, p); // Optimized ASM Kernel

  if (m == sorted_start) {
    if (p == sorted_start) {
      if (sorted_start > 0) {
        upwave_par(arr, start, sorted_start - 1);
      }
      return;
    }
    if (p > 0) {
      downwave_par(arr, start, sorted_start, p - 1);
    }
    return;
  }

  block_swap(arr, m, sorted_start, p); // Optimized ASM Kernel

  if (m == start) {
    if (p == sorted_start) {
      upwave_par(arr, m + 1, end);
      return;
    }
    size_t p_next = p + 1;
    downwave_par(arr, m + p_next - sorted_start, p_next, end);
    return;
  }

  if (p == sorted_start) {
    if (m > 0) {
      upwave_par(arr, start, m - 1);
    }
    upwave_par(arr, m + 1, end);
    return;
  }

  size_t right_part_len = p - sorted_start;
  size_t split_point = m + right_part_len;

  // Parallel Split
  if (split_point > 0) {
// Use OpenMP 'if' clause to handle cutoff; runs serially if condition is false
#pragma omp task if (split_point - start > TASK_CUTOFF)
    downwave_par(arr, start, m, split_point - 1);
  }

  downwave_par(arr, split_point + 1, p + 1, end);
#pragma omp taskwait
}

static void upwave_par(int32_t *restrict arr, size_t start, size_t end) {
  if (start == end) {
    return;
  }
  size_t sorted_start = end;
  size_t sorted_len = 1;
  if (end == 0) {
    return;
  }
  size_t left_bound = end - 1;
  size_t total_len = end - start + 1;

  while (true) {
    downwave_par(arr, left_bound, sorted_start, end);
    sorted_start = left_bound;
    sorted_len = end - sorted_start + 1;

    // Check for sufficient growth to avoid infinite loops or inefficient steps
    if (total_len < (sorted_len << 2)) {
      break;
    }

    size_t next_expansion = (sorted_len << 1) + 1;
    if (end < next_expansion || (end - next_expansion) < start) {
      left_bound = start;
    } else {
      left_bound = end - next_expansion;
    }

    if (left_bound < start) {
      left_bound = start;
    }
    if (sorted_start == start) {
      break;
    }
  }
  downwave_par(arr, start, sorted_start, end);
}

void wave_sort_parallel(int32_t *restrict arr, size_t n) {
  if (!arr || n < 2) {
    return;
  }
#pragma omp parallel
  {
#pragma omp single
    upwave_par(arr, 0, n - 1);
  }
}

// ============================================================================
// Test Driver
// ============================================================================

int compare_ints(const void *a, const void *b) {
  int32_t arg1 = *(const int32_t *)a;
  int32_t arg2 = *(const int32_t *)b;
  if (arg1 < arg2)
    return -1;
  if (arg1 > arg2)
    return 1;
  return 0;
}

bool is_sorted(const int32_t *arr, size_t n) {
  for (size_t i = 0; i < n - 1; i++) {
    if (arr[i] > arr[i + 1]) {
      return false;
    }
  }
  return true;
}

int main(void) {
  printf("Initializing W-Sort Benchmark...\n");
  printf("Cores available: %d\n", omp_get_max_threads());

  const size_t ARR_SIZE = 100000000;

  int32_t *arr_base = malloc(ARR_SIZE * sizeof(*arr_base));
  int32_t *arr_qsort = malloc(ARR_SIZE * sizeof(*arr_qsort));
  int32_t *arr_asm = malloc(ARR_SIZE * sizeof(*arr_asm));
  int32_t *arr_par = malloc(ARR_SIZE * sizeof(*arr_par));

  if (!arr_base || !arr_qsort || !arr_asm || !arr_par) {
    fprintf(stderr, "Memory allocation failed\n");
    // Freeing partially allocated memory is good practice,
    // though OS cleans up on exit.
    free(arr_base);
    free(arr_qsort);
    free(arr_asm);
    free(arr_par);
    return 1;
  }

  srand((unsigned int)time(NULL));

  printf("Generating %zu random integers...\n", ARR_SIZE);
  for (size_t i = 0; i < ARR_SIZE; i++) {
    // rand() range varies by platform; keeping original logic logic.
    int32_t r = (int32_t)((rand() % 2000000) - 1000000);
    arr_base[i] = r;
    arr_qsort[i] = r;
    arr_asm[i] = r;
    arr_par[i] = r;
  }

  // 1. QSort
  printf("\nRunning qsort...\n");
  clock_t start = clock();
  qsort(arr_qsort, ARR_SIZE, sizeof(int32_t), compare_ints);
  clock_t end = clock();
  printf("qsort time: %.4f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

  // 2. Serial ASM
  printf("\nRunning Serial ASM wave_sort...\n");
  start = clock();
  wave_sort(arr_asm, ARR_SIZE);
  end = clock();
  printf("Serial ASM time: %.4f seconds\n",
         (double)(end - start) / CLOCKS_PER_SEC);
  if (!is_sorted(arr_asm, ARR_SIZE)) {
    printf("Serial ASM FAILED!\n");
  }

  // 3. Parallel C + ASM Kernels
  printf("\nRunning Parallel C+ASM wave_sort (OMP)...\n");
  double omp_start = omp_get_wtime();
  wave_sort_parallel(arr_par, ARR_SIZE);
  double omp_end = omp_get_wtime();
  printf("Parallel C+ASM time: %.4f seconds\n", omp_end - omp_start);
  if (!is_sorted(arr_par, ARR_SIZE)) {
    printf("Parallel C+ASM FAILED!\n");
  }

  free(arr_base);
  free(arr_qsort);
  free(arr_asm);
  free(arr_par);

  return 0;
}
