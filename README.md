# `wavesort`

WaveSort algorithm implemented in C

## Demo

```
Initializing benchmark for 100000000 integer samples...
Data generated. Starting sort...

[OK] WaveSort Verification Passed.
[OK] Qsort    Verification Passed.

--- Results (Lower is Better) ---
WaveSort: 7.786132 seconds
Qsort:    9.737445 seconds

Ratio (Wave/Qsort): 0.80x (WaveSort is faster)
```

### Assembly 

```
Initializing Wave Sort Test...
Generating 100000000 random integers...
Running qsort...
qsort time: 9.0394 seconds
Running wave_sort (ASM)...
wave_sort time: 3.8327 seconds
SUCCESS: Array is sorted correctly.
```

### Assembly (serial vs. parallel)

```
Initializing Wave Sort Benchmark...
Cores available: 24
Generating 1000000000 random integers...

Running qsort...
qsort skipped for large array size to save time.
qsort time: 0.0000 seconds

Running Serial ASM wave_sort...
Serial ASM time: 39.2279 seconds

Running Parallel C+ASM wave_sort (OMP)...
Parallel C+ASM time: 7.6819 seconds
```
