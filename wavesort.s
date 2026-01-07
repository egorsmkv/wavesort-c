/*
 * ============================================================================
 * WaveSort Implementation for Apple Silicon (ARM64)
 * ============================================================================
 *
 * ABI Compliance: AAPCS64
 * Target: Apple M1/M2/M3 (Mach-O format)
 *
 * Registers:
 * x0 - x7   : Arguments / Scratch (Caller-saved)
 * x8        : Indirect result location / Scratch
 * x9 - x15  : Scratch (Caller-saved)
 * x16 - x17 : Intra-procedure-call scratch
 * x18       : RESERVED (Platform register on macOS)
 * x19 - x28 : Callee-saved (Must preserve if used)
 * x29       : Frame Pointer (FP)
 * x30       : Link Register (LR)
 *
 * Data Types:
 * int32_t   : 32-bit signed integer (w registers)
 * size_t    : 64-bit unsigned integer (x registers)
 */

.text
.p2align 2

// ==========================================
// Helper: swap
// Not exported, inlined manually where critical, 
// or called via bl if code density is preferred.
// Signature: void swap(int32_t *a, int32_t *b)
// Inputs: x0 = ptr_a, x1 = ptr_b
// ==========================================
_swap_internal:
    ldr     w9, [x0]
    ldr     w10, [x1]
    str     w10, [x0]
    str     w9, [x1]
    ret

// ==========================================
// Function: block_swap_sl
// Signature: void block_swap_sl(int32_t *arr, size_t m, size_t p, size_t ll)
// Inputs: x0=arr, x1=m, x2=p, x3=ll
// ==========================================
_block_swap_sl:
    // Leaf function, no stack frame needed unless we run out of regs.
    // We need:
    // arr (x0), m (x1), p (x2), ll (x3)
    // tmp (w4), init (x5), j (x6), nm (x7), total_len (x8)
    // count (x9), k (x10), scratch (x11, w12)

    ldr     w4, [x0, x1, lsl #2]    // tmp = arr[m]
    mov     x5, x1                  // init = m
    mov     x6, x1                  // j = m
    
    // nm = p - ll + 1
    sub     x7, x2, x3
    add     x7, x7, #1
    
    // total_len = p - m + 1
    sub     x8, x2, x1
    add     x8, x8, #1
    
    mov     x9, #0                  // count = 0

1:  // Loop start
    cmp     x9, x8
    b.hs    2f                      // if count >= total_len, break

    // if (j >= nm)
    cmp     x6, x7
    b.lo    3f                      // Jump to else

    // Then block (j >= nm)
    // size_t k = j - nm + m
    sub     x10, x6, x7
    add     x10, x10, x1
    
    cmp     x10, x5                 // if (k == init)
    b.ne    4f
    
    // k == init
    add     x5, x5, #1              // init++
    str     w4, [x0, x6, lsl #2]    // arr[j] = tmp
    mov     x6, x5                  // j = init
    ldr     w4, [x0, x6, lsl #2]    // tmp = arr[j]
    b       5f                      // End of iteration

4:  // k != init (inside Then block)
    ldr     w12, [x0, x10, lsl #2]  // load arr[k]
    str     w12, [x0, x6, lsl #2]   // arr[j] = arr[k]
    mov     x6, x10                 // j = k
    b       5f

3:  // Else block (j < nm)
    // size_t k = j + ll
    add     x10, x6, x3
    ldr     w12, [x0, x10, lsl #2]  // load arr[k]
    str     w12, [x0, x6, lsl #2]   // arr[j] = arr[k]
    mov     x6, x10                 // j = k

5:  // Loop increment
    add     x9, x9, #1
    b       1b

2:  // End
    ret

// ==========================================
// Function: block_swap_sr
// Signature: void block_swap_sr(int32_t *arr, size_t m, size_t r, size_t p)
// Inputs: x0=arr, x1=m, x2=r, x3=p
// ==========================================
_block_swap_sr:
    // i = m (x1 is i)
    ldr     w4, [x0, x1, lsl #2]    // tmp = arr[i]
    mov     x5, x2                  // j = r

1:  // While loop
    cmp     x5, x3                  // while (j < p)
    b.hs    2f
    
    ldr     w6, [x0, x5, lsl #2]    // load arr[j]
    str     w6, [x0, x1, lsl #2]    // arr[i] = arr[j]
    add     x1, x1, #1              // i++
    
    ldr     w6, [x0, x1, lsl #2]    // load arr[i] (new i)
    str     w6, [x0, x5, lsl #2]    // arr[j] = arr[i]
    add     x5, x5, #1              // j++
    b       1b

2:  // After loop
    ldr     w6, [x0, x5, lsl #2]    // load arr[j]
    str     w6, [x0, x1, lsl #2]    // arr[i] = arr[j]
    str     w4, [x0, x5, lsl #2]    // arr[j] = tmp
    ret

// ==========================================
// Function: block_swap
// Signature: void block_swap(int32_t *arr, size_t m, size_t r, size_t p)
// Inputs: x0=arr, x1=m, x2=r, x3=p
// ==========================================
_block_swap:
    // ll = r - m
    sub     x4, x2, x1
    cbz     x4, 3f                  // if (ll == 0) return

    // lr = p - r + 1
    sub     x5, x3, x2
    add     x5, x5, #1
    
    cmp     x5, #1                  // if (lr == 1)
    b.ne    1f
    
    // swap(&arr[m], &arr[p])
    // Calculate addresses
    add     x9, x0, x1, lsl #2
    add     x10, x0, x3, lsl #2
    
    // Inline swap
    ldr     w11, [x9]
    ldr     w12, [x10]
    str     w12, [x9]
    str     w11, [x10]
    ret

1:  // Check lr <= ll
    cmp     x5, x4
    b.hi    2f
    
    // Tail call block_swap_sr(arr, m, r, p)
    // Args x0, x1, x2, x3 are already set correctly
    b       _block_swap_sr

2:  // Else call block_swap_sl(arr, m, p, ll)
    // Map args: x0=arr (ok), x1=m (ok), x2=p (move x3->x2), x3=ll (move x4->x3)
    mov     x2, x3
    mov     x3, x4
    b       _block_swap_sl

3:
    ret

// ==========================================
// Function: partition
// Signature: size_t partition(int32_t *arr, size_t l, size_t r, size_t p_idx)
// Inputs: x0=arr, x1=l, x2=r, x3=p_idx
// Output: x0 = return value (i)
// ==========================================
_partition:
    // pivot_val = arr[p_idx]
    ldr     w4, [x0, x3, lsl #2]    // w4 = pivot_val
    
    sub     x1, x1, #1              // i = l - 1
    // x2 is j (already r)
    // x0 is arr base

    // Registers:
    // x1: i
    // x2: j
    // w4: pivot_val
    // w5: arr[i]
    // w6: arr[j]
    // x0: arr base

1:  // Outer loop (while true)

    // Inner loop 1: while (true) { i++; ... }
2:
    add     x1, x1, #1
    cmp     x1, x2
    b.eq    5f                      // if (i == j) return i
    
    ldr     w5, [x0, x1, lsl #2]    // load arr[i]
    cmp     w5, w4
    b.ge    3f                      // if (arr[i] >= pivot_val) break
    b       2b

3:  // Inner loop 2: while (true) { j--; ... }
    sub     x2, x2, #1
    cmp     x2, x1
    b.eq    5f                      // if (j == i) return i
    
    ldr     w6, [x0, x2, lsl #2]    // load arr[j]
    cmp     w6, w4
    b.le    4f                      // if (arr[j] <= pivot_val) break
    b       3b

4:  // swap(&arr[i], &arr[j])
    // w5 has arr[i], w6 has arr[j] from comparisons
    // But we need to verify strict ordering isn't violated or values changed?
    // Actually, simple load/store swap logic is safe here.
    str     w6, [x0, x1, lsl #2]
    str     w5, [x0, x2, lsl #2]
    b       1b                      // Continue outer loop

5:  // Return i
    mov     x0, x1
    ret

// ==========================================
// Function: downwave
// Signature: void downwave(int32_t *arr, size_t start, size_t sorted_start, size_t end)
// Inputs: x0=arr, x1=start, x2=sorted_start, x3=end
// Note: Recursive. Needs callee-saved regs.
// ==========================================
_downwave:
    // Prologue
    stp     x29, x30, [sp, -80]!    // Alloc 80 bytes (align 16)
    mov     x29, sp
    // Save callee-saved registers we will use to persist variables across calls
    stp     x19, x20, [sp, 16]      // x19=arr, x20=start
    stp     x21, x22, [sp, 32]      // x21=sorted_start, x22=end
    stp     x23, x24, [sp, 48]      // x23=p, x24=m
    str     x25, [sp, 64]           // x25=scratch/p_next

    // if (sorted_start == start) return;
    cmp     x2, x1
    b.eq    _downwave_exit

    // Save args to callee-saved regs
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3

    // size_t p = sorted_start + (end - sorted_start) / 2;
    sub     x8, x22, x21
    lsr     x8, x8, #1
    add     x23, x21, x8            // x23 = p

    // m = partition(arr, start, sorted_start, p);
    // x0 is already arr
    // x1 is start
    // x2 is sorted_start
    mov     x3, x23                 // p_idx = p
    bl      _partition
    mov     x24, x0                 // m = result

    // if (m == sorted_start)
    cmp     x24, x21
    b.ne    _downwave_m_ne_sorted

    // Case: m == sorted_start
    cmp     x23, x21                // if (p == sorted_start)
    b.ne    1f

    // p == sorted_start
    cmp     x21, #0                 // if (sorted_start > 0)
    b.eq    _downwave_exit
    
    // upwave(arr, start, sorted_start - 1)
    mov     x0, x19
    mov     x1, x20
    sub     x2, x21, #1
    bl      _upwave
    b       _downwave_exit

1:  // p > 0 (implied by p != sorted_start and logic context, but check strictly if p > 0)
    // Logic: if (p > 0) downwave(arr, start, sorted_start, p - 1)
    cmp     x23, #0
    b.eq    _downwave_exit
    mov     x0, x19
    mov     x1, x20
    mov     x2, x21
    sub     x3, x23, #1
    bl      _downwave
    b       _downwave_exit

_downwave_m_ne_sorted:
    // block_swap(arr, m, sorted_start, p)
    mov     x0, x19
    mov     x1, x24
    mov     x2, x21
    mov     x3, x23
    bl      _block_swap

    // if (m == start)
    cmp     x24, x20
    b.ne    _downwave_m_ne_start

    // Case: m == start
    cmp     x23, x21                // if (p == sorted_start)
    b.ne    2f
    
    // upwave(arr, m + 1, end)
    mov     x0, x19
    add     x1, x24, #1
    mov     x2, x22
    bl      _upwave
    b       _downwave_exit

2:  // size_t p_next = p + 1;
    add     x25, x23, #1
    // downwave(arr, m + p_next - sorted_start, p_next, end)
    mov     x0, x19
    add     x1, x24, x25
    sub     x1, x1, x21             // start param
    mov     x2, x25
    mov     x3, x22
    bl      _downwave
    b       _downwave_exit

_downwave_m_ne_start:
    // if (p == sorted_start)
    cmp     x23, x21
    b.ne    _downwave_final_split

    cmp     x24, #0                 // if (m > 0)
    b.eq    3f
    // upwave(arr, start, m - 1)
    mov     x0, x19
    mov     x1, x20
    sub     x2, x24, #1
    bl      _upwave
3:
    // upwave(arr, m + 1, end)
    mov     x0, x19
    add     x1, x24, #1
    mov     x2, x22
    bl      _upwave
    b       _downwave_exit

_downwave_final_split:
    // size_t right_part_len = p - sorted_start;
    sub     x8, x23, x21
    // size_t split_point = m + right_part_len;
    add     x25, x24, x8            // x25 = split_point

    // if (split_point > 0)
    cmp     x25, #0
    b.eq    4f
    
    // downwave(arr, start, m, split_point - 1)
    mov     x0, x19
    mov     x1, x20
    mov     x2, x24
    sub     x3, x25, #1
    bl      _downwave

4:
    // downwave(arr, split_point + 1, p + 1, end)
    mov     x0, x19
    add     x1, x25, #1
    add     x2, x23, #1
    mov     x3, x22
    bl      _downwave

_downwave_exit:
    // Epilogue
    ldp     x19, x20, [sp, 16]
    ldp     x21, x22, [sp, 32]
    ldp     x23, x24, [sp, 48]
    ldr     x25, [sp, 64]
    ldp     x29, x30, [sp], 80
    ret

// ==========================================
// Function: upwave
// Signature: void upwave(int32_t *arr, size_t start, size_t end)
// Inputs: x0=arr, x1=start, x2=end
// ==========================================
_upwave:
    // Prologue
    stp     x29, x30, [sp, -64]!
    mov     x29, sp
    stp     x19, x20, [sp, 16]      // x19=arr, x20=start
    stp     x21, x22, [sp, 32]      // x21=end, x22=sorted_start
    stp     x23, x24, [sp, 48]      // x23=left_bound, x24=sorted_len

    // if (start == end) return;
    cmp     x1, x2
    b.eq    _upwave_exit

    // if (end == 0) return;
    cbz     x2, _upwave_exit

    mov     x19, x0
    mov     x20, x1
    mov     x21, x2

    // sorted_start = end
    mov     x22, x21
    // sorted_len = 1
    mov     x24, #1
    // left_bound = end - 1
    sub     x23, x21, #1
    // total_len = end - start + 1 (kept in reg or calc'd)
    // We check total_len inside loop

1:  // Loop while(true)
    // downwave(arr, left_bound, sorted_start, end)
    mov     x0, x19
    mov     x1, x23
    mov     x2, x22
    mov     x3, x21
    bl      _downwave

    // sorted_start = left_bound
    mov     x22, x23
    
    // sorted_len = end - sorted_start + 1
    sub     x24, x21, x22
    add     x24, x24, #1

    // total_len = end - start + 1
    sub     x8, x21, x20
    add     x8, x8, #1

    // if (total_len < (sorted_len << 2)) break;
    lsl     x9, x24, #2
    cmp     x8, x9
    b.lo    2f

    // size_t next_expansion = (sorted_len << 1) + 1;
    lsl     x10, x24, #1
    add     x10, x10, #1

    // if (end < next_expansion || (end - next_expansion) < start)
    cmp     x21, x10
    b.lo    _upwave_set_lb_start
    
    sub     x11, x21, x10
    cmp     x11, x20
    b.lo    _upwave_set_lb_start

    // else left_bound = end - next_expansion
    mov     x23, x11
    b       _upwave_check_lb_lower

_upwave_set_lb_start:
    mov     x23, x20

_upwave_check_lb_lower:
    // if (left_bound < start) left_bound = start
    cmp     x23, x20
    csel    x23, x20, x23, lo

    // if (sorted_start == start) break;
    cmp     x22, x20
    b.eq    2f

    b       1b  // Continue loop

2:  // After loop
    // downwave(arr, start, sorted_start, end)
    mov     x0, x19
    mov     x1, x20
    mov     x2, x22
    mov     x3, x21
    bl      _downwave

_upwave_exit:
    ldp     x19, x20, [sp, 16]
    ldp     x21, x22, [sp, 32]
    ldp     x23, x24, [sp, 48]
    ldp     x29, x30, [sp], 64
    ret

// ==========================================
// Function: wave_sort
// Signature: void wave_sort(int32_t *arr, size_t n)
// Inputs: x0=arr, x1=n
// ==========================================
.global _wave_sort
_wave_sort:
    stp     x29, x30, [sp, -16]!
    mov     x29, sp

    // if (!arr || n < 2) return;
    cbz     x0, 1f
    cmp     x1, #2
    b.lo    1f

    // upwave(arr, 0, n - 1)
    sub     x2, x1, #1  // end = n - 1
    mov     x1, #0      // start = 0
    bl      _upwave

1:
    ldp     x29, x30, [sp], 16
    ret
