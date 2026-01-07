/*
 * ============================================================================
 * WaveSort Implementation for Apple Silicon (ARM64) - Optimized
 * ============================================================================
 *
 * ABI Compliance: AAPCS64
 * Target: Apple M1/M2/M3 (Mach-O format)
 * Optimizations: NEON SIMD (Partitioning & Block Swaps), Loop Unrolling
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
.p2align 4

// ==========================================
// Helper: swap
// ==========================================
_swap_internal:
    ldr     w9, [x0]
    ldr     w10, [x1]
    str     w10, [x0]
    str     w9, [x1]
    ret

// ==========================================
// Function: block_swap_sl
// Scalar implementation due to complex cyclic dependency
// ==========================================
.p2align 4
_block_swap_sl:
    // x0=arr, x1=m, x2=p, x3=ll
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

L_bsl_loop:
    cmp     x9, x8
    b.hs    L_bsl_end               // if count >= total_len, break

    cmp     x6, x7
    b.lo    L_bsl_else              // if j < nm

    // Then block (j >= nm)
    sub     x10, x6, x7
    add     x10, x10, x1            // k = j - nm + m
    
    cmp     x10, x5                 // if (k == init)
    b.ne    L_bsl_k_ne_init
    
    // k == init
    add     x5, x5, #1              // init++
    str     w4, [x0, x6, lsl #2]    // arr[j] = tmp
    mov     x6, x5                  // j = init
    ldr     w4, [x0, x6, lsl #2]    // tmp = arr[j]
    b       L_bsl_next

L_bsl_k_ne_init:
    ldr     w12, [x0, x10, lsl #2]  // load arr[k]
    str     w12, [x0, x6, lsl #2]   // arr[j] = arr[k]
    mov     x6, x10                 // j = k
    b       L_bsl_next

L_bsl_else:
    add     x10, x6, x3             // k = j + ll
    ldr     w12, [x0, x10, lsl #2]  // load arr[k]
    str     w12, [x0, x6, lsl #2]   // arr[j] = arr[k]
    mov     x6, x10                 // j = k

L_bsl_next:
    add     x9, x9, #1
    b       L_bsl_loop

L_bsl_end:
    ret

// ==========================================
// Function: block_swap_sr
// Optimized with NEON for shifting blocks
// ==========================================
.p2align 4
_block_swap_sr:
    // x0=arr, x1=m (i), x2=r (j), x3=p
    // Logic: while (j < p) { arr[i]=arr[j]; arr[j]=arr[i+1]; i++; j++; }
    
    ldr     w4, [x0, x1, lsl #2]    // tmp = arr[i] (initial)
    mov     x5, x2                  // j = r
    
    // Safety check: The vectorized rotation assumes a safe distance between i and j
    // to avoid reading data that was just written in a way that breaks the rotation logic.
    // Increased safety margin to 32 elements to prevent edge case overlaps.
    sub     x9, x2, x1              // D = r - m
    cmp     x9, #32
    b.lo    L_bsr_scalar_loop

    // Also check total loop count (p - j) >= 4 for SIMD entry
    sub     x9, x3, x5
    cmp     x9, #4
    b.lo    L_bsr_scalar_loop

L_bsr_neon_loop:
    // Loop while j <= p - 4
    sub     x10, x3, #4

L_bsr_neon_inner:
    cmp     x5, x10
    b.hi    L_bsr_scalar_loop

    // Operations:
    // 1. dest_A (arr[i...i+3]) gets src_A (arr[j...j+3])
    // 2. dest_B (arr[j...j+3]) gets src_B (arr[i+1...i+4])
    // Load src_B FIRST to capture state before dest_B overwrite if overlap exists
    add     x11, x1, #1             // i+1
    add     x12, x0, x11, lsl #2    // addr = arr + (i+1)*4
    ldr     q1, [x12]
    
    // Load src_A
    add     x13, x0, x5, lsl #2     // addr = arr + j*4
    ldr     q0, [x13]

    // Store dest_A
    add     x14, x0, x1, lsl #2     // addr = arr + i*4
    str     q0, [x14]

    // Store dest_B
    str     q1, [x13]               // addr = arr + j*4

    add     x1, x1, #4
    add     x5, x5, #4
    b       L_bsr_neon_inner

L_bsr_scalar_loop:
    cmp     x5, x3
    b.hs    L_bsr_finish
    
    ldr     w6, [x0, x5, lsl #2]    // load arr[j]
    str     w6, [x0, x1, lsl #2]    // arr[i] = arr[j]
    add     x1, x1, #1              // i++
    
    ldr     w6, [x0, x1, lsl #2]    // load arr[i] (new i)
    str     w6, [x0, x5, lsl #2]    // arr[j] = arr[i]
    add     x5, x5, #1              // j++
    b       L_bsr_scalar_loop

L_bsr_finish:
    ldr     w6, [x0, x5, lsl #2]    // load arr[j]
    str     w6, [x0, x1, lsl #2]    // arr[i] = arr[j]
    str     w4, [x0, x5, lsl #2]    // arr[j] = tmp
    ret

// ==========================================
// Function: block_swap
// ==========================================
.p2align 4
_block_swap:
    sub     x4, x2, x1              // ll = r - m
    cbz     x4, L_bs_ret            // if (ll == 0) return

    sub     x5, x3, x2              // lr_raw = p - r
    add     x5, x5, #1              // lr = p - r + 1
    
    cmp     x5, #1
    b.ne    L_bs_check_lr
    
    // swap(&arr[m], &arr[p])
    add     x9, x0, x1, lsl #2
    add     x10, x0, x3, lsl #2
    ldr     w11, [x9]
    ldr     w12, [x10]
    str     w12, [x9]
    str     w11, [x10]
    ret

L_bs_check_lr:
    cmp     x5, x4
    b.hi    L_bs_call_sl
    
    b       _block_swap_sr

L_bs_call_sl:
    mov     x2, x3
    mov     x3, x4
    b       _block_swap_sl

L_bs_ret:
    ret

// ==========================================
// Function: partition
// Optimized with NEON SIMD Scanning
// Inputs: x0=arr, x1=l, x2=r, x3=p_idx
// ==========================================
.p2align 4
_partition:
    // Load pivot
    ldr     w4, [x0, x3, lsl #2]
    
    sub     x1, x1, #1              // i = l - 1
    // x2 is j (already r)
    
    // Small array optimization: If range < 64, skip NEON setup
    sub     x9, x2, x1
    cmp     x9, #64
    b.lo    L_partition_outer_scalar

    // Replicate pivot to NEON register v0 for SIMD comparison
    dup     v0.4s, w4

L_partition_outer:

    // --- Inner Loop 1: i++ while arr[i] < pivot ---
L_scan_i_start:
    // Check safety margin for vector load (need 8 elements to be safe/worth it)
    sub     x9, x2, x1
    cmp     x9, #8
    b.lo    L_scan_i_scalar

    // Pre-calculate address for next load: arr[i+1]
    add     x10, x1, #1
    add     x11, x0, x10, lsl #2
    ld1     {v1.4s}, [x11]
    
    // Compare: v2 = (v1 >= v0) -> -1 if true, 0 if false
    cmge    v2.4s, v1.4s, v0.4s
    
    // Check if any element matches
    umaxv   s3, v2.4s
    fmov    w9, s3
    cbnz    w9, L_scan_i_scalar
    
    // No match, advance by 4
    add     x1, x1, #4
    b       L_scan_i_start

L_scan_i_scalar:
    add     x1, x1, #1
    cmp     x1, x2
    b.eq    L_partition_done
    
    ldr     w5, [x0, x1, lsl #2]
    cmp     w5, w4
    b.lt    L_scan_i_scalar         // if arr[i] < pivot, continue
    // Found arr[i] >= pivot, break to scan_j

    // --- Inner Loop 2: j-- while arr[j] > pivot ---
L_scan_j_start:
    sub     x9, x2, x1
    cmp     x9, #8
    b.lo    L_scan_j_scalar
    
    // Load arr[j-4...j-1]
    // Crucial fix: The C loop decrement (j--) happens BEFORE the check.
    // The previous code checked arr[j-3...j], including arr[j].
    // arr[j] is the exclusive bound and must NOT be checked.
    // We now shift the window to j-4 ... j-1.
    sub     x10, x2, #4
    add     x11, x0, x10, lsl #2
    ld1     {v1.4s}, [x11]
    
    // Compare: v2 = (v1 <= v0)
    cmle    v2.4s, v1.4s, v0.4s
    
    umaxv   s3, v2.4s
    fmov    w9, s3
    cbnz    w9, L_scan_j_scalar
    
    sub     x2, x2, #4
    b       L_scan_j_start

L_scan_j_scalar:
    sub     x2, x2, #1
    cmp     x2, x1
    b.eq    L_partition_done
    
    ldr     w6, [x0, x2, lsl #2]
    cmp     w6, w4
    b.gt    L_scan_j_scalar         // if arr[j] > pivot, continue

L_partition_swap:
    str     w6, [x0, x1, lsl #2]
    str     w5, [x0, x2, lsl #2]
    b       L_partition_outer

    // --- Scalar-Only Outer Loop (for small ranges) ---
L_partition_outer_scalar:
L_scan_i_pure_scalar:
    add     x1, x1, #1
    cmp     x1, x2
    b.eq    L_partition_done
    ldr     w5, [x0, x1, lsl #2]
    cmp     w5, w4
    b.lt    L_scan_i_pure_scalar

L_scan_j_pure_scalar:
    sub     x2, x2, #1
    cmp     x2, x1
    b.eq    L_partition_done
    ldr     w6, [x0, x2, lsl #2]
    cmp     w6, w4
    b.gt    L_scan_j_pure_scalar

    // Swap
    str     w6, [x0, x1, lsl #2]
    str     w5, [x0, x2, lsl #2]
    b       L_partition_outer_scalar

L_partition_done:
    mov     x0, x1
    ret

// ==========================================
// Function: downwave
// ==========================================
.p2align 4
_downwave:
    stp     x29, x30, [sp, -80]!
    mov     x29, sp
    stp     x19, x20, [sp, 16]
    stp     x21, x22, [sp, 32]
    stp     x23, x24, [sp, 48]
    str     x25, [sp, 64]

    cmp     x2, x1
    b.eq    L_dw_exit

    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3

    sub     x8, x22, x21
    lsr     x8, x8, #1
    add     x23, x21, x8            // p

    mov     x3, x23
    bl      _partition
    mov     x24, x0                 // m

    cmp     x24, x21
    b.ne    L_dw_m_ne_sorted

    // m == sorted_start
    cmp     x23, x21
    b.ne    1f
    cmp     x21, #0
    b.eq    L_dw_exit
    mov     x0, x19
    mov     x1, x20
    sub     x2, x21, #1
    bl      _upwave
    b       L_dw_exit

1:  // p > sorted_start
    cmp     x23, #0
    b.eq    L_dw_exit
    mov     x0, x19
    mov     x1, x20
    mov     x2, x21
    sub     x3, x23, #1
    bl      _downwave
    b       L_dw_exit

L_dw_m_ne_sorted:
    mov     x0, x19
    mov     x1, x24
    mov     x2, x21
    mov     x3, x23
    bl      _block_swap

    cmp     x24, x20
    b.ne    L_dw_m_ne_start

    // m == start
    cmp     x23, x21
    b.ne    2f
    mov     x0, x19
    add     x1, x24, #1
    mov     x2, x22
    bl      _upwave
    b       L_dw_exit

2:  add     x25, x23, #1
    mov     x0, x19
    add     x1, x24, x25
    sub     x1, x1, x21
    mov     x2, x25
    mov     x3, x22
    bl      _downwave
    b       L_dw_exit

L_dw_m_ne_start:
    cmp     x23, x21
    b.ne    L_dw_final

    cmp     x24, #0
    b.eq    3f
    mov     x0, x19
    mov     x1, x20
    sub     x2, x24, #1
    bl      _upwave
3:  mov     x0, x19
    add     x1, x24, #1
    mov     x2, x22
    bl      _upwave
    b       L_dw_exit

L_dw_final:
    sub     x8, x23, x21
    add     x25, x24, x8            // split_point
    cmp     x25, #0
    b.eq    4f
    mov     x0, x19
    mov     x1, x20
    mov     x2, x24
    sub     x3, x25, #1
    bl      _downwave
4:  mov     x0, x19
    add     x1, x25, #1
    add     x2, x23, #1
    mov     x3, x22
    bl      _downwave

L_dw_exit:
    ldp     x19, x20, [sp, 16]
    ldp     x21, x22, [sp, 32]
    ldp     x23, x24, [sp, 48]
    ldr     x25, [sp, 64]
    ldp     x29, x30, [sp], 80
    ret

// ==========================================
// Function: upwave
// ==========================================
.p2align 4
_upwave:
    stp     x29, x30, [sp, -64]!
    mov     x29, sp
    stp     x19, x20, [sp, 16]
    stp     x21, x22, [sp, 32]
    stp     x23, x24, [sp, 48]

    cmp     x1, x2
    b.eq    L_uw_exit
    cbz     x2, L_uw_exit

    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x21
    mov     x24, #1
    sub     x23, x21, #1

L_uw_loop:
    mov     x0, x19
    mov     x1, x23
    mov     x2, x22
    mov     x3, x21
    bl      _downwave

    mov     x22, x23
    sub     x24, x21, x22
    add     x24, x24, #1
    sub     x8, x21, x20
    add     x8, x8, #1

    lsl     x9, x24, #2
    cmp     x8, x9
    b.lo    L_uw_after

    lsl     x10, x24, #1
    add     x10, x10, #1
    cmp     x21, x10
    b.lo    L_uw_set_start
    sub     x11, x21, x10
    cmp     x11, x20
    b.lo    L_uw_set_start
    mov     x23, x11
    b       L_uw_check_start

L_uw_set_start:
    mov     x23, x20

L_uw_check_start:
    cmp     x23, x20
    csel    x23, x20, x23, lo
    cmp     x22, x20
    b.eq    L_uw_after
    b       L_uw_loop

L_uw_after:
    mov     x0, x19
    mov     x1, x20
    mov     x2, x22
    mov     x3, x21
    bl      _downwave

L_uw_exit:
    ldp     x19, x20, [sp, 16]
    ldp     x21, x22, [sp, 32]
    ldp     x23, x24, [sp, 48]
    ldp     x29, x30, [sp], 64
    ret

// ==========================================
// Function: wave_sort
// ==========================================
.global _wave_sort
.p2align 4
_wave_sort:
    stp     x29, x30, [sp, -16]!
    mov     x29, sp

    cbz     x0, 1f
    cmp     x1, #2
    b.lo    1f

    sub     x2, x1, #1
    mov     x1, #0
    bl      _upwave

1:
    ldp     x29, x30, [sp], 16
    ret
