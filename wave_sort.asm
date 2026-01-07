; ==============================================================================
; Wave Sort - Optimized AMD64 Assembly Implementation
; Target: AMD64 (x86_64) with AVX2 support
; ==============================================================================

section .text
global wave_sort

; ==============================================================================
; Helper Macros and Constants
; ==============================================================================

%define SIZEOF_INT 4

; ==============================================================================
; Function: swap
; RDI = int32_t *a
; RSI = int32_t *b
; ==============================================================================
swap:
    mov     eax, [rdi]
    mov     ecx, [rsi]
    mov     [rdi], ecx
    mov     [rsi], eax
    ret

; ==============================================================================
; Function: block_swap_sl
; Signature: void block_swap_sl(int32_t *arr, size_t m, size_t p, size_t ll)
; Params: RDI=arr, RSI=m, RDX=p, RCX=ll
; ==============================================================================
block_swap_sl:
    push    r12
    push    r13
    push    rbx

    ; RDI=arr, RSI=m, RDX=p, RCX=ll

    mov     r8d, [rdi + rsi*4]  ; tmp = arr[m]
    mov     r9, rsi             ; init = m
    mov     r10, rsi            ; j = m
    
    ; nm = p - ll + 1
    mov     r11, rdx
    sub     r11, rcx
    inc     r11

    ; total_len = p - m + 1
    mov     r13, rdx
    sub     r13, rsi
    inc     r13

    xor     r12, r12            ; count = 0

.loop_body:
    cmp     r12, r13
    jge     .exit_sl

    cmp     r10, r11            ; if (j >= nm)
    jb      .sl_else

    ; k = j - nm + m
    ; k is needed temp. Use RBX.
    mov     rbx, r10
    sub     rbx, r11
    add     rbx, rsi            ; rsi is still 'm'

    cmp     rbx, r9             ; if (k == init)
    jne     .sl_cycle_cont

    ; init++; arr[j] = tmp; j = init; tmp = arr[j];
    inc     r9
    mov     [rdi + r10*4], r8d
    mov     r10, r9
    mov     r8d, [rdi + r10*4]
    jmp     .sl_next

.sl_cycle_cont:
    ; arr[j] = arr[k]; j = k;
    mov     eax, [rdi + rbx*4]
    mov     [rdi + r10*4], eax
    mov     r10, rbx
    jmp     .sl_next

.sl_else:
    ; size_t k = j + ll;
    mov     rbx, r10
    add     rbx, rcx
    
    ; arr[j] = arr[k]; j = k;
    mov     eax, [rdi + rbx*4]
    mov     [rdi + r10*4], eax
    mov     r10, rbx

.sl_next:
    inc     r12
    jmp     .loop_body

.exit_sl:
    pop     rbx
    pop     r13
    pop     r12
    ret

; ==============================================================================
; Function: block_swap_sr
; Signature: void block_swap_sr(int32_t *restrict arr, size_t m, size_t r, size_t p)
; Params: RDI=arr, RSI=m, RDX=r, RCX=p
; ==============================================================================
block_swap_sr:
    ; i = m  (RSI)
    ; tmp = arr[i]
    mov     r8d, [rdi + rsi*4] ; r8d = tmp
    ; j = r  (RDX)
    
.sr_loop:
    cmp     rdx, rcx ; while (j < p)
    jge     .sr_done
    
    ; arr[i] = arr[j]
    mov     r9d, [rdi + rdx*4]
    mov     [rdi + rsi*4], r9d
    
    inc     rsi      ; i++
    
    ; arr[j] = arr[i]
    mov     r9d, [rdi + rsi*4]
    mov     [rdi + rdx*4], r9d
    
    inc     rdx      ; j++
    jmp     .sr_loop

.sr_done:
    ; arr[i] = arr[j] (wait, j is p now? C code: j increments loop, then accesses j)
    ; After loop, j == p.
    ; arr[i] = arr[j];
    mov     r9d, [rdi + rdx*4]
    mov     [rdi + rsi*4], r9d
    
    ; arr[j] = tmp;
    mov     [rdi + rdx*4], r8d
    ret

; ==============================================================================
; Function: block_swap
; Signature: void block_swap(int32_t *arr, size_t m, size_t r, size_t p)
; Params: RDI=arr, RSI=m, RDX=r, RCX=p
; ==============================================================================
block_swap:
    ; size_t ll = r - m;
    mov     r8, rdx
    sub     r8, rsi  ; r8 = ll
    jz      .bs_ret  ; if (ll == 0) return

    ; size_t lr = p - r + 1;
    mov     r9, rcx
    sub     r9, rdx
    inc     r9       ; r9 = lr

    cmp     r9, 1
    jne     .bs_check_size
    
    ; if (lr == 1) swap(&arr[m], &arr[p]);
    ; Need to calculate addresses
    lea     rax, [rdi + rsi*4]
    lea     rdx, [rdi + rcx*4]
    ; swap logic inline to avoid call overhead
    mov     r8d, [rax]
    mov     r9d, [rdx]
    mov     [rax], r9d
    mov     [rdx], r8d
    ret

.bs_check_size:
    ; if (lr <= ll) block_swap_sr(arr, m, r, p);
    cmp     r9, r8
    ja      .bs_call_sl
    
    ; block_swap_sr(arr, m, r, p)
    ; Args match current registers: RDI=arr, RSI=m, RDX=r, RCX=p
    call    block_swap_sr
    ret

.bs_call_sl:
    ; block_swap_sl(arr, m, p, ll)
    ; Current: RDI=arr, RSI=m, RDX=r, RCX=p
    ; Target:  RDI=arr, RSI=m, RDX=p, RCX=ll
    mov     rdx, rcx ; p
    mov     rcx, r8  ; ll
    call    block_swap_sl
.bs_ret:
    ret

; ==============================================================================
; Function: partition
; Signature: size_t partition(int32_t *arr, size_t l, size_t r, size_t p_idx)
; Params: RDI=arr, RSI=l, RDX=r, RCX=p_idx
; Returns: i (RAX)
; Optimized with AVX2
; ==============================================================================
partition:
    ; pivot_val = arr[p_idx]
    mov     r10d, [rdi + rcx*4] ; r10d = pivot_val

    ; i = l - 1
    mov     rax, rsi
    dec     rax      ; rax = i

    ; j = r
    mov     r8, rdx  ; r8 = j

    ; Prepare AVX2 Pivot
    vmovd   xmm0, r10d
    vpbroadcastd ymm0, xmm0

.part_loop:
    ; --- Inner Loop i ---
    ; while(true) { i++; if(i==j) return i; if(arr[i] >= pivot) break; }

.scan_i:
    inc     rax         ; i++
    cmp     rax, r8     ; if (i == j)
    je      .part_done  ; return i (in RAX)

    ; AVX2 Check: Scan forward from i
    ; Check if (j - i) >= 8
    mov     r9, r8
    sub     r9, rax
    cmp     r9, 8
    jl      .scalar_i_check

    ; AVX2 Block Check
    ; Load 8 elements from arr[i]
    vmovdqu ymm1, [rdi + rax*4]
    
    ; We want: arr[i] >= pivot
    ; Logic: VPCMPGTD dest, src1, src2  => dest = (src1 > src2) ? -1 : 0
    ; We have Pivot (YMM0). We load Val (YMM1).
    ; We break if Val >= Pivot.
    ; This is equivalent to NOT (Val < Pivot)
    ; Or NOT (Pivot > Val).
    ; VPCMPGTD YMM2, YMM0, YMM1  => YMM2 = (Pivot > Val)
    ; If Val < Pivot, bit is 1. We continue.
    ; If Val >= Pivot, bit is 0. We STOP.
    
    vpcmpgtd ymm2, ymm0, ymm1
    vpmovmskb r9d, ymm2
    
    ; r9d contains mask. 1111 per dword if (Pivot > Val).
    ; We want the first dword where this is FALSE (0000).
    ; Invert mask. We look for the first 1.
    not     r9d
    
    test    r9d, r9d
    jz      .advance_i_simd ; Mask is 0 (all 1s originally), so all Val < Pivot. Safe to skip.

    ; Found an element >= pivot.
    tzcnt   r9d, r9d    ; Find index of first 1 bit
    shr     r9d, 2      ; Convert bit index to int index (4 bits per int)
    add     rax, r9     ; i += offset
    jmp     .break_i

.advance_i_simd:
    add     rax, 7      ; Skip 8 elements (inc rax handled 1, add 7 = +8 total)
    jmp     .scan_i

.scalar_i_check:
    mov     r11d, [rdi + rax*4]
    cmp     r11d, r10d
    jge     .break_i
    jmp     .scan_i

.break_i:

    ; --- Inner Loop j ---
    ; while(true) { j--; if(j==i) return i; if(arr[j] <= pivot) break; }

.scan_j:
    dec     r8          ; j--
    cmp     r8, rax     ; if (j == i)
    je      .part_done  ; return i

    ; AVX2 Check: Scan backward from j
    ; Check if (j - i) >= 8. Note j is upper bound.
    mov     r9, r8
    sub     r9, rax
    cmp     r9, 8
    jl      .scalar_j_check

    ; Load 8 elements ENDING at j.
    ; Range: [j-7, ..., j]
    ; Load address: rdi + (r8 - 7)*4
    mov     r9, r8
    sub     r9, 7
    vmovdqu ymm1, [rdi + r9*4]

    ; We want: arr[j] <= pivot
    ; Logic: Break if Val <= Pivot.
    ; Continue if Val > Pivot.
    ; VPCMPGTD YMM2, YMM1, YMM0 => YMM2 = (Val > Pivot)
    ; Mask bits: 1 if Val > Pivot (continue).
    ;            0 if Val <= Pivot (STOP).
    
    vpcmpgtd ymm2, ymm1, ymm0
    vpmovmskb r11d, ymm2
    
    ; We want the LAST element (highest index) where Val <= Pivot.
    ; In vector [0..7], index 7 is 'j', index 0 is 'j-7'.
    ; The loop goes j, j-1...
    ; So we want the highest index in the vector where mask bit is 0.
    ; Invert mask. Look for set bits.
    not     r11d
    
    test    r11d, r11d
    jz      .advance_j_simd ; All Val > Pivot.
    
    ; Found one or more elements <= pivot.
    ; We want the one closest to 'j' (highest index).
    ; Use LZCNT (Leading Zero Count) on 32-bit reg?
    ; BSR (Bit Scan Reverse) or LZCNT.
    ; If we use lzcnt, we count zeros from MSB.
    ; The bits correspond to bytes. 0..31.
    ; Highest int index corresponds to bits 28-31.
    ; MSB is at the "left".
    ; We want the highest set bit.
    bsr     r11d, r11d  ; Find index of MSB set.
    shr     r11d, 2     ; Convert to int index (0..7).
    
    ; r11d is the offset inside the vector [j-7 ... j].
    ; Offset 7 means j. Offset 0 means j-7.
    ; We need to update r8 (j).
    ; r8 currently points to j.
    ; New j = (r8 - 7) + offset
    sub     r8, 7
    add     r8, r11
    jmp     .break_j

.advance_j_simd:
    sub     r8, 7       ; Done 8 elements.
    jmp     .scan_j

.scalar_j_check:
    mov     r11d, [rdi + r8*4]
    cmp     r11d, r10d
    jle     .break_j
    jmp     .scan_j

.break_j:
    ; swap(&arr[i], &arr[j])
    mov     r9d, [rdi + rax*4]  ; tmp_i
    mov     r11d, [rdi + r8*4]  ; tmp_j
    mov     [rdi + rax*4], r11d
    mov     [rdi + r8*4], r9d
    
    jmp     .part_loop

.part_done:
    ; i is in rax, which is return value
    ret

; ==============================================================================
; Function: downwave
; Signature: void downwave(int32_t *arr, size_t start, size_t sorted_start, size_t end)
; Params: RDI=arr, RSI=start, RDX=sorted_start, RCX=end
; ==============================================================================
downwave:
    ; Recursive function. Setup stack frame.
    push    rbp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8

    ; if (sorted_start == start) return;
    cmp     rdx, rsi
    je      .dw_return

    ; Store args
    mov     rbx, rdi ; arr
    mov     r12, rsi ; start
    mov     r13, rdx ; sorted_start
    mov     r14, rcx ; end

    ; size_t p = sorted_start + (end - sorted_start) / 2;
    mov     rax, r14
    sub     rax, r13
    shr     rax, 1
    add     rax, r13
    mov     r15, rax ; p

    ; size_t m = partition(arr, start, sorted_start, p);
    ; RDI=arr, RSI=start, RDX=sorted_start, RCX=p
    mov     rdi, rbx
    mov     rsi, r12
    mov     rdx, r13
    mov     rcx, r15
    call    partition
    ; m is in RAX.
    
    ; if (m == sorted_start)
    cmp     rax, r13
    jne     .dw_not_sorted_start

    ; Case: m == sorted_start
    cmp     r15, r13 ; if (p == sorted_start)
    jne     .dw_check_p_gt_0

    ; if (sorted_start > 0) upwave(arr, start, sorted_start - 1);
    test    r13, r13
    jz      .dw_return
    mov     rdi, rbx
    mov     rsi, r12
    mov     rdx, r13
    dec     rdx
    call    upwave
    jmp     .dw_return

.dw_check_p_gt_0:
    ; if (p > 0) downwave(arr, start, sorted_start, p - 1);
    test    r15, r15
    jz      .dw_return
    mov     rdi, rbx
    mov     rsi, r12
    mov     rdx, r13
    mov     rcx, r15
    dec     rcx
    call    downwave
    jmp     .dw_return

.dw_not_sorted_start:
    ; Save m in BP (safe reg? we pushed RBP)
    ; Actually stack or RBP. Let's use RBP as general purpose here since frame is custom
    mov     rbp, rax ; m

    ; block_swap(arr, m, sorted_start, p);
    mov     rdi, rbx
    mov     rsi, rbp
    mov     rdx, r13
    mov     rcx, r15
    call    block_swap

    ; if (m == start)
    cmp     rbp, r12
    jne     .dw_check_p_sorted

    ; Case: m == start
    cmp     r15, r13 ; if (p == sorted_start)
    jne     .dw_m_start_next

    ; upwave(arr, m + 1, end);
    mov     rdi, rbx
    mov     rsi, rbp
    inc     rsi
    mov     rdx, r14
    call    upwave
    jmp     .dw_return

.dw_m_start_next:
    ; size_t p_next = p + 1;
    ; downwave(arr, m + p_next - sorted_start, p_next, end);
    ; arg1: start = m + (p+1) - sorted_start
    lea     rax, [r15 + 1] ; p_next
    mov     rsi, rbp
    add     rsi, rax
    sub     rsi, r13
    
    mov     rdi, rbx
    mov     rdx, rax       ; sorted_start = p_next
    mov     rcx, r14       ; end
    call    downwave
    jmp     .dw_return

.dw_check_p_sorted:
    ; if (p == sorted_start)
    cmp     r15, r13
    jne     .dw_final_split

    ; if (m > 0) upwave(arr, start, m - 1);
    test    rbp, rbp
    jz      .dw_do_second_up
    mov     rdi, rbx
    mov     rsi, r12
    mov     rdx, rbp
    dec     rdx
    call    upwave

.dw_do_second_up:
    ; upwave(arr, m + 1, end);
    mov     rdi, rbx
    mov     rsi, rbp
    inc     rsi
    mov     rdx, r14
    call    upwave
    jmp     .dw_return

.dw_final_split:
    ; size_t right_part_len = p - sorted_start;
    mov     rax, r15
    sub     rax, r13 ; rax = right_part_len

    ; size_t split_point = m + right_part_len;
    mov     r8, rbp
    add     r8, rax ; r8 = split_point

    ; if (split_point > 0) downwave(arr, start, m, split_point - 1);
    test    r8, r8
    jz      .dw_second_rec
    
    ; We need to save r8 (split_point) across call?
    ; Yes. Push it.
    push    r8
    
    mov     rdi, rbx
    mov     rsi, r12
    mov     rdx, rbp
    mov     rcx, r8
    dec     rcx
    call    downwave
    
    pop     r8

.dw_second_rec:
    ; downwave(arr, split_point + 1, p + 1, end);
    mov     rdi, rbx
    mov     rsi, r8
    inc     rsi
    mov     rdx, r15
    inc     rdx
    mov     rcx, r14
    call    downwave

.dw_return:
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ==============================================================================
; Function: upwave
; Signature: void upwave(int32_t *arr, size_t start, size_t end)
; Params: RDI=arr, RSI=start, RDX=end
; ==============================================================================
upwave:
    push    rbp
    push    rbx
    push    r12
    push    r13
    push    r14
    
    ; if (start == end) return;
    cmp     rsi, rdx
    je      .uw_exit

    ; arr in RBX, start in R12, end in R13
    mov     rbx, rdi
    mov     r12, rsi
    mov     r13, rdx

    ; if (end == 0) return;
    test    r13, r13
    jz      .uw_exit

    ; size_t sorted_start = end; (R14)
    mov     r14, r13
    
    ; size_t sorted_len = 1; (RBP)
    mov     rbp, 1
    
    ; size_t left_bound = end - 1; (R15 - wait, need push R15)
    push    r15
    mov     r15, r13
    dec     r15

    ; size_t total_len = end - start + 1; (saved in Stack or calculated)
    ; we can calc on fly or store in register if available. 
    ; Let's recalculate when needed to save regs.
    
.uw_loop:
    ; downwave(arr, left_bound, sorted_start, end);
    mov     rdi, rbx
    mov     rsi, r15
    mov     rdx, r14
    mov     rcx, r13
    call    downwave

    ; sorted_start = left_bound;
    mov     r14, r15

    ; sorted_len = end - sorted_start + 1;
    mov     rbp, r13
    sub     rbp, r14
    inc     rbp

    ; Calc total_len = end - start + 1
    mov     rax, r13
    sub     rax, r12
    inc     rax

    ; if (total_len < (sorted_len << 2)) break;
    mov     rcx, rbp
    shl     rcx, 2
    cmp     rax, rcx
    jl      .uw_break

    ; size_t next_expansion = (sorted_len << 1) + 1;
    mov     rcx, rbp
    shl     rcx, 1
    inc     rcx

    ; if (end < next_expansion || (end - next_expansion) < start) left_bound = start;
    cmp     r13, rcx
    jb      .uw_set_start

    mov     rax, r13
    sub     rax, rcx
    cmp     rax, r12
    jb      .uw_set_start

    ; else left_bound = end - next_expansion;
    mov     r15, rax
    jmp     .uw_check_lb

.uw_set_start:
    mov     r15, r12

.uw_check_lb:
    ; if (left_bound < start) left_bound = start;
    cmp     r15, r12
    jae     .uw_check_ss
    mov     r15, r12

.uw_check_ss:
    ; if (sorted_start == start) break;
    cmp     r14, r12
    je      .uw_break
    
    jmp     .uw_loop

.uw_break:
    ; downwave(arr, start, sorted_start, end);
    mov     rdi, rbx
    mov     rsi, r12
    mov     rdx, r14
    mov     rcx, r13
    call    downwave

    pop     r15
.uw_exit:
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ==============================================================================
; Function: wave_sort
; Signature: void wave_sort(int32_t *arr, size_t n)
; Params: RDI=arr, RSI=n
; ==============================================================================
wave_sort:
    ; if (!arr || n < 2) return;
    test    rdi, rdi
    jz      .ws_done
    cmp     rsi, 2
    jb      .ws_done

    ; upwave(arr, 0, n - 1);
    dec     rsi      ; end = n - 1
    mov     rdx, rsi
    xor     rsi, rsi ; start = 0
    call    upwave

.ws_done:
    ret
