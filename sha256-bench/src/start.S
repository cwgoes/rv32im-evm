/*
 * Bare-metal RISC-V startup code for rv32im
 * Sets up stack pointer and calls main, then triggers ecall to return
 */

.section .text.start
.global _start
.type _start, @function

_start:
    /* Set up stack pointer (already done by our EVM compiler, but be safe) */
    lui sp, %hi(__stack_top)
    addi sp, sp, %lo(__stack_top)

    /* Clear BSS section */
    la t0, __bss_start
    la t1, __bss_end
1:
    bge t0, t1, 2f
    sw zero, 0(t0)
    addi t0, t0, 4
    j 1b
2:

    /* Call main */
    call main

    /* Return value is in a0, trigger ecall to return to EVM */
    ecall

    /* Should never reach here */
    j .

.size _start, . - _start
