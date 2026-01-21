# Minimal startup code for rv32im
# Sets up stack and jumps to _start

.section .text._start
.global _start
.type _start, @function

_start:
    # Initialize stack pointer
    lui sp, %hi(__stack_top)
    addi sp, sp, %lo(__stack_top)

    # Clear BSS
    lui a0, %hi(__bss_start)
    addi a0, a0, %lo(__bss_start)
    lui a1, %hi(__bss_end)
    addi a1, a1, %lo(__bss_end)
1:
    bge a0, a1, 2f
    sw zero, 0(a0)
    addi a0, a0, 4
    j 1b
2:
    # Call main C entry point
    call main

    # Exit with return value in a0
    ecall

    # Loop forever if ecall returns
3:
    j 3b

.size _start, . - _start
