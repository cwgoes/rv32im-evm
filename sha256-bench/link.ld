/*
 * Linker script for bare-metal RISC-V rv32im
 * Memory layout matches our EVM compiler's configuration
 */

ENTRY(_start)

MEMORY
{
    /* Program memory - where the code is loaded */
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 64K
    /* Stack grows down from here */
    STACK (rw) : ORIGIN = 0x80010000, LENGTH = 64K
    /* Data area for input/output */
    DATA (rw) : ORIGIN = 0x80020000, LENGTH = 64K
}

SECTIONS
{
    /* Code section */
    .text : {
        *(.text.start)  /* Startup code first */
        *(.text*)
        *(.rodata*)     /* Read-only data with code */
    } > RAM

    /* Initialized data */
    .data : {
        __data_start = .;
        *(.data*)
        __data_end = .;
    } > RAM

    /* Uninitialized data (BSS) */
    .bss : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    /* Stack */
    .stack (NOLOAD) : {
        . = ALIGN(16);
        __stack_bottom = .;
        . = . + 0x8000;  /* 32KB stack */
        __stack_top = .;
    } > STACK

    /* Discard unwanted sections */
    /DISCARD/ : {
        *(.comment)
        *(.note*)
        *(.eh_frame*)
    }
}
