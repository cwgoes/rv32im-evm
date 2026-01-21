/* Linker script for rv32im bare-metal */

ENTRY(_start)

MEMORY
{
    /* Code starts at the standard RISC-V load address */
    ROM (rx)  : ORIGIN = 0x80000000, LENGTH = 64K
    /* Data/BSS after code */
    RAM (rwx) : ORIGIN = 0x80010000, LENGTH = 64K
}

SECTIONS
{
    .text : {
        *(.text._start)
        *(.text*)
    } > ROM

    .rodata : {
        *(.rodata*)
    } > ROM

    .data : {
        *(.data*)
    } > RAM AT > ROM

    .bss : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    /* Stack at top of RAM */
    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
