//! RISC-V rv32im to EVM Compiler - Bare metal binary
//!
//! This binary is compiled to rv32im and can then be compiled to EVM
//! for meta-compilation experiments.

#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use core::panic::PanicInfo;
use core::alloc::{GlobalAlloc, Layout};

use rv32im_compiler_nostd::{Compiler, CompilerConfig};

// Memory layout for the embedded compiler
// Input program at 0x80001000
// Output buffer at 0x80002000
// Heap at 0x80010000

const INPUT_ADDR: usize = 0x80001000;
const INPUT_LEN_ADDR: usize = 0x80000F00;
const OUTPUT_ADDR: usize = 0x80002000;
const OUTPUT_LEN_ADDR: usize = 0x80001F00;
const HEAP_START: usize = 0x80010000;
const HEAP_SIZE: usize = 0x10000; // 64KB heap

/// Simple bump allocator for bare metal (single-threaded, no atomics)
struct BumpAllocator {
    head: core::cell::UnsafeCell<usize>,
}

impl BumpAllocator {
    const fn new() -> Self {
        Self {
            head: core::cell::UnsafeCell::new(HEAP_START),
        }
    }
}

// SAFETY: Single-threaded bare metal environment
unsafe impl Sync for BumpAllocator {}

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();

        let head = *self.head.get();
        let aligned = (head + align - 1) & !(align - 1);
        let new_head = aligned + size;

        if new_head > HEAP_START + HEAP_SIZE {
            return core::ptr::null_mut();
        }

        *self.head.get() = new_head;
        aligned as *mut u8
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocator doesn't support deallocation
    }
}

#[global_allocator]
static ALLOCATOR: BumpAllocator = BumpAllocator::new();

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

/// Read a 32-bit value from memory
#[inline(always)]
unsafe fn read_u32(addr: usize) -> u32 {
    core::ptr::read_volatile(addr as *const u32)
}

/// Write a 32-bit value to memory
#[inline(always)]
unsafe fn write_u32(addr: usize, value: u32) {
    core::ptr::write_volatile(addr as *mut u32, value);
}

/// Read a byte from memory
#[inline(always)]
unsafe fn read_u8(addr: usize) -> u8 {
    core::ptr::read_volatile(addr as *const u8)
}

/// Write a byte to memory
#[inline(always)]
unsafe fn write_u8(addr: usize, value: u8) {
    core::ptr::write_volatile(addr as *mut u8, value);
}

/// Entry point - reads input program, compiles it, writes output
#[no_mangle]
pub extern "C" fn _start() -> ! {
    unsafe {
        // Read input length
        let input_len = read_u32(INPUT_LEN_ADDR) as usize;

        // Read input program
        let mut input = Vec::with_capacity(input_len);
        for i in 0..input_len {
            input.push(read_u8(INPUT_ADDR + i));
        }

        // Configure compiler
        let config = CompilerConfig {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x20000,
        };

        // Compile
        let mut compiler = Compiler::with_config(config);
        let output = match compiler.compile(&input) {
            Ok(bytecode) => bytecode,
            Err(_) => {
                write_u32(OUTPUT_LEN_ADDR, 0);
                loop {}
            }
        };

        // Write output
        let output_len = output.len();
        for (i, &byte) in output.iter().enumerate() {
            write_u8(OUTPUT_ADDR + i, byte);
        }
        write_u32(OUTPUT_LEN_ADDR, output_len as u32);

        // Return output length in a0 (via ecall simulation)
        // Store result and halt
        core::arch::asm!(
            "mv a0, {0}",
            "ecall",
            in(reg) output_len,
            options(noreturn)
        );
    }
}
