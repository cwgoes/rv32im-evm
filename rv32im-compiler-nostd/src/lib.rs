//! RISC-V rv32im to EVM Compiler (no_std version)
//!
//! This is a minimal no_std version of the compiler that can be
//! compiled to rv32im itself for meta-compilation experiments.

#![no_std]

extern crate alloc;

pub mod decoder;
pub mod evm;
pub mod compiler;

pub use decoder::{Instruction, decode_instruction};
pub use evm::{Opcode, EvmBytecode};
pub use compiler::{Compiler, CompilerConfig};
