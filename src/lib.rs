//! RISC-V rv32im to Ethereum EVM Compiler
//!
//! This crate provides a compiler that translates RISC-V rv32im programs
//! into EVM bytecode that can be executed on the Ethereum Virtual Machine.

pub mod decoder;
pub mod evm;
pub mod compiler;
pub mod runtime;

pub use compiler::Compiler;
pub use decoder::{Instruction, decode_instruction};
pub use evm::EvmBytecode;
