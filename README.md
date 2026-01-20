# rv32im-evm

A compiler that translates RISC-V rv32im machine code to Ethereum Virtual Machine (EVM) bytecode.

## Overview

This project compiles RISC-V rv32im (base integer + multiply/divide extension) binary code into EVM bytecode that can be executed on the Ethereum Virtual Machine. It uses [revm](https://github.com/bluealloy/revm) (reth's EVM implementation) as the execution environment.

## Features

- Full rv32im instruction set support (47 instructions)
- Direct binary-to-bytecode compilation (no intermediate representation)
- EVM Cancun spec support (uses PUSH0 for efficiency)
- Comprehensive test suite (87 tests covering the rv32im spec)
- **Optimized JUMPDEST emission** - only emits at actual jump targets
- **Register DUP optimization** - uses DUP1 when same register is loaded twice consecutively
- **Low-bits register storage** - eliminates SHL/SHR for register access (~30% gas savings)
- **Selective 32-bit masking** - only masks when overflow is possible
- **Optimized memory access** - single MLOAD + BYTE extraction for word/halfword loads
- **Branch vs zero optimization** - BNE/BEQ against zero skip unnecessary comparisons
- **Loop-aware register caching** - keeps hot registers on EVM stack during loops (~18% gas savings)

## Usage

```rust
use rv32im_evm::{Compiler, CompilerConfig, Runtime};

// RISC-V machine code (little-endian)
let program = vec![
    0x13, 0x05, 0xa0, 0x02, // addi a0, zero, 42
    0x73, 0x00, 0x00, 0x00, // ecall (return)
];

let config = CompilerConfig {
    load_address: 0x80000000,
    stack_pointer: 0x80010000,
    memory_size: 0x20000,
};

let mut compiler = Compiler::with_config(config);
let bytecode = compiler.compile(&program)?;

let runtime = Runtime::new(bytecode);
let output = runtime.execute()?;
println!("Result: {}", output.return_value); // 42
```

## Benchmark Results

The benchmark compiles various mathematical functions from RISC-V assembly to EVM and measures gas costs:

| Function | Input | Result | Total Gas | Exec Gas |
|----------|-------|--------|-----------|----------|
| Factorial | 10! | 3,628,800 | 22,006 | 842 |
| Fibonacci | fib(20) | 6,765 | 23,488 | 2,324 |
| GCD | gcd(10000, 7777) | 1 | 22,309 | 1,145 |
| Sum | 1+...+1000 | 500,500 | 92,264 | 71,100 |
| Power | 2^10 | 1,024 | 22,056 | 892 |
| Is Prime | 997 | prime | 26,389 | 5,225 |

*Exec Gas = Total Gas minus ~21k base overhead*

### Gas Cost Per RISC-V Instruction

| Loop Type | Gas/Iteration | Instructions/Iter | Gas/Instruction |
|-----------|---------------|-------------------|-----------------|
| Factorial (mul) | ~73 | 4 | ~18 |
| Fibonacci (add) | ~109 | 6 | ~18 |
| Sum (add) | ~71 | 4 | ~18 |

**Mean overhead: ~18 EVM gas per rv32im instruction**

This overhead factor includes:
- Register load/store operations (optimized with DUP for hot registers in loops)
- Control flow translation (RISC-V PC to EVM jump targets)
- 32-bit masking for arithmetic operations (RISC-V semantics require wraparound)

### Code Size Expansion

A typical 32-byte rv32im program (8 instructions) compiles to ~300 bytes of EVM bytecode, giving an expansion ratio of approximately **9-10x**.

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```

## Running the Benchmarks

```bash
# Factorial and other mathematical functions
cargo run --bin factorial-benchmark --release

# SHA-256-like hash function benchmark
cargo run --bin sha256-benchmark --release
```

## SHA-256 Benchmark Results

The SHA-256 benchmark demonstrates compiling cryptographic code from C to rv32im to EVM:

### Real SHA-256 (C -> rv32im -> EVM)

| Metric | Value |
|--------|-------|
| RISC-V code size | 1,824 bytes |
| EVM bytecode size | 19,953 bytes |
| Expansion ratio | 10.9x |

*Full SHA-256 implementation compiled with GCC (`-march=rv32im -O2`)*

### SHA-256-like Hash Mixing (Hand-assembled)

| Rounds | Total Gas | Gas/Round |
|--------|-----------|-----------|
| 1 | 50,373 | 50,373 |
| 4 | 53,745 | 13,436 |
| 16 | 67,233 | 4,202 |
| 64 | 121,185 | 1,894 |
| 256 | 336,993 | 1,316 |

The hand-assembled benchmark implements SHA-256-like operations:
- ROTR (rotate right) using SRL + SLL + OR
- XOR, AND for bit mixing (Sigma, Ch, Maj functions)
- ADD for combining values

Each round performs ~50 RISC-V instructions, similar to a real SHA-256 round.

### Building the SHA-256 C Code

To build the real SHA-256 from C (requires RISC-V toolchain):

```bash
# Install toolchain (Ubuntu/Debian)
sudo apt install gcc-riscv64-unknown-elf

# Build
cd sha256-bench && make
```

## License

MIT
