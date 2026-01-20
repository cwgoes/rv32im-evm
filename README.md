# rv32im-evm

A compiler that translates RISC-V rv32im machine code to Ethereum Virtual Machine (EVM) bytecode.

## Overview

This project compiles RISC-V rv32im (base integer + multiply/divide extension) binary code into EVM bytecode that can be executed on the Ethereum Virtual Machine. It uses [revm](https://github.com/bluealloy/revm) (reth's EVM implementation) as the execution environment.

## Features

- Full rv32im instruction set support (47 instructions)
- Direct binary-to-bytecode compilation (no intermediate representation)
- EVM Cancun spec support (uses PUSH0 for efficiency)
- Comprehensive test suite (87 tests covering the rv32im spec)
- **Optimized JUMPDEST emission** - only emits at actual jump targets (~2-3% gas savings)
- **Register DUP optimization** - uses DUP1 when same register is loaded twice consecutively (e.g., `mul rd, rs, rs`)

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
| Factorial | 10! | 3,628,800 | 22,537 | 1,344 |
| Fibonacci | fib(20) | 6,765 | 25,172 | 3,979 |
| GCD | gcd(10000, 7777) | 1 | 22,989 | 1,796 |
| Sum | 1+...+1000 | 500,500 | 144,286 | 123,093 |
| Power | 2^10 | 1,024 | 22,563 | 1,370 |
| Is Prime | 997 | prime | 28,495 | 7,302 |

*Exec Gas = Total Gas minus ~21k base overhead*

### Gas Cost Per RISC-V Instruction

| Loop Type | Gas/Iteration | Instructions/Iter | Gas/Instruction |
|-----------|---------------|-------------------|-----------------|
| Factorial (mul) | ~125 | 4 | ~31 |
| Fibonacci (add) | ~193 | 6 | ~32 |
| Sum (add) | ~123 | 4 | ~31 |

**Mean overhead: ~31 EVM gas per rv32im instruction**

This overhead factor includes:
- Register load/store operations (EVM memory access)
- 32-bit value masking (EVM uses 256-bit words)
- Control flow translation (RISC-V PC to EVM jump targets)

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
| EVM bytecode size | 31,768 bytes |
| Expansion ratio | 17.4x |

*Full SHA-256 implementation compiled with GCC (`-march=rv32im -O2`)*

### SHA-256-like Hash Mixing (Hand-assembled)

| Rounds | Total Gas | Gas/Round |
|--------|-----------|-----------|
| 1 | 54,686 | 54,686 |
| 4 | 60,584 | 15,146 |
| 16 | 84,176 | 5,261 |
| 64 | 178,544 | 2,790 |
| 256 | 556,016 | 2,172 |

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
