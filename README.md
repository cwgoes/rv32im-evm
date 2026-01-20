# rv32im-evm

A compiler that translates RISC-V rv32im machine code to Ethereum Virtual Machine (EVM) bytecode.

## Overview

This project compiles RISC-V rv32im (base integer + multiply/divide extension) binary code into EVM bytecode that can be executed on the Ethereum Virtual Machine. It uses [revm](https://github.com/bluealloy/revm) (reth's EVM implementation) as the execution environment.

## Features

- Full rv32im instruction set support (47 instructions)
- Direct binary-to-bytecode compilation (no intermediate representation)
- EVM Cancun spec support (uses PUSH0 for efficiency)
- Comprehensive test suite (87 tests covering the rv32im spec)

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
| Factorial | 10! | 3,628,800 | 22,569 | 1,375 |
| Fibonacci | fib(20) | 6,765 | 25,275 | 4,081 |
| GCD | gcd(10000, 7777) | 1 | 23,026 | 1,832 |
| Sum | 1+...+1000 | 500,500 | 147,288 | 126,094 |
| Power | 2^10 | 1,024 | 22,596 | 1,402 |
| Is Prime | 997 | prime | 28,928 | 7,734 |

*Exec Gas = Total Gas minus ~21k base overhead*

### Gas Cost Per RISC-V Instruction

| Loop Type | Gas/Iteration | Instructions/Iter | Gas/Instruction |
|-----------|---------------|-------------------|-----------------|
| Factorial (mul) | ~128 | 4 | ~32 |
| Fibonacci (add) | ~198 | 6 | ~33 |
| Sum (add) | ~126 | 4 | ~31 |

**Mean overhead: ~32 EVM gas per rv32im instruction**

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

## Running the Benchmark

```bash
cargo run --bin factorial-benchmark --release
```

## License

MIT
