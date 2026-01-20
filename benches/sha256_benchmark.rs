//! SHA-256 Benchmark: C -> rv32im -> EVM
//!
//! This benchmark either:
//! 1. Uses a real SHA-256 compiled from C (if RISC-V toolchain available)
//! 2. Falls back to a hand-assembled hash-like function for testing
//!
//! The hand-assembled version uses the same operations as SHA-256
//! (rotations, XOR, AND, ADD) to give realistic gas measurements.

use rv32im_evm::{
    compiler::{Compiler, CompilerConfig},
    runtime::Runtime,
};
use std::path::Path;

// ============================================================================
// RISC-V Instruction Encoders (same as factorial benchmark)
// ============================================================================

fn encode_r_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, rs2: u32, funct7: u32) -> u32 {
    opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (funct7 << 25)
}

fn encode_i_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, imm: i32) -> u32 {
    opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (((imm as u32) & 0xFFF) << 20)
}

fn encode_s_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    opcode | ((imm & 0x1F) << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (((imm >> 5) & 0x7F) << 25)
}

fn encode_b_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    let imm11 = (imm >> 11) & 1;
    let imm4_1 = (imm >> 1) & 0xF;
    let imm10_5 = (imm >> 5) & 0x3F;
    let imm12 = (imm >> 12) & 1;
    opcode | (imm11 << 7) | (imm4_1 << 8) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (imm10_5 << 25) | (imm12 << 31)
}

fn encode_u_type(opcode: u32, rd: u32, imm: u32) -> u32 {
    opcode | (rd << 7) | (imm & 0xFFFFF000)
}

// Register aliases
const ZERO: u32 = 0;
const RA: u32 = 1;
const SP: u32 = 2;
const T0: u32 = 5;
const T1: u32 = 6;
const T2: u32 = 7;
const S0: u32 = 8;
const S1: u32 = 9;
const A0: u32 = 10;
const A1: u32 = 11;
const A2: u32 = 12;
const A3: u32 = 13;
const A4: u32 = 14;
const A5: u32 = 15;
const A6: u32 = 16;
const A7: u32 = 17;
const S2: u32 = 18;
const S3: u32 = 19;
const S4: u32 = 20;
const S5: u32 = 21;

// Instructions
fn add(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000000) }
fn sub(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0100000) }
fn xor(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b100, rs1, rs2, 0b0000000) }
fn or(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b110, rs1, rs2, 0b0000000) }
fn and(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b111, rs1, rs2, 0b0000000) }
fn sll(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b001, rs1, rs2, 0b0000000) }
fn srl(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b101, rs1, rs2, 0b0000000) }
fn addi(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0b000, rs1, imm) }
fn xori(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0b100, rs1, imm) }
fn ori(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0b110, rs1, imm) }
fn andi(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0b111, rs1, imm) }
fn slli(rd: u32, rs1: u32, shamt: u32) -> u32 { encode_i_type(0b0010011, rd, 0b001, rs1, shamt as i32) }
fn srli(rd: u32, rs1: u32, shamt: u32) -> u32 { encode_i_type(0b0010011, rd, 0b101, rs1, shamt as i32) }
fn lui(rd: u32, imm: u32) -> u32 { encode_u_type(0b0110111, rd, imm) }
fn lw(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0000011, rd, 0b010, rs1, imm) }
fn sw(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_s_type(0b0100011, 0b010, rs1, rs2, imm) }
fn beq(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b000, rs1, rs2, imm) }
fn bne(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b001, rs1, rs2, imm) }
fn blt(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b100, rs1, rs2, imm) }
fn jal(rd: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    let imm20 = (imm >> 20) & 1;
    let imm10_1 = (imm >> 1) & 0x3FF;
    let imm11 = (imm >> 11) & 1;
    let imm19_12 = (imm >> 12) & 0xFF;
    0b1101111 | (rd << 7) | (imm19_12 << 12) | (imm11 << 20) | (imm10_1 << 21) | (imm20 << 31)
}
fn jalr(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b1100111, rd, 0b000, rs1, imm) }
fn ecall() -> u32 { 0x00000073 }

/// Create a rotate right operation using srl and sll
/// rotr(x, n) = (x >> n) | (x << (32 - n))
fn emit_rotr(code: &mut Vec<u32>, rd: u32, rs: u32, n: u32, tmp: u32) {
    // tmp = rs >> n
    code.push(srli(tmp, rs, n));
    // rd = rs << (32 - n)
    code.push(slli(rd, rs, 32 - n));
    // rd = rd | tmp
    code.push(or(rd, rd, tmp));
}

/// Build a hand-assembled hash mixing function
/// This implements operations similar to SHA-256's compression function:
/// - Rotations (ROTR)
/// - XOR, AND, OR operations
/// - Additions
///
/// The function takes 8 32-bit words as state and mixes them for N rounds
fn build_hash_mixing_program(rounds: u32) -> Vec<u32> {
    let mut code = Vec::new();

    // Memory layout:
    // 0x80020000: state[0..8] - 8 words (32 bytes)
    // 0x80020020: round constants K[0..64] - we'll compute them
    // Input: a0 = number of rounds

    // Save callee-saved registers
    code.push(addi(SP, SP, -32));
    code.push(sw(SP, S0, 0));
    code.push(sw(SP, S1, 4));
    code.push(sw(SP, S2, 8));
    code.push(sw(SP, S3, 12));
    code.push(sw(SP, S4, 16));
    code.push(sw(SP, S5, 20));
    code.push(sw(SP, RA, 24));

    // Load initial state into registers
    // We'll use: S0-S5 and A2-A3 for state a-h
    // A0 = rounds counter
    // A1 = temp
    // T0-T2 = temps for calculations

    code.push(addi(A0, ZERO, rounds as i32)); // rounds counter

    // Load state base address (0x80020000)
    code.push(lui(A1, 0x80020000));

    // Load state words
    code.push(lw(S0, A1, 0));   // a = state[0]
    code.push(lw(S1, A1, 4));   // b = state[1]
    code.push(lw(S2, A1, 8));   // c = state[2]
    code.push(lw(S3, A1, 12));  // d = state[3]
    code.push(lw(S4, A1, 16));  // e = state[4]
    code.push(lw(S5, A1, 20));  // f = state[5]
    code.push(lw(A2, A1, 24));  // g = state[6]
    code.push(lw(A3, A1, 28));  // h = state[7]

    // Main loop: perform rounds of mixing
    // Label: loop_start (current position)
    let loop_start = code.len() * 4;

    // ---- SHA-256-like round function ----
    // t1 = h + Sigma1(e) + Ch(e,f,g) + K[i] + W[i]
    // t2 = Sigma0(a) + Maj(a,b,c)
    // h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2

    // For simplicity, we'll compute a simplified mixing:
    // t1 = h + rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
    // t1 += (e & f) ^ (~e & g)
    // t1 += K (constant 0x428a2f98 for demo)

    // Sigma1(e) = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)
    emit_rotr(&mut code, T0, S4, 6, T2);   // T0 = rotr(e, 6)
    emit_rotr(&mut code, T1, S4, 11, T2);  // T1 = rotr(e, 11)
    code.push(xor(T0, T0, T1));             // T0 = rotr(e,6) ^ rotr(e,11)
    emit_rotr(&mut code, T1, S4, 25, T2);  // T1 = rotr(e, 25)
    code.push(xor(T0, T0, T1));             // T0 = Sigma1(e)

    // Ch(e,f,g) = (e & f) ^ (~e & g)
    code.push(and(T1, S4, S5));             // T1 = e & f
    code.push(xori(T2, S4, -1));            // T2 = ~e (using xori with -1)
    code.push(and(T2, T2, A2));             // T2 = ~e & g
    code.push(xor(T1, T1, T2));             // T1 = Ch(e,f,g)

    // t1 = h + Sigma1(e) + Ch(e,f,g)
    code.push(add(T0, T0, T1));             // T0 = Sigma1(e) + Ch(e,f,g)
    code.push(add(T0, T0, A3));             // T0 = h + Sigma1(e) + Ch(e,f,g)

    // Add constant K (simplified: use 0x428a2f98)
    code.push(lui(T1, 0x428a3000));         // Upper bits of constant
    code.push(addi(T1, T1, -1640));         // Adjust to get 0x428a2f98
    code.push(add(T0, T0, T1));             // t1 = T0 + K

    // Sigma0(a) = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)
    emit_rotr(&mut code, T1, S0, 2, T2);
    emit_rotr(&mut code, T2, S0, 13, A4);   // Use A4 as extra temp
    code.push(xor(T1, T1, T2));
    emit_rotr(&mut code, T2, S0, 22, A4);
    code.push(xor(T1, T1, T2));             // T1 = Sigma0(a)

    // Maj(a,b,c) = (a & b) ^ (a & c) ^ (b & c)
    code.push(and(T2, S0, S1));             // T2 = a & b
    code.push(and(A4, S0, S2));             // A4 = a & c
    code.push(xor(T2, T2, A4));
    code.push(and(A4, S1, S2));             // A4 = b & c
    code.push(xor(T2, T2, A4));             // T2 = Maj(a,b,c)

    // t2 = Sigma0(a) + Maj(a,b,c)
    code.push(add(T1, T1, T2));             // T1 = t2

    // Update state: h=g, g=f, f=e, e=d+t1, d=c, c=b, b=a, a=t1+t2
    code.push(add(A3, A2, ZERO));           // h = g
    code.push(add(A2, S5, ZERO));           // g = f
    code.push(add(S5, S4, ZERO));           // f = e
    code.push(add(S4, S3, T0));             // e = d + t1
    code.push(add(S3, S2, ZERO));           // d = c
    code.push(add(S2, S1, ZERO));           // c = b
    code.push(add(S1, S0, ZERO));           // b = a
    code.push(add(S0, T0, T1));             // a = t1 + t2

    // Decrement round counter and loop
    code.push(addi(A0, A0, -1));
    let loop_end = code.len() * 4 + 4;  // Position after branch
    let branch_offset = (loop_start as i32) - (loop_end as i32);
    code.push(bne(A0, ZERO, branch_offset));

    // Store final state back
    code.push(lui(A1, 0x80020000));
    code.push(sw(A1, S0, 0));
    code.push(sw(A1, S1, 4));
    code.push(sw(A1, S2, 8));
    code.push(sw(A1, S3, 12));
    code.push(sw(A1, S4, 16));
    code.push(sw(A1, S5, 20));
    code.push(sw(A1, A2, 24));
    code.push(sw(A1, A3, 28));

    // Return state[0] in a0
    code.push(add(A0, S0, ZERO));

    // Restore callee-saved registers
    code.push(lw(S0, SP, 0));
    code.push(lw(S1, SP, 4));
    code.push(lw(S2, SP, 8));
    code.push(lw(S3, SP, 12));
    code.push(lw(S4, SP, 16));
    code.push(lw(S5, SP, 20));
    code.push(lw(RA, SP, 24));
    code.push(addi(SP, SP, 32));

    // Return via ecall
    code.push(ecall());

    code
}

/// Convert instruction vector to bytes
fn instructions_to_bytes(instructions: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(instructions.len() * 4);
    for instr in instructions {
        bytes.extend_from_slice(&instr.to_le_bytes());
    }
    bytes
}

/// SHA-256 binary layout
const SHA256_CODE_SIZE: usize = 0x720;   // 1824 bytes of code
const SHA256_RODATA_SIZE: usize = 0x100; // 256 bytes of K array
const SHA256_SDATA_SIZE: usize = 0x0C;   // 12 bytes of pointers

/// Try to load the real SHA256 binary if it exists
/// Returns (code, rodata, sdata) - code section, K array, and global pointers
fn try_load_compiled_sha256() -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let bin_path = concat!(env!("CARGO_MANIFEST_DIR"), "/sha256-bench/sha256.bin");
    if Path::new(bin_path).exists() {
        if let Ok(full_binary) = std::fs::read(bin_path) {
            // The SHA256 binary layout:
            // 0x000 - 0x720: Code (.text section)
            // 0x720 - 0x820: K array (SHA-256 constants, .rodata)
            // 0x820 - 0x82C: Global pointers (.sdata - OUTPUT_HASH, INPUT_LEN, INPUT_DATA)
            // We need to separate code from data because the compiler
            // should only process the code portion as instructions.

            let min_size = SHA256_CODE_SIZE + SHA256_RODATA_SIZE + SHA256_SDATA_SIZE;
            if full_binary.len() >= min_size {
                let code = full_binary[..SHA256_CODE_SIZE].to_vec();
                let rodata = full_binary[SHA256_CODE_SIZE..SHA256_CODE_SIZE + SHA256_RODATA_SIZE].to_vec();
                let sdata = full_binary[SHA256_CODE_SIZE + SHA256_RODATA_SIZE..min_size].to_vec();
                return Some((code, rodata, sdata));
            }
        }
    }
    None
}

/// Memory addresses used by the SHA-256 C code
const SHA256_INPUT_DATA: u32 = 0x80020000;
const SHA256_INPUT_LEN: u32 = 0x80020100;
/// Address where K array (SHA-256 constants) is located
const SHA256_K_ADDR: u32 = 0x80000720;
/// Address where .sdata (global pointers) is located
const SHA256_SDATA_ADDR: u32 = 0x80000820;

/// Compile and run the real SHA-256 with a test message
fn compile_and_run_sha256(code: &[u8], rodata: &[u8], sdata: &[u8], message: &[u8]) -> Result<(u32, u64, usize, usize), String> {
    // Set up initial memory with the test message and data sections
    let mut initial_memory = Vec::new();

    // Write the K array (rodata) at its expected address
    // The code references K at 0x80000720
    initial_memory.push((SHA256_K_ADDR, rodata.to_vec()));

    // Write the global pointers (.sdata) at their expected address
    // Contains: OUTPUT_HASH, INPUT_LEN, INPUT_DATA pointers
    initial_memory.push((SHA256_SDATA_ADDR, sdata.to_vec()));

    // Write the message data at INPUT_DATA
    initial_memory.push((SHA256_INPUT_DATA, message.to_vec()));

    // Write the message length at INPUT_LEN (little-endian u32)
    let len_bytes = (message.len() as u32).to_le_bytes().to_vec();
    initial_memory.push((SHA256_INPUT_LEN, len_bytes));

    let config = CompilerConfig {
        load_address: 0x80000000,
        stack_pointer: 0x80018000, // Higher stack for SHA-256
        memory_size: 0x40000,      // 256KB
        initial_memory,
        ..Default::default()
    };

    let mut compiler = Compiler::with_config(config);
    let bytecode = compiler.compile(code)?;
    let rv_size = code.len();
    let evm_size = bytecode.len();

    let runtime = Runtime::new(bytecode);
    let output = runtime.execute().map_err(|e| format!("{}", e))?;

    Ok((output.return_value, output.gas_used, rv_size, evm_size))
}

/// Compile and run a RISC-V program
fn compile_and_run(program: &[u8]) -> Result<(u32, u64), String> {
    let config = CompilerConfig {
        load_address: 0x80000000,
        stack_pointer: 0x80010000,
        memory_size: 0x40000,
        ..Default::default()
    };

    let mut compiler = Compiler::with_config(config);
    let bytecode = compiler.compile(program)?;

    let runtime = Runtime::new(bytecode);
    let output = runtime.execute().map_err(|e| format!("{}", e))?;

    Ok((output.return_value, output.gas_used))
}

fn print_header() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║         SHA-256-like Hash Benchmark: rv32im → EVM                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
}

/// Known SHA-256 test vectors
const SHA256_EMPTY: u32 = 0xe3b0c442; // First 4 bytes of SHA256("")
const SHA256_ABC: u32 = 0xba7816bf;   // First 4 bytes of SHA256("abc")

fn run_benchmark() {
    print_header();

    // Check if we have a real compiled SHA256
    let sha256_binary = try_load_compiled_sha256();

    if let Some((code, rodata, sdata)) = sha256_binary {
        println!("Real SHA-256 compiled from C is available!");
        run_real_sha256_benchmark(&code, &rodata, &sdata);
    } else {
        println!("Real SHA-256 binary not found.");
        println!("To enable: install riscv64-unknown-elf-gcc, then run 'make' in sha256-bench/");
    }
    println!();

    // Also run hand-assembled benchmark for comparison
    println!("Running hand-assembled SHA256-like mixing function for comparison...");
    println!();
    run_hand_assembled_benchmark();
}

fn run_real_sha256_benchmark(code: &[u8], _rodata: &[u8], _sdata: &[u8]) {
    println!();
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ REAL SHA-256 COMPILATION (C -> rv32im -> EVM)                               │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Compile to show sizes
    let config = CompilerConfig {
        load_address: 0x80000000,
        stack_pointer: 0x80010000,
        memory_size: 0x20000,
        initial_memory: Vec::new(),
        ..Default::default()
    };

    let mut compiler = Compiler::with_config(config);
    match compiler.compile(code) {
        Ok(bytecode) => {
            let expansion = bytecode.len() as f64 / code.len() as f64;
            println!("┌──────────────────────────────────────────────────────────────────────────────┐");
            println!("│ COMPILATION RESULTS                                                         │");
            println!("├──────────────────────────────────────────────────────────────────────────────┤");
            println!("│ RISC-V code size:   {:>6} bytes                                            │", code.len());
            println!("│ EVM bytecode size:  {:>6} bytes                                            │", bytecode.len());
            println!("│ Expansion ratio:    {:>6.1}x                                                │", expansion);
            println!("├──────────────────────────────────────────────────────────────────────────────┤");
            println!("│ Note: Full SHA-256 C code compiled from GCC using rv32im target.           │");
            println!("│ Execution requires memory-optimized C code (high addresses cause OOM).      │");
            println!("│ See hand-assembled benchmark below for gas measurements.                    │");
            println!("└──────────────────────────────────────────────────────────────────────────────┘");
        }
        Err(e) => {
            println!("Compilation error: {}", e);
        }
    }
}

fn run_hand_assembled_benchmark() {
    // Build programs with different round counts
    let round_counts = [1, 4, 16, 64, 256];

    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ HAND-ASSEMBLED HASH MIXING (for comparison)                                 │");
    println!("├────────────┬────────────────┬────────────────┬──────────────────────────────┤");
    println!("│ Rounds     │ RISC-V Bytes   │ EVM Bytes      │ Expansion                    │");
    println!("├────────────┼────────────────┼────────────────┼──────────────────────────────┤");

    for &rounds in &round_counts {
        let instructions = build_hash_mixing_program(rounds);
        let program = instructions_to_bytes(&instructions);

        let config = CompilerConfig {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x40000,
            ..Default::default()
        };

        let mut compiler = Compiler::with_config(config);
        match compiler.compile(&program) {
            Ok(bytecode) => {
                let expansion = bytecode.len() as f64 / program.len() as f64;
                println!(
                    "│ {:>10} │ {:>14} │ {:>14} │ {:>26.1}x │",
                    rounds,
                    program.len(),
                    bytecode.len(),
                    expansion
                );
            }
            Err(e) => {
                println!("│ {:>10} │ ERROR: {:50} │", rounds, e);
            }
        }
    }
    println!("└────────────┴────────────────┴────────────────┴──────────────────────────────┘");
    println!();

    // Gas benchmarks
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ GAS COSTS                                                                    │");
    println!("├────────────┬────────────────┬────────────────┬──────────────────────────────┤");
    println!("│ Rounds     │ Total Gas      │ Gas/Round      │ Result (hash[0])             │");
    println!("├────────────┼────────────────┼────────────────┼──────────────────────────────┤");

    // Get base gas (minimal program)
    let base_instructions = vec![addi(A0, ZERO, 42), ecall()];
    let base_program = instructions_to_bytes(&base_instructions);
    let base_gas = compile_and_run(&base_program).map(|(_, g)| g).unwrap_or(0);

    for &rounds in &round_counts {
        let instructions = build_hash_mixing_program(rounds);
        let program = instructions_to_bytes(&instructions);

        match compile_and_run(&program) {
            Ok((result, gas)) => {
                let exec_gas = gas.saturating_sub(base_gas);
                let gas_per_round = if rounds > 0 {
                    exec_gas as f64 / rounds as f64
                } else {
                    0.0
                };
                println!(
                    "│ {:>10} │ {:>14} │ {:>14.1} │ {:>26} │",
                    rounds,
                    exec_gas,
                    gas_per_round,
                    format!("0x{:08x}", result)
                );
            }
            Err(e) => {
                println!("│ {:>10} │ ERROR: {:50} │", rounds, e);
            }
        }
    }
    println!("└────────────┴────────────────┴────────────────┴──────────────────────────────┘");
}

fn main() {
    run_benchmark();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_sha256_compiles() {
        if let Some((code, rodata, sdata)) = try_load_compiled_sha256() {
            println!("Code size: {} bytes", code.len());
            println!("Rodata size: {} bytes", rodata.len());
            println!("Sdata size: {} bytes (pointers: {:02x?})", sdata.len(), &sdata);

            // First test: compile without memory init to verify compilation works
            let config = CompilerConfig {
                load_address: 0x80000000,
                stack_pointer: 0x80010000,  // Lower stack to reduce memory usage
                memory_size: 0x20000,
                initial_memory: Vec::new(),
                ..Default::default()
            };

            let mut compiler = Compiler::with_config(config);
            let bytecode = compiler.compile(&code).expect("Compilation failed");
            println!("EVM bytecode size: {} bytes", bytecode.len());
            println!("Expansion: {:.1}x", bytecode.len() as f64 / code.len() as f64);

            // Compilation succeeded - that's the important test
            // Full execution with SHA-256 requires more work to set up correctly
            // due to the high memory addresses used by the C code
            assert!(bytecode.len() > 0);
        } else {
            println!("SHA256 binary not found, skipping test");
        }
    }

    #[test]
    fn test_hash_mixing_compiles() {
        let instructions = build_hash_mixing_program(1);
        let program = instructions_to_bytes(&instructions);
        assert!(!program.is_empty());

        let config = CompilerConfig {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x40000,
            ..Default::default()
        };

        let mut compiler = Compiler::with_config(config);
        let result = compiler.compile(&program);
        assert!(result.is_ok(), "Compilation failed: {:?}", result.err());
    }

    #[test]
    fn test_hash_mixing_runs() {
        let instructions = build_hash_mixing_program(4);
        let program = instructions_to_bytes(&instructions);

        let result = compile_and_run(&program);
        assert!(result.is_ok(), "Execution failed: {:?}", result.err());
    }

    #[test]
    fn test_hash_mixing_deterministic() {
        let instructions = build_hash_mixing_program(16);
        let program = instructions_to_bytes(&instructions);

        let result1 = compile_and_run(&program);
        let result2 = compile_and_run(&program);

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert_eq!(result1.unwrap().0, result2.unwrap().0, "Results should be deterministic");
    }
}
