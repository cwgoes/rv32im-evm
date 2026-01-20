//! Comprehensive tests for the RISC-V rv32im to EVM compiler
//!
//! These tests cover all instructions in the RV32I base integer instruction set
//! and the M extension for multiplication/division.

use rv32im_evm::{
    compiler::{Compiler, CompilerConfig},
    runtime::Runtime,
    decoder::{decode_instruction, Instruction},
};

/// Helper to create a compiler with default test configuration
fn test_compiler() -> Compiler {
    let config = CompilerConfig {
        load_address: 0x80000000,
        stack_pointer: 0x80010000,
        memory_size: 0x20000,
        ..Default::default()
    };
    Compiler::with_config(config)
}

/// Helper to compile and execute a program, returning the a0 register value
fn run_program(instructions: &[u32]) -> u32 {
    run_program_debug(instructions, false)
}

/// Helper to compile and execute with optional debug output
fn run_program_debug(instructions: &[u32], debug: bool) -> u32 {
    let mut program = Vec::new();
    for instr in instructions {
        program.extend_from_slice(&instr.to_le_bytes());
    }

    let mut compiler = test_compiler();
    let bytecode = compiler.compile(&program).expect("Compilation failed");

    if debug {
        eprintln!("Bytecode length: {} bytes", bytecode.len());
        eprintln!("Bytecode: {}", hex::encode(&bytecode));
    }

    let runtime = Runtime::new(bytecode);
    let output = runtime.execute().expect("Execution failed");
    output.return_value
}

// ============================================================
// Instruction encoding helpers
// ============================================================

/// Encode a U-type instruction (LUI, AUIPC)
fn encode_u_type(opcode: u32, rd: u32, imm: i32) -> u32 {
    opcode | (rd << 7) | ((imm as u32) & 0xFFFFF000)
}

/// Encode an I-type instruction
fn encode_i_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, imm: i32) -> u32 {
    opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (((imm as u32) & 0xFFF) << 20)
}

/// Encode an R-type instruction
fn encode_r_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, rs2: u32, funct7: u32) -> u32 {
    opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (funct7 << 25)
}

/// Encode a B-type instruction
fn encode_b_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    let imm12 = (imm >> 12) & 1;
    let imm11 = (imm >> 11) & 1;
    let imm10_5 = (imm >> 5) & 0x3F;
    let imm4_1 = (imm >> 1) & 0xF;

    opcode | (imm11 << 7) | (imm4_1 << 8) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (imm10_5 << 25) | (imm12 << 31)
}

/// Encode a J-type instruction (JAL)
fn encode_j_type(opcode: u32, rd: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    let imm20 = (imm >> 20) & 1;
    let imm10_1 = (imm >> 1) & 0x3FF;
    let imm11 = (imm >> 11) & 1;
    let imm19_12 = (imm >> 12) & 0xFF;

    opcode | (rd << 7) | (imm19_12 << 12) | (imm11 << 20) | (imm10_1 << 21) | (imm20 << 31)
}

/// Encode a S-type instruction
fn encode_s_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    let imm4_0 = imm & 0x1F;
    let imm11_5 = (imm >> 5) & 0x7F;

    opcode | (imm4_0 << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (imm11_5 << 25)
}

// Register aliases
const ZERO: u32 = 0;
const RA: u32 = 1;
const SP: u32 = 2;
#[allow(dead_code)]
const GP: u32 = 3;
#[allow(dead_code)]
const TP: u32 = 4;
const T0: u32 = 5;
const T1: u32 = 6;
const T2: u32 = 7;
#[allow(dead_code)]
const S0: u32 = 8;
#[allow(dead_code)]
const S1: u32 = 9;
const A0: u32 = 10;
#[allow(dead_code)]
const A1: u32 = 11;
#[allow(dead_code)]
const A2: u32 = 12;
#[allow(dead_code)]
const A3: u32 = 13;
#[allow(dead_code)]
const A4: u32 = 14;
#[allow(dead_code)]
const A5: u32 = 15;
#[allow(dead_code)]
const A6: u32 = 16;
#[allow(dead_code)]
const A7: u32 = 17;
const T3: u32 = 28;
#[allow(dead_code)]
const T4: u32 = 29;
#[allow(dead_code)]
const T5: u32 = 30;
#[allow(dead_code)]
const T6: u32 = 31;

// Instruction macros
fn lui(rd: u32, imm: i32) -> u32 { encode_u_type(0b0110111, rd, imm) }
fn auipc(rd: u32, imm: i32) -> u32 { encode_u_type(0b0010111, rd, imm) }

fn jal(rd: u32, imm: i32) -> u32 { encode_j_type(0b1101111, rd, imm) }
fn jalr(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b1100111, rd, 0, rs1, imm) }

fn beq(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b000, rs1, rs2, imm) }
fn bne(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b001, rs1, rs2, imm) }
fn blt(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b100, rs1, rs2, imm) }
fn bge(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b101, rs1, rs2, imm) }
fn bltu(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b110, rs1, rs2, imm) }
fn bgeu(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b111, rs1, rs2, imm) }

fn lb(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0000011, rd, 0b000, rs1, imm) }
fn lh(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0000011, rd, 0b001, rs1, imm) }
fn lw(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0000011, rd, 0b010, rs1, imm) }
fn lbu(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0000011, rd, 0b100, rs1, imm) }
fn lhu(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0000011, rd, 0b101, rs1, imm) }

fn sb(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_s_type(0b0100011, 0b000, rs1, rs2, imm) }
fn sh(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_s_type(0b0100011, 0b001, rs1, rs2, imm) }
fn sw(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_s_type(0b0100011, 0b010, rs1, rs2, imm) }

fn addi(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0b000, rs1, imm) }
fn slti(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0b010, rs1, imm) }
fn sltiu(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0b011, rs1, imm) }
fn xori(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0b100, rs1, imm) }
fn ori(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0b110, rs1, imm) }
fn andi(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0b111, rs1, imm) }
fn slli(rd: u32, rs1: u32, shamt: u32) -> u32 { encode_i_type(0b0010011, rd, 0b001, rs1, shamt as i32) }
fn srli(rd: u32, rs1: u32, shamt: u32) -> u32 { encode_i_type(0b0010011, rd, 0b101, rs1, shamt as i32) }
fn srai(rd: u32, rs1: u32, shamt: u32) -> u32 { encode_i_type(0b0010011, rd, 0b101, rs1, (shamt | 0x400) as i32) }

fn add(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000000) }
fn sub(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0100000) }
fn sll(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b001, rs1, rs2, 0b0000000) }
fn slt(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b010, rs1, rs2, 0b0000000) }
fn sltu(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b011, rs1, rs2, 0b0000000) }
fn xor(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b100, rs1, rs2, 0b0000000) }
fn srl(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b101, rs1, rs2, 0b0000000) }
fn sra(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b101, rs1, rs2, 0b0100000) }
fn or(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b110, rs1, rs2, 0b0000000) }
fn and(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b111, rs1, rs2, 0b0000000) }

// M extension
fn mul(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000001) }
fn mulh(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b001, rs1, rs2, 0b0000001) }
fn mulhsu(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b010, rs1, rs2, 0b0000001) }
fn mulhu(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b011, rs1, rs2, 0b0000001) }
fn div(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b100, rs1, rs2, 0b0000001) }
fn divu(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b101, rs1, rs2, 0b0000001) }
fn rem(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b110, rs1, rs2, 0b0000001) }
fn remu(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b111, rs1, rs2, 0b0000001) }

// System instructions
fn ecall() -> u32 { 0x00000073 }
fn ebreak() -> u32 { 0x00100073 }
fn fence(pred: u32, succ: u32) -> u32 { 0b0001111 | (succ << 20) | (pred << 24) }

// Pseudo-instructions
fn nop() -> u32 { addi(ZERO, ZERO, 0) }
fn li(rd: u32, imm: i32) -> u32 { addi(rd, ZERO, imm) }
fn mv(rd: u32, rs: u32) -> u32 { addi(rd, rs, 0) }

// ============================================================
// RV32I Base Instruction Set Tests
// ============================================================

#[test]
fn test_lui() {
    // lui a0, 0x12345
    let result = run_program(&[
        lui(A0, 0x12345000u32 as i32),
        ecall(),
    ]);
    assert_eq!(result, 0x12345000);
}

#[test]
fn test_lui_negative() {
    // lui a0, 0xFFFFF (negative upper bits)
    let result = run_program(&[
        lui(A0, 0xFFFFF000u32 as i32),
        ecall(),
    ]);
    assert_eq!(result, 0xFFFFF000);
}

#[test]
fn test_auipc() {
    // auipc a0, 0
    // The result should be the PC (0x80000000)
    let result = run_program(&[
        auipc(A0, 0),
        ecall(),
    ]);
    assert_eq!(result, 0x80000000);
}

#[test]
fn test_auipc_with_offset() {
    // auipc a0, 1 (adds 0x1000 to PC)
    let result = run_program(&[
        auipc(A0, 0x1000),
        ecall(),
    ]);
    assert_eq!(result, 0x80001000);
}

#[test]
fn test_addi() {
    // addi a0, zero, 42
    let result = run_program(&[
        addi(A0, ZERO, 42),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

#[test]
fn test_addi_negative() {
    // addi a0, zero, -1
    let result = run_program(&[
        addi(A0, ZERO, -1),
        ecall(),
    ]);
    assert_eq!(result, 0xFFFFFFFF);
}

#[test]
fn test_addi_chain() {
    // addi t0, zero, 10
    // addi a0, t0, 20
    let result = run_program(&[
        addi(T0, ZERO, 10),
        addi(A0, T0, 20),
        ecall(),
    ]);
    assert_eq!(result, 30);
}

#[test]
fn test_slti_less() {
    // t0 = 5, slti a0, t0, 10 -> 1 (5 < 10)
    let result = run_program(&[
        addi(T0, ZERO, 5),
        slti(A0, T0, 10),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_slti_not_less() {
    // t0 = 15, slti a0, t0, 10 -> 0 (15 >= 10)
    let result = run_program(&[
        addi(T0, ZERO, 15),
        slti(A0, T0, 10),
        ecall(),
    ]);
    assert_eq!(result, 0);
}

#[test]
fn test_slti_negative() {
    // t0 = -5, slti a0, t0, 0 -> 1 (-5 < 0)
    let result = run_program(&[
        addi(T0, ZERO, -5),
        slti(A0, T0, 0),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_sltiu() {
    // t0 = 5, sltiu a0, t0, 10 -> 1
    let result = run_program(&[
        addi(T0, ZERO, 5),
        sltiu(A0, T0, 10),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_sltiu_negative_as_large_unsigned() {
    // t0 = -1 (0xFFFFFFFF), sltiu a0, t0, 1 -> 0 (0xFFFFFFFF is large unsigned)
    let result = run_program(&[
        addi(T0, ZERO, -1),
        sltiu(A0, T0, 1),
        ecall(),
    ]);
    assert_eq!(result, 0);
}

#[test]
fn test_xori() {
    // t0 = 0xFF, xori a0, t0, 0x0F -> 0xF0
    let result = run_program(&[
        addi(T0, ZERO, 0xFF),
        xori(A0, T0, 0x0F),
        ecall(),
    ]);
    assert_eq!(result, 0xF0);
}

#[test]
fn test_ori() {
    // t0 = 0xF0, ori a0, t0, 0x0F -> 0xFF
    let result = run_program(&[
        addi(T0, ZERO, 0xF0),
        ori(A0, T0, 0x0F),
        ecall(),
    ]);
    assert_eq!(result, 0xFF);
}

#[test]
fn test_andi() {
    // t0 = 0xFF, andi a0, t0, 0x0F -> 0x0F
    let result = run_program(&[
        addi(T0, ZERO, 0xFF),
        andi(A0, T0, 0x0F),
        ecall(),
    ]);
    assert_eq!(result, 0x0F);
}

#[test]
fn test_slli() {
    // t0 = 1, slli a0, t0, 4 -> 16
    let result = run_program(&[
        addi(T0, ZERO, 1),
        slli(A0, T0, 4),
        ecall(),
    ]);
    assert_eq!(result, 16);
}

#[test]
fn test_srli() {
    // t0 = 256, srli a0, t0, 4 -> 16
    let result = run_program(&[
        addi(T0, ZERO, 256),
        srli(A0, T0, 4),
        ecall(),
    ]);
    assert_eq!(result, 16);
}

#[test]
fn test_srli_no_sign_extend() {
    // t0 = 0x80000000, srli a0, t0, 4 -> 0x08000000 (logical shift)
    let result = run_program(&[
        lui(T0, 0x80000000u32 as i32),
        srli(A0, T0, 4),
        ecall(),
    ]);
    assert_eq!(result, 0x08000000);
}

#[test]
fn test_srai() {
    // t0 = 256, srai a0, t0, 4 -> 16
    let result = run_program(&[
        addi(T0, ZERO, 256),
        srai(A0, T0, 4),
        ecall(),
    ]);
    assert_eq!(result, 16);
}

#[test]
fn test_srai_sign_extend() {
    // t0 = 0x80000000, srai a0, t0, 4 -> 0xF8000000 (arithmetic shift)
    let result = run_program(&[
        lui(T0, 0x80000000u32 as i32),
        srai(A0, T0, 4),
        ecall(),
    ]);
    assert_eq!(result, 0xF8000000);
}

#[test]
fn test_add() {
    // t0 = 10, t1 = 20, add a0, t0, t1 -> 30
    let result = run_program(&[
        addi(T0, ZERO, 10),
        addi(T1, ZERO, 20),
        add(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 30);
}

#[test]
fn test_add_overflow() {
    // t0 = 0xFFFFFFFF, t1 = 1, add a0, t0, t1 -> 0 (overflow)
    let result = run_program(&[
        addi(T0, ZERO, -1),
        addi(T1, ZERO, 1),
        add(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 0);
}

#[test]
fn test_sub() {
    // t0 = 30, t1 = 20, sub a0, t0, t1 -> 10
    let result = run_program(&[
        addi(T0, ZERO, 30),
        addi(T1, ZERO, 20),
        sub(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 10);
}

#[test]
fn test_sub_underflow() {
    // t0 = 0, t1 = 1, sub a0, t0, t1 -> 0xFFFFFFFF
    let result = run_program(&[
        addi(T0, ZERO, 0),
        addi(T1, ZERO, 1),
        sub(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 0xFFFFFFFF);
}

#[test]
fn test_sll() {
    // t0 = 1, t1 = 4, sll a0, t0, t1 -> 16
    let result = run_program(&[
        addi(T0, ZERO, 1),
        addi(T1, ZERO, 4),
        sll(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 16);
}

#[test]
fn test_sll_only_low_5_bits() {
    // t0 = 1, t1 = 36 (0b100100), sll a0, t0, t1 -> 16 (uses only low 5 bits: 4)
    let result = run_program(&[
        addi(T0, ZERO, 1),
        addi(T1, ZERO, 36),
        sll(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 16);
}

#[test]
fn test_slt_less() {
    // t0 = -1, t1 = 1, slt a0, t0, t1 -> 1 (signed)
    let result = run_program(&[
        addi(T0, ZERO, -1),
        addi(T1, ZERO, 1),
        slt(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_slt_not_less() {
    // t0 = 1, t1 = -1, slt a0, t0, t1 -> 0 (signed)
    let result = run_program(&[
        addi(T0, ZERO, 1),
        addi(T1, ZERO, -1),
        slt(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 0);
}

#[test]
fn test_sltu_less() {
    // t0 = 1, t1 = 0xFFFFFFFF, sltu a0, t0, t1 -> 1 (unsigned: 1 < 0xFFFFFFFF)
    let result = run_program(&[
        addi(T0, ZERO, 1),
        addi(T1, ZERO, -1),
        sltu(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_sltu_not_less() {
    // t0 = 0xFFFFFFFF, t1 = 1, sltu a0, t0, t1 -> 0 (unsigned)
    let result = run_program(&[
        addi(T0, ZERO, -1),
        addi(T1, ZERO, 1),
        sltu(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 0);
}

#[test]
fn test_xor() {
    // t0 = 0xFF, t1 = 0x0F, xor a0, t0, t1 -> 0xF0
    let result = run_program(&[
        addi(T0, ZERO, 0xFF),
        addi(T1, ZERO, 0x0F),
        xor(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 0xF0);
}

#[test]
fn test_srl() {
    // t0 = 256, t1 = 4, srl a0, t0, t1 -> 16
    let result = run_program(&[
        addi(T0, ZERO, 256),
        addi(T1, ZERO, 4),
        srl(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 16);
}

#[test]
fn test_sra() {
    // t0 = 0x80000000, t1 = 4, sra a0, t0, t1 -> 0xF8000000
    let result = run_program(&[
        lui(T0, 0x80000000u32 as i32),
        addi(T1, ZERO, 4),
        sra(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 0xF8000000);
}

#[test]
fn test_or() {
    // t0 = 0xF0, t1 = 0x0F, or a0, t0, t1 -> 0xFF
    let result = run_program(&[
        addi(T0, ZERO, 0xF0),
        addi(T1, ZERO, 0x0F),
        or(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 0xFF);
}

#[test]
fn test_and() {
    // t0 = 0xFF, t1 = 0x0F, and a0, t0, t1 -> 0x0F
    let result = run_program(&[
        addi(T0, ZERO, 0xFF),
        addi(T1, ZERO, 0x0F),
        and(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 0x0F);
}

// ============================================================
// Branch Tests
// ============================================================

#[test]
fn test_beq_taken() {
    // t0 = 5, t1 = 5, beq t0, t1, skip
    // li a0, 1  ; skip this
    // skip: li a0, 42
    let result = run_program(&[
        addi(T0, ZERO, 5),
        addi(T1, ZERO, 5),
        beq(T0, T1, 8),  // skip next instruction
        addi(A0, ZERO, 1),
        addi(A0, ZERO, 42),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

#[test]
fn test_beq_not_taken() {
    // t0 = 5, t1 = 6, beq t0, t1, skip (not taken)
    let result = run_program(&[
        addi(T0, ZERO, 5),
        addi(T1, ZERO, 6),
        beq(T0, T1, 8),
        addi(A0, ZERO, 42),
        addi(A0, ZERO, 1),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_bne_taken() {
    let result = run_program(&[
        addi(T0, ZERO, 5),
        addi(T1, ZERO, 6),
        bne(T0, T1, 8),
        addi(A0, ZERO, 1),
        addi(A0, ZERO, 42),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

#[test]
fn test_bne_not_taken() {
    let result = run_program(&[
        addi(T0, ZERO, 5),
        addi(T1, ZERO, 5),
        bne(T0, T1, 8),
        addi(A0, ZERO, 42),
        addi(A0, ZERO, 1),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_blt_taken() {
    // -1 < 0 (signed)
    let result = run_program(&[
        addi(T0, ZERO, -1),
        addi(T1, ZERO, 0),
        blt(T0, T1, 8),
        addi(A0, ZERO, 1),
        addi(A0, ZERO, 42),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

#[test]
fn test_blt_not_taken() {
    // 0 >= -1 (signed)
    let result = run_program(&[
        addi(T0, ZERO, 0),
        addi(T1, ZERO, -1),
        blt(T0, T1, 8),
        addi(A0, ZERO, 42),
        addi(A0, ZERO, 1),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_bge_taken() {
    // 0 >= -1 (signed)
    let result = run_program(&[
        addi(T0, ZERO, 0),
        addi(T1, ZERO, -1),
        bge(T0, T1, 8),
        addi(A0, ZERO, 1),
        addi(A0, ZERO, 42),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

#[test]
fn test_bge_not_taken() {
    // -1 < 0 (signed)
    let result = run_program(&[
        addi(T0, ZERO, -1),
        addi(T1, ZERO, 0),
        bge(T0, T1, 8),
        addi(A0, ZERO, 42),
        addi(A0, ZERO, 1),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_bltu_taken() {
    // 1 < 0xFFFFFFFF (unsigned)
    let result = run_program(&[
        addi(T0, ZERO, 1),
        addi(T1, ZERO, -1),
        bltu(T0, T1, 8),
        addi(A0, ZERO, 1),
        addi(A0, ZERO, 42),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

#[test]
fn test_bltu_not_taken() {
    // 0xFFFFFFFF >= 1 (unsigned)
    let result = run_program(&[
        addi(T0, ZERO, -1),
        addi(T1, ZERO, 1),
        bltu(T0, T1, 8),
        addi(A0, ZERO, 42),
        addi(A0, ZERO, 1),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_bgeu_taken() {
    // 0xFFFFFFFF >= 1 (unsigned)
    let result = run_program(&[
        addi(T0, ZERO, -1),
        addi(T1, ZERO, 1),
        bgeu(T0, T1, 8),
        addi(A0, ZERO, 1),
        addi(A0, ZERO, 42),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

#[test]
fn test_bgeu_not_taken() {
    // 1 < 0xFFFFFFFF (unsigned)
    let result = run_program(&[
        addi(T0, ZERO, 1),
        addi(T1, ZERO, -1),
        bgeu(T0, T1, 8),
        addi(A0, ZERO, 42),
        addi(A0, ZERO, 1),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

// ============================================================
// Jump Tests
// ============================================================

#[test]
fn test_jal() {
    // jal ra, 8 ; skip next, save return address
    // li a0, 1  ; skipped
    // li a0, 42
    let result = run_program(&[
        jal(RA, 8),
        addi(A0, ZERO, 1),
        addi(A0, ZERO, 42),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

#[test]
fn test_jal_return_address() {
    // jal ra, 8 ; ra = PC + 4 = 0x80000004
    // nop
    // mv a0, ra
    let result = run_program(&[
        jal(RA, 8),
        nop(),
        mv(A0, RA),
        ecall(),
    ]);
    assert_eq!(result, 0x80000004);
}

// ============================================================
// RV32M Multiply/Divide Tests
// ============================================================

#[test]
fn test_mul() {
    // 6 * 7 = 42
    let result = run_program(&[
        addi(T0, ZERO, 6),
        addi(T1, ZERO, 7),
        mul(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

#[test]
fn test_mul_negative() {
    // -6 * 7 = -42
    let result = run_program(&[
        addi(T0, ZERO, -6),
        addi(T1, ZERO, 7),
        mul(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, (-42i32) as u32);
}

#[test]
fn test_mul_overflow() {
    // 0x10000 * 0x10000 = 0x100000000, but low 32 bits = 0
    let result = run_program(&[
        lui(T0, 0x10000),
        lui(T1, 0x10000),
        mul(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 0);
}

#[test]
fn test_mulh() {
    // mulh: high 32 bits of signed multiply
    // 0x7FFFFFFF * 0x7FFFFFFF -> high bits
    // To load 0x7FFFFFFF: lui gives 0x80000000, then addi -1 gives 0x7FFFFFFF
    let result = run_program(&[
        lui(T0, 0x80000000u32 as i32),  // T0 = 0x80000000
        addi(T0, T0, -1),                // T0 = 0x7FFFFFFF
        mv(T1, T0),
        mulh(A0, T0, T1),
        ecall(),
    ]);
    // 0x7FFFFFFF^2 = 0x3FFFFFFF00000001, high 32 bits = 0x3FFFFFFF
    assert_eq!(result, 0x3FFFFFFF);
}

#[test]
fn test_mulhu() {
    // mulhu: high 32 bits of unsigned multiply
    let result = run_program(&[
        lui(T0, 0x10000),
        lui(T1, 0x10000),
        mulhu(A0, T0, T1),
        ecall(),
    ]);
    // 0x10000000 * 0x10000000 = 0x100000000000000, high = 0x1
    assert_eq!(result, 1);
}

#[test]
fn test_div() {
    // 42 / 6 = 7
    let result = run_program(&[
        addi(T0, ZERO, 42),
        addi(T1, ZERO, 6),
        div(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 7);
}

#[test]
fn test_div_negative() {
    // -42 / 6 = -7
    let result = run_program(&[
        addi(T0, ZERO, -42),
        addi(T1, ZERO, 6),
        div(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, (-7i32) as u32);
}

#[test]
fn test_div_by_zero() {
    // Division by zero returns -1 (0xFFFFFFFF)
    let result = run_program(&[
        addi(T0, ZERO, 42),
        addi(T1, ZERO, 0),
        div(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 0xFFFFFFFF);
}

#[test]
fn test_divu() {
    // 42 / 6 = 7 (unsigned)
    let result = run_program(&[
        addi(T0, ZERO, 42),
        addi(T1, ZERO, 6),
        divu(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 7);
}

#[test]
fn test_divu_by_zero() {
    // Unsigned division by zero returns MAX_UINT
    let result = run_program(&[
        addi(T0, ZERO, 42),
        addi(T1, ZERO, 0),
        divu(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 0xFFFFFFFF);
}

#[test]
fn test_rem() {
    // 43 % 6 = 1
    let result = run_program(&[
        addi(T0, ZERO, 43),
        addi(T1, ZERO, 6),
        rem(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_rem_negative() {
    // -43 % 6 = -1
    let result = run_program(&[
        addi(T0, ZERO, -43),
        addi(T1, ZERO, 6),
        rem(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, (-1i32) as u32);
}

#[test]
fn test_rem_by_zero() {
    // Remainder by zero returns dividend
    let result = run_program(&[
        addi(T0, ZERO, 42),
        addi(T1, ZERO, 0),
        rem(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

#[test]
fn test_remu() {
    // 43 % 6 = 1 (unsigned)
    let result = run_program(&[
        addi(T0, ZERO, 43),
        addi(T1, ZERO, 6),
        remu(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 1);
}

#[test]
fn test_remu_by_zero() {
    // Unsigned remainder by zero returns dividend
    let result = run_program(&[
        addi(T0, ZERO, 42),
        addi(T1, ZERO, 0),
        remu(A0, T0, T1),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

// ============================================================
// Load/Store Tests
// ============================================================

#[test]
fn test_sw_lw() {
    // Store and load a word
    // sw t0, 0(sp) ; store 0x12345678
    // lw a0, 0(sp) ; load it back
    let result = run_program(&[
        lui(T0, 0x12345000u32 as i32),
        ori(T0, T0, 0x678),
        sw(SP, T0, 0),
        lw(A0, SP, 0),
        ecall(),
    ]);
    assert_eq!(result, 0x12345678);
}

#[test]
fn test_sh_lh() {
    // Store and load a halfword (signed)
    // Use LUI + ORI to load 0x1234 (which doesn't fit in 12-bit immediate)
    let result = run_program(&[
        lui(T0, 0x1000),       // T0 = 0x00001000
        ori(T0, T0, 0x234),    // T0 = 0x00001234
        sh(SP, T0, 0),
        lh(A0, SP, 0),
        ecall(),
    ]);
    assert_eq!(result, 0x1234);
}

#[test]
fn test_sh_lh_sign_extend() {
    // Store 0x8000, load should sign-extend to 0xFFFF8000
    // LUI with bit 31 set gives 0x80000000, then SRLI by 16 gives 0x8000
    let result = run_program(&[
        lui(T0, 0x80000000u32 as i32),  // T0 = 0x80000000
        srli(T0, T0, 16),                // T0 = 0x8000
        sh(SP, T0, 0),
        lh(A0, SP, 0),
        ecall(),
    ]);
    assert_eq!(result, 0xFFFF8000);
}

#[test]
fn test_sh_lhu() {
    // Store 0x8000, load unsigned should not sign-extend
    let result = run_program(&[
        lui(T0, 0x80000000u32 as i32),  // T0 = 0x80000000
        srli(T0, T0, 16),                // T0 = 0x8000
        sh(SP, T0, 0),
        lhu(A0, SP, 0),
        ecall(),
    ]);
    assert_eq!(result, 0x8000);
}

#[test]
fn test_sb_lb() {
    // Store and load a byte (signed)
    let result = run_program(&[
        addi(T0, ZERO, 0x42),
        sb(SP, T0, 0),
        lb(A0, SP, 0),
        ecall(),
    ]);
    assert_eq!(result, 0x42);
}

#[test]
fn test_sb_lb_sign_extend() {
    // Store 0x80, load should sign-extend
    let result = run_program(&[
        addi(T0, ZERO, 0x80),
        sb(SP, T0, 0),
        lb(A0, SP, 0),
        ecall(),
    ]);
    assert_eq!(result, 0xFFFFFF80);
}

#[test]
fn test_sb_lbu() {
    // Store 0x80, load unsigned should not sign-extend
    let result = run_program(&[
        addi(T0, ZERO, 0x80),
        sb(SP, T0, 0),
        lbu(A0, SP, 0),
        ecall(),
    ]);
    assert_eq!(result, 0x80);
}

// ============================================================
// Zero Register Tests
// ============================================================

#[test]
fn test_zero_register_always_zero() {
    // Writing to zero register should have no effect
    let result = run_program(&[
        addi(ZERO, ZERO, 42), // Write to zero - should be ignored
        mv(A0, ZERO),        // A0 = zero = 0
        ecall(),
    ]);
    assert_eq!(result, 0);
}

#[test]
fn test_add_with_zero() {
    // add rd, rs, zero = mv rd, rs
    let result = run_program(&[
        addi(T0, ZERO, 42),
        add(A0, T0, ZERO),
        ecall(),
    ]);
    assert_eq!(result, 42);
}

// ============================================================
// Complex Program Tests
// ============================================================

#[test]
fn test_fibonacci() {
    // Calculate Fibonacci(10) = 55
    // fib(n):
    //   if n <= 1: return n
    //   return fib(n-1) + fib(n-2)
    //
    // Iterative version:
    // a = 0, b = 1
    // for i in range(n):
    //     a, b = b, a + b
    // return a

    let result = run_program(&[
        // n = 10
        addi(T0, ZERO, 10),  // PC 0: t0 = n = 10
        addi(T1, ZERO, 0),   // PC 4: t1 = a = 0
        addi(T2, ZERO, 1),   // PC 8: t2 = b = 1
        // loop:
        beq(T0, ZERO, 24),   // PC 12: if n == 0, goto end (PC 36)
        add(T3, T1, T2),     // PC 16: t3 = a + b
        mv(T1, T2),          // PC 20: a = b
        mv(T2, T3),          // PC 24: b = a + b
        addi(T0, T0, -1),    // PC 28: n--
        jal(ZERO, -20),      // PC 32: goto loop (PC 12)
        // end:
        mv(A0, T1),          // PC 36: return a
        ecall(),             // PC 40
    ]);
    assert_eq!(result, 55);
}

#[test]
fn test_factorial() {
    // Calculate factorial(6) = 720
    // fact(n):
    //   result = 1
    //   while n > 0:
    //     result *= n
    //     n -= 1
    //   return result

    let result = run_program(&[
        addi(T0, ZERO, 6),   // t0 = n = 6
        addi(T1, ZERO, 1),   // t1 = result = 1
        // loop:
        beq(T0, ZERO, 16),   // if n == 0, goto end
        mul(T1, T1, T0),     // result *= n
        addi(T0, T0, -1),    // n--
        jal(ZERO, -12),      // goto loop
        // end:
        mv(A0, T1),
        ecall(),
    ]);
    assert_eq!(result, 720);
}

#[test]
fn test_gcd() {
    // Calculate GCD(48, 18) = 6
    // Using Euclidean algorithm
    // gcd(a, b):
    //   while b != 0:
    //     a, b = b, a % b
    //   return a

    let result = run_program(&[
        addi(T0, ZERO, 48),  // t0 = a = 48
        addi(T1, ZERO, 18),  // t1 = b = 18
        // loop:
        beq(T1, ZERO, 20),   // if b == 0, goto end
        remu(T2, T0, T1),    // t2 = a % b
        mv(T0, T1),          // a = b
        mv(T1, T2),          // b = a % b
        jal(ZERO, -16),      // goto loop
        // end:
        mv(A0, T0),
        ecall(),
    ]);
    assert_eq!(result, 6);
}

#[test]
fn test_sum_1_to_n() {
    // Sum of 1 to 10 = 55
    let result = run_program(&[
        addi(T0, ZERO, 10),  // t0 = n = 10
        addi(T1, ZERO, 0),   // t1 = sum = 0
        // loop:
        beq(T0, ZERO, 16),   // if n == 0, goto end
        add(T1, T1, T0),     // sum += n
        addi(T0, T0, -1),    // n--
        jal(ZERO, -12),      // goto loop
        // end:
        mv(A0, T1),
        ecall(),
    ]);
    assert_eq!(result, 55);
}

#[test]
fn test_power_of_2() {
    // Calculate 2^10 = 1024
    let result = run_program(&[
        addi(T0, ZERO, 10),  // t0 = exponent = 10
        addi(T1, ZERO, 1),   // t1 = result = 1
        // loop:
        beq(T0, ZERO, 16),   // if exp == 0, goto end
        slli(T1, T1, 1),     // result *= 2
        addi(T0, T0, -1),    // exp--
        jal(ZERO, -12),      // goto loop
        // end:
        mv(A0, T1),
        ecall(),
    ]);
    assert_eq!(result, 1024);
}

// ============================================================
// Decoder Tests
// ============================================================

#[test]
fn test_decode_lui() {
    let instr = decode_instruction(lui(A0, 0x12345000u32 as i32));
    assert!(matches!(instr, Instruction::Lui { rd: 10, imm: 0x12345000 }));
}

#[test]
fn test_decode_addi() {
    let instr = decode_instruction(addi(A0, T0, 42));
    assert!(matches!(instr, Instruction::Addi { rd: 10, rs1: 5, imm: 42 }));
}

#[test]
fn test_decode_add() {
    let instr = decode_instruction(add(A0, T0, T1));
    assert!(matches!(instr, Instruction::Add { rd: 10, rs1: 5, rs2: 6 }));
}

#[test]
fn test_decode_mul() {
    let instr = decode_instruction(mul(A0, T0, T1));
    assert!(matches!(instr, Instruction::Mul { rd: 10, rs1: 5, rs2: 6 }));
}

#[test]
fn test_decode_beq() {
    let instr = decode_instruction(beq(T0, T1, 8));
    assert!(matches!(instr, Instruction::Beq { rs1: 5, rs2: 6, imm: 8 }));
}

#[test]
fn test_decode_jal() {
    let instr = decode_instruction(jal(RA, 16));
    assert!(matches!(instr, Instruction::Jal { rd: 1, imm: 16 }));
}

#[test]
fn test_decode_ecall() {
    let instr = decode_instruction(ecall());
    assert!(matches!(instr, Instruction::Ecall));
}

#[test]
fn test_debug_bytecode() {
    // Simple test: just return 42
    let mut program = Vec::new();
    program.extend_from_slice(&addi(A0, ZERO, 42).to_le_bytes());
    program.extend_from_slice(&ecall().to_le_bytes());

    let mut compiler = test_compiler();
    let bytecode = compiler.compile(&program).expect("Compilation failed");

    eprintln!("Simple program bytecode length: {} bytes", bytecode.len());

    let runtime = Runtime::new(bytecode);
    let output = runtime.execute().expect("Execution failed");
    assert_eq!(output.return_value, 42);
}

#[test]
fn test_debug_branch() {
    // Test SLT (set less than) which uses EQ-like comparison
    // This tests if register comparison works correctly
    let instrs = [
        addi(T0, ZERO, 5),   // 0: t0 = 5
        addi(T1, ZERO, 6),   // 1: t1 = 6
        slt(A0, T0, T1),     // 2: a0 = (t0 < t1) = 1
        ecall(),             // 3
    ];

    eprintln!("\n=== SLT Test ===");

    // Decode and print each instruction
    for (i, &raw) in instrs.iter().enumerate() {
        let decoded = decode_instruction(raw);
        eprintln!("Instr {}: {:#010x} -> {}", i, raw, decoded);
    }

    let mut program = Vec::new();
    for instr in &instrs {
        program.extend_from_slice(&instr.to_le_bytes());
    }

    let mut compiler = test_compiler();
    let bytecode = compiler.compile(&program).expect("Compilation failed");

    eprintln!("Bytecode length: {} bytes", bytecode.len());
    eprintln!("Bytecode hex: {}", hex::encode(&bytecode[0..100.min(bytecode.len())]));

    let runtime = Runtime::new(bytecode);
    let result = runtime.execute();

    match result {
        Ok(output) => {
            eprintln!("Result: {} (expected 1)", output.return_value);
            assert_eq!(output.return_value, 1);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            panic!("Execution failed: {}", e);
        }
    }
}
