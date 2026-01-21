//! RISC-V rv32im instruction decoder (no_std version)

use alloc::vec::Vec;

/// Decoded RISC-V instruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    // RV32I Base Integer Instructions
    Lui { rd: u8, imm: i32 },
    Auipc { rd: u8, imm: i32 },
    Jal { rd: u8, imm: i32 },
    Jalr { rd: u8, rs1: u8, imm: i32 },
    Beq { rs1: u8, rs2: u8, imm: i32 },
    Bne { rs1: u8, rs2: u8, imm: i32 },
    Blt { rs1: u8, rs2: u8, imm: i32 },
    Bge { rs1: u8, rs2: u8, imm: i32 },
    Bltu { rs1: u8, rs2: u8, imm: i32 },
    Bgeu { rs1: u8, rs2: u8, imm: i32 },
    Lb { rd: u8, rs1: u8, imm: i32 },
    Lh { rd: u8, rs1: u8, imm: i32 },
    Lw { rd: u8, rs1: u8, imm: i32 },
    Lbu { rd: u8, rs1: u8, imm: i32 },
    Lhu { rd: u8, rs1: u8, imm: i32 },
    Sb { rs1: u8, rs2: u8, imm: i32 },
    Sh { rs1: u8, rs2: u8, imm: i32 },
    Sw { rs1: u8, rs2: u8, imm: i32 },
    Addi { rd: u8, rs1: u8, imm: i32 },
    Slti { rd: u8, rs1: u8, imm: i32 },
    Sltiu { rd: u8, rs1: u8, imm: i32 },
    Xori { rd: u8, rs1: u8, imm: i32 },
    Ori { rd: u8, rs1: u8, imm: i32 },
    Andi { rd: u8, rs1: u8, imm: i32 },
    Slli { rd: u8, rs1: u8, shamt: u8 },
    Srli { rd: u8, rs1: u8, shamt: u8 },
    Srai { rd: u8, rs1: u8, shamt: u8 },
    Add { rd: u8, rs1: u8, rs2: u8 },
    Sub { rd: u8, rs1: u8, rs2: u8 },
    Sll { rd: u8, rs1: u8, rs2: u8 },
    Slt { rd: u8, rs1: u8, rs2: u8 },
    Sltu { rd: u8, rs1: u8, rs2: u8 },
    Xor { rd: u8, rs1: u8, rs2: u8 },
    Srl { rd: u8, rs1: u8, rs2: u8 },
    Sra { rd: u8, rs1: u8, rs2: u8 },
    Or { rd: u8, rs1: u8, rs2: u8 },
    And { rd: u8, rs1: u8, rs2: u8 },
    Fence { pred: u8, succ: u8 },
    Ecall,
    Ebreak,
    // CSR instructions (treated as no-ops)
    Csrrw { rd: u8, rs1: u8, csr: u16 },
    Csrrs { rd: u8, rs1: u8, csr: u16 },
    Csrrc { rd: u8, rs1: u8, csr: u16 },
    Csrrwi { rd: u8, uimm: u8, csr: u16 },
    Csrrsi { rd: u8, uimm: u8, csr: u16 },
    Csrrci { rd: u8, uimm: u8, csr: u16 },
    // RV32M
    Mul { rd: u8, rs1: u8, rs2: u8 },
    Mulh { rd: u8, rs1: u8, rs2: u8 },
    Mulhsu { rd: u8, rs1: u8, rs2: u8 },
    Mulhu { rd: u8, rs1: u8, rs2: u8 },
    Div { rd: u8, rs1: u8, rs2: u8 },
    Divu { rd: u8, rs1: u8, rs2: u8 },
    Rem { rd: u8, rs1: u8, rs2: u8 },
    Remu { rd: u8, rs1: u8, rs2: u8 },
    Unknown { raw: u32 },
}

/// Sign-extend a value from `bits` to 32 bits
fn sign_extend(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((value << shift) as i32) >> shift
}

/// Decode a single RISC-V instruction
pub fn decode_instruction(raw: u32) -> Instruction {
    let opcode = raw & 0x7F;
    let rd = ((raw >> 7) & 0x1F) as u8;
    let funct3 = (raw >> 12) & 0x7;
    let rs1 = ((raw >> 15) & 0x1F) as u8;
    let rs2 = ((raw >> 20) & 0x1F) as u8;
    let funct7 = (raw >> 25) & 0x7F;

    match opcode {
        0b0110111 => {
            let imm = (raw & 0xFFFFF000) as i32;
            Instruction::Lui { rd, imm }
        }
        0b0010111 => {
            let imm = (raw & 0xFFFFF000) as i32;
            Instruction::Auipc { rd, imm }
        }
        0b1101111 => {
            let imm20 = (raw >> 31) & 1;
            let imm10_1 = (raw >> 21) & 0x3FF;
            let imm11 = (raw >> 20) & 1;
            let imm19_12 = (raw >> 12) & 0xFF;
            let imm = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
            let imm = sign_extend(imm, 21);
            Instruction::Jal { rd, imm }
        }
        0b1100111 => {
            let imm = sign_extend(raw >> 20, 12);
            Instruction::Jalr { rd, rs1, imm }
        }
        0b1100011 => {
            let imm12 = (raw >> 31) & 1;
            let imm10_5 = (raw >> 25) & 0x3F;
            let imm4_1 = (raw >> 8) & 0xF;
            let imm11 = (raw >> 7) & 1;
            let imm = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
            let imm = sign_extend(imm, 13);
            match funct3 {
                0b000 => Instruction::Beq { rs1, rs2, imm },
                0b001 => Instruction::Bne { rs1, rs2, imm },
                0b100 => Instruction::Blt { rs1, rs2, imm },
                0b101 => Instruction::Bge { rs1, rs2, imm },
                0b110 => Instruction::Bltu { rs1, rs2, imm },
                0b111 => Instruction::Bgeu { rs1, rs2, imm },
                _ => Instruction::Unknown { raw },
            }
        }
        0b0000011 => {
            let imm = sign_extend(raw >> 20, 12);
            match funct3 {
                0b000 => Instruction::Lb { rd, rs1, imm },
                0b001 => Instruction::Lh { rd, rs1, imm },
                0b010 => Instruction::Lw { rd, rs1, imm },
                0b100 => Instruction::Lbu { rd, rs1, imm },
                0b101 => Instruction::Lhu { rd, rs1, imm },
                _ => Instruction::Unknown { raw },
            }
        }
        0b0100011 => {
            let imm11_5 = (raw >> 25) & 0x7F;
            let imm4_0 = (raw >> 7) & 0x1F;
            let imm = sign_extend((imm11_5 << 5) | imm4_0, 12);
            match funct3 {
                0b000 => Instruction::Sb { rs1, rs2, imm },
                0b001 => Instruction::Sh { rs1, rs2, imm },
                0b010 => Instruction::Sw { rs1, rs2, imm },
                _ => Instruction::Unknown { raw },
            }
        }
        0b0010011 => {
            let imm = sign_extend(raw >> 20, 12);
            let shamt = ((raw >> 20) & 0x1F) as u8;
            match funct3 {
                0b000 => Instruction::Addi { rd, rs1, imm },
                0b010 => Instruction::Slti { rd, rs1, imm },
                0b011 => Instruction::Sltiu { rd, rs1, imm },
                0b100 => Instruction::Xori { rd, rs1, imm },
                0b110 => Instruction::Ori { rd, rs1, imm },
                0b111 => Instruction::Andi { rd, rs1, imm },
                0b001 => Instruction::Slli { rd, rs1, shamt },
                0b101 => {
                    if funct7 == 0b0100000 {
                        Instruction::Srai { rd, rs1, shamt }
                    } else {
                        Instruction::Srli { rd, rs1, shamt }
                    }
                }
                _ => Instruction::Unknown { raw },
            }
        }
        0b0110011 => {
            match (funct7, funct3) {
                (0b0000000, 0b000) => Instruction::Add { rd, rs1, rs2 },
                (0b0100000, 0b000) => Instruction::Sub { rd, rs1, rs2 },
                (0b0000000, 0b001) => Instruction::Sll { rd, rs1, rs2 },
                (0b0000000, 0b010) => Instruction::Slt { rd, rs1, rs2 },
                (0b0000000, 0b011) => Instruction::Sltu { rd, rs1, rs2 },
                (0b0000000, 0b100) => Instruction::Xor { rd, rs1, rs2 },
                (0b0000000, 0b101) => Instruction::Srl { rd, rs1, rs2 },
                (0b0100000, 0b101) => Instruction::Sra { rd, rs1, rs2 },
                (0b0000000, 0b110) => Instruction::Or { rd, rs1, rs2 },
                (0b0000000, 0b111) => Instruction::And { rd, rs1, rs2 },
                (0b0000001, 0b000) => Instruction::Mul { rd, rs1, rs2 },
                (0b0000001, 0b001) => Instruction::Mulh { rd, rs1, rs2 },
                (0b0000001, 0b010) => Instruction::Mulhsu { rd, rs1, rs2 },
                (0b0000001, 0b011) => Instruction::Mulhu { rd, rs1, rs2 },
                (0b0000001, 0b100) => Instruction::Div { rd, rs1, rs2 },
                (0b0000001, 0b101) => Instruction::Divu { rd, rs1, rs2 },
                (0b0000001, 0b110) => Instruction::Rem { rd, rs1, rs2 },
                (0b0000001, 0b111) => Instruction::Remu { rd, rs1, rs2 },
                _ => Instruction::Unknown { raw },
            }
        }
        0b0001111 => {
            let pred = ((raw >> 24) & 0xF) as u8;
            let succ = ((raw >> 20) & 0xF) as u8;
            Instruction::Fence { pred, succ }
        }
        0b1110011 => {
            let csr = ((raw >> 20) & 0xFFF) as u16;
            let uimm = ((raw >> 15) & 0x1F) as u8;
            match funct3 {
                0b000 => {
                    // ECALL/EBREAK/other system instructions
                    match csr {
                        0 => Instruction::Ecall,
                        1 => Instruction::Ebreak,
                        _ => Instruction::Unknown { raw },
                    }
                }
                0b001 => Instruction::Csrrw { rd, rs1, csr },
                0b010 => Instruction::Csrrs { rd, rs1, csr },
                0b011 => Instruction::Csrrc { rd, rs1, csr },
                0b101 => Instruction::Csrrwi { rd, uimm, csr },
                0b110 => Instruction::Csrrsi { rd, uimm, csr },
                0b111 => Instruction::Csrrci { rd, uimm, csr },
                _ => Instruction::Unknown { raw },
            }
        }
        _ => Instruction::Unknown { raw },
    }
}

/// Decode a sequence of bytes into instructions
pub fn decode_program(bytes: &[u8]) -> Vec<(u32, Instruction)> {
    let mut instructions = Vec::new();
    let mut offset = 0u32;

    while offset as usize + 4 <= bytes.len() {
        let raw = u32::from_le_bytes([
            bytes[offset as usize],
            bytes[offset as usize + 1],
            bytes[offset as usize + 2],
            bytes[offset as usize + 3],
        ]);
        instructions.push((offset, decode_instruction(raw)));
        offset += 4;
    }

    instructions
}
