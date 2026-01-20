//! RISC-V rv32im instruction decoder
//!
//! Decodes 32-bit RISC-V instructions into a structured format.

use std::fmt;

/// RISC-V register names
pub const REG_NAMES: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
];

/// Decoded RISC-V instruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    // RV32I Base Integer Instructions

    // U-type
    Lui { rd: u8, imm: i32 },
    Auipc { rd: u8, imm: i32 },

    // J-type
    Jal { rd: u8, imm: i32 },

    // I-type (jumps)
    Jalr { rd: u8, rs1: u8, imm: i32 },

    // B-type (branches)
    Beq { rs1: u8, rs2: u8, imm: i32 },
    Bne { rs1: u8, rs2: u8, imm: i32 },
    Blt { rs1: u8, rs2: u8, imm: i32 },
    Bge { rs1: u8, rs2: u8, imm: i32 },
    Bltu { rs1: u8, rs2: u8, imm: i32 },
    Bgeu { rs1: u8, rs2: u8, imm: i32 },

    // I-type (loads)
    Lb { rd: u8, rs1: u8, imm: i32 },
    Lh { rd: u8, rs1: u8, imm: i32 },
    Lw { rd: u8, rs1: u8, imm: i32 },
    Lbu { rd: u8, rs1: u8, imm: i32 },
    Lhu { rd: u8, rs1: u8, imm: i32 },

    // S-type (stores)
    Sb { rs1: u8, rs2: u8, imm: i32 },
    Sh { rs1: u8, rs2: u8, imm: i32 },
    Sw { rs1: u8, rs2: u8, imm: i32 },

    // I-type (arithmetic immediate)
    Addi { rd: u8, rs1: u8, imm: i32 },
    Slti { rd: u8, rs1: u8, imm: i32 },
    Sltiu { rd: u8, rs1: u8, imm: i32 },
    Xori { rd: u8, rs1: u8, imm: i32 },
    Ori { rd: u8, rs1: u8, imm: i32 },
    Andi { rd: u8, rs1: u8, imm: i32 },
    Slli { rd: u8, rs1: u8, shamt: u8 },
    Srli { rd: u8, rs1: u8, shamt: u8 },
    Srai { rd: u8, rs1: u8, shamt: u8 },

    // R-type (arithmetic register)
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

    // System instructions
    Fence { pred: u8, succ: u8 },
    Ecall,
    Ebreak,

    // RV32M Standard Extension (Multiply/Divide)
    Mul { rd: u8, rs1: u8, rs2: u8 },
    Mulh { rd: u8, rs1: u8, rs2: u8 },
    Mulhsu { rd: u8, rs1: u8, rs2: u8 },
    Mulhu { rd: u8, rs1: u8, rs2: u8 },
    Div { rd: u8, rs1: u8, rs2: u8 },
    Divu { rd: u8, rs1: u8, rs2: u8 },
    Rem { rd: u8, rs1: u8, rs2: u8 },
    Remu { rd: u8, rs1: u8, rs2: u8 },

    // Unknown/invalid instruction
    Unknown { raw: u32 },
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Lui { rd, imm } => write!(f, "lui {}, {:#x}", REG_NAMES[*rd as usize], imm),
            Instruction::Auipc { rd, imm } => write!(f, "auipc {}, {:#x}", REG_NAMES[*rd as usize], imm),
            Instruction::Jal { rd, imm } => write!(f, "jal {}, {}", REG_NAMES[*rd as usize], imm),
            Instruction::Jalr { rd, rs1, imm } => write!(f, "jalr {}, {}({})", REG_NAMES[*rd as usize], imm, REG_NAMES[*rs1 as usize]),
            Instruction::Beq { rs1, rs2, imm } => write!(f, "beq {}, {}, {}", REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize], imm),
            Instruction::Bne { rs1, rs2, imm } => write!(f, "bne {}, {}, {}", REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize], imm),
            Instruction::Blt { rs1, rs2, imm } => write!(f, "blt {}, {}, {}", REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize], imm),
            Instruction::Bge { rs1, rs2, imm } => write!(f, "bge {}, {}, {}", REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize], imm),
            Instruction::Bltu { rs1, rs2, imm } => write!(f, "bltu {}, {}, {}", REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize], imm),
            Instruction::Bgeu { rs1, rs2, imm } => write!(f, "bgeu {}, {}, {}", REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize], imm),
            Instruction::Lb { rd, rs1, imm } => write!(f, "lb {}, {}({})", REG_NAMES[*rd as usize], imm, REG_NAMES[*rs1 as usize]),
            Instruction::Lh { rd, rs1, imm } => write!(f, "lh {}, {}({})", REG_NAMES[*rd as usize], imm, REG_NAMES[*rs1 as usize]),
            Instruction::Lw { rd, rs1, imm } => write!(f, "lw {}, {}({})", REG_NAMES[*rd as usize], imm, REG_NAMES[*rs1 as usize]),
            Instruction::Lbu { rd, rs1, imm } => write!(f, "lbu {}, {}({})", REG_NAMES[*rd as usize], imm, REG_NAMES[*rs1 as usize]),
            Instruction::Lhu { rd, rs1, imm } => write!(f, "lhu {}, {}({})", REG_NAMES[*rd as usize], imm, REG_NAMES[*rs1 as usize]),
            Instruction::Sb { rs1, rs2, imm } => write!(f, "sb {}, {}({})", REG_NAMES[*rs2 as usize], imm, REG_NAMES[*rs1 as usize]),
            Instruction::Sh { rs1, rs2, imm } => write!(f, "sh {}, {}({})", REG_NAMES[*rs2 as usize], imm, REG_NAMES[*rs1 as usize]),
            Instruction::Sw { rs1, rs2, imm } => write!(f, "sw {}, {}({})", REG_NAMES[*rs2 as usize], imm, REG_NAMES[*rs1 as usize]),
            Instruction::Addi { rd, rs1, imm } => write!(f, "addi {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], imm),
            Instruction::Slti { rd, rs1, imm } => write!(f, "slti {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], imm),
            Instruction::Sltiu { rd, rs1, imm } => write!(f, "sltiu {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], imm),
            Instruction::Xori { rd, rs1, imm } => write!(f, "xori {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], imm),
            Instruction::Ori { rd, rs1, imm } => write!(f, "ori {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], imm),
            Instruction::Andi { rd, rs1, imm } => write!(f, "andi {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], imm),
            Instruction::Slli { rd, rs1, shamt } => write!(f, "slli {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], shamt),
            Instruction::Srli { rd, rs1, shamt } => write!(f, "srli {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], shamt),
            Instruction::Srai { rd, rs1, shamt } => write!(f, "srai {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], shamt),
            Instruction::Add { rd, rs1, rs2 } => write!(f, "add {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Sub { rd, rs1, rs2 } => write!(f, "sub {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Sll { rd, rs1, rs2 } => write!(f, "sll {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Slt { rd, rs1, rs2 } => write!(f, "slt {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Sltu { rd, rs1, rs2 } => write!(f, "sltu {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Xor { rd, rs1, rs2 } => write!(f, "xor {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Srl { rd, rs1, rs2 } => write!(f, "srl {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Sra { rd, rs1, rs2 } => write!(f, "sra {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Or { rd, rs1, rs2 } => write!(f, "or {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::And { rd, rs1, rs2 } => write!(f, "and {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Fence { pred, succ } => write!(f, "fence {}, {}", pred, succ),
            Instruction::Ecall => write!(f, "ecall"),
            Instruction::Ebreak => write!(f, "ebreak"),
            Instruction::Mul { rd, rs1, rs2 } => write!(f, "mul {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Mulh { rd, rs1, rs2 } => write!(f, "mulh {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Mulhsu { rd, rs1, rs2 } => write!(f, "mulhsu {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Mulhu { rd, rs1, rs2 } => write!(f, "mulhu {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Div { rd, rs1, rs2 } => write!(f, "div {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Divu { rd, rs1, rs2 } => write!(f, "divu {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Rem { rd, rs1, rs2 } => write!(f, "rem {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Remu { rd, rs1, rs2 } => write!(f, "remu {}, {}, {}", REG_NAMES[*rd as usize], REG_NAMES[*rs1 as usize], REG_NAMES[*rs2 as usize]),
            Instruction::Unknown { raw } => write!(f, "unknown {:#010x}", raw),
        }
    }
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
        // LUI
        0b0110111 => {
            let imm = (raw & 0xFFFFF000) as i32;
            Instruction::Lui { rd, imm }
        }

        // AUIPC
        0b0010111 => {
            let imm = (raw & 0xFFFFF000) as i32;
            Instruction::Auipc { rd, imm }
        }

        // JAL
        0b1101111 => {
            let imm20 = (raw >> 31) & 1;
            let imm10_1 = (raw >> 21) & 0x3FF;
            let imm11 = (raw >> 20) & 1;
            let imm19_12 = (raw >> 12) & 0xFF;
            let imm = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
            let imm = sign_extend(imm, 21);
            Instruction::Jal { rd, imm }
        }

        // JALR
        0b1100111 => {
            let imm = sign_extend(raw >> 20, 12);
            Instruction::Jalr { rd, rs1, imm }
        }

        // Branch instructions
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

        // Load instructions
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

        // Store instructions
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

        // Immediate arithmetic
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

        // Register arithmetic (including M extension)
        0b0110011 => {
            match (funct7, funct3) {
                // RV32I
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

                // RV32M
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

        // FENCE
        0b0001111 => {
            let pred = ((raw >> 24) & 0xF) as u8;
            let succ = ((raw >> 20) & 0xF) as u8;
            Instruction::Fence { pred, succ }
        }

        // SYSTEM (ECALL/EBREAK)
        0b1110011 => {
            let imm = (raw >> 20) & 0xFFF;
            match imm {
                0 => Instruction::Ecall,
                1 => Instruction::Ebreak,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lui() {
        // lui x1, 0x12345
        let instr = decode_instruction(0x12345_0b7);
        assert!(matches!(instr, Instruction::Lui { rd: 1, imm: 0x12345000 }));
    }

    #[test]
    fn test_addi() {
        // addi x1, x2, 100
        let instr = decode_instruction(0x06410093);
        assert!(matches!(instr, Instruction::Addi { rd: 1, rs1: 2, imm: 100 }));
    }

    #[test]
    fn test_add() {
        // add x1, x2, x3
        let instr = decode_instruction(0x003100b3);
        assert!(matches!(instr, Instruction::Add { rd: 1, rs1: 2, rs2: 3 }));
    }

    #[test]
    fn test_beq() {
        // beq x1, x2, 8
        let instr = decode_instruction(0x00208463);
        assert!(matches!(instr, Instruction::Beq { rs1: 1, rs2: 2, imm: 8 }));
    }

    #[test]
    fn test_jal() {
        // jal x1, 16
        let instr = decode_instruction(0x010000ef);
        assert!(matches!(instr, Instruction::Jal { rd: 1, imm: 16 }));
    }

    #[test]
    fn test_mul() {
        // mul x1, x2, x3
        let instr = decode_instruction(0x023100b3);
        assert!(matches!(instr, Instruction::Mul { rd: 1, rs1: 2, rs2: 3 }));
    }
}
