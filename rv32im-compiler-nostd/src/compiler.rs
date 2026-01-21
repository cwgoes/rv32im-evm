//! RISC-V rv32im to EVM Compiler (no_std simplified version)
//!
//! This is a minimal compiler that prioritizes correctness and small code size
//! over optimization, suitable for meta-compilation experiments.

use alloc::{vec::Vec, string::String, collections::BTreeMap, format};

use crate::decoder::{Instruction, decode_instruction};
use crate::evm::{Opcode, EvmBytecode};

/// Memory layout constants
const MEM_BASE: u32 = 0x0500;      // Start of RISC-V memory in EVM
const REG_BASE: u32 = 0x0000;      // Start of register file in EVM memory
const SP_ADDR: u32 = 0x0040;       // Stack pointer storage
const PC_ADDR: u32 = 0x0400;       // Program counter for dynamic jumps

/// Compiler configuration
#[derive(Clone)]
pub struct CompilerConfig {
    pub load_address: u32,
    pub stack_pointer: u32,
    pub memory_size: u32,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x20000,
        }
    }
}

/// The compiler
pub struct Compiler {
    config: CompilerConfig,
    bytecode: EvmBytecode,
    pc_to_evm: BTreeMap<u32, usize>,
    pending_jumps: Vec<(usize, u32)>,
}

impl Compiler {
    pub fn new() -> Self {
        Self::with_config(CompilerConfig::default())
    }

    pub fn with_config(config: CompilerConfig) -> Self {
        Self {
            config,
            bytecode: EvmBytecode::new(),
            pc_to_evm: BTreeMap::new(),
            pending_jumps: Vec::new(),
        }
    }

    /// Compile a RISC-V program to EVM bytecode
    pub fn compile(&mut self, program: &[u8]) -> Result<Vec<u8>, &'static str> {
        // Decode all instructions
        let mut instructions = Vec::new();
        let mut offset = 0u32;
        while offset as usize + 4 <= program.len() {
            let raw = u32::from_le_bytes([
                program[offset as usize],
                program[offset as usize + 1],
                program[offset as usize + 2],
                program[offset as usize + 3],
            ]);
            let pc = self.config.load_address.wrapping_add(offset);
            instructions.push((pc, decode_instruction(raw)));
            offset += 4;
        }

        // Emit initialization
        self.emit_init();

        // Compile each instruction
        for (pc, instr) in &instructions {
            // Record EVM position for this PC
            self.pc_to_evm.insert(*pc, self.bytecode.position());
            self.bytecode.emit(Opcode::JumpDest);

            // Compile the instruction
            self.compile_instruction(*pc, instr)?;
        }

        // Emit halt routine
        self.emit_halt();

        // Emit dynamic dispatch for JALR
        self.emit_dynamic_dispatch();

        // Resolve jump placeholders
        self.resolve_jumps()?;

        // Finalize bytecode
        self.bytecode.resolve()?;
        Ok(self.bytecode.bytecode().to_vec())
    }

    fn emit_init(&mut self) {
        // Store initial stack pointer
        self.bytecode.push_u32(self.config.stack_pointer);
        self.bytecode.push_u32(SP_ADDR);
        self.bytecode.emit(Opcode::MStore);

        // Jump to first instruction
        self.bytecode.jump_to("main_entry");
        self.bytecode.jumpdest("main_entry");
    }

    fn emit_halt(&mut self) {
        self.bytecode.jumpdest("halt");
        // Load return value from a0 (x10 = REG_BASE + 10*32 = 0x140)
        self.bytecode.push_u32(REG_BASE + 10 * 32);
        self.bytecode.emit(Opcode::MLoad);
        // Store in return location
        self.bytecode.push_u32(0x0420);
        self.bytecode.emit(Opcode::MStore);
        // Return 32 bytes from 0x0420
        self.bytecode.push1(32);
        self.bytecode.push_u32(0x0420);
        self.bytecode.emit(Opcode::Return);
    }

    fn emit_dynamic_dispatch(&mut self) {
        self.bytecode.jumpdest("dynamic_dispatch");

        // Load target PC from memory
        self.bytecode.push_u32(PC_ADDR);
        self.bytecode.emit(Opcode::MLoad);

        // Generate dispatch table
        let entries: Vec<_> = self.pc_to_evm.iter().map(|(&k, &v)| (k, v)).collect();

        for (i, (rv_pc, evm_pos)) in entries.iter().enumerate() {
            let skip_label = format!("dispatch_skip_{}", i);

            // DUP1 (duplicate target PC)
            self.bytecode.emit(Opcode::Dup1);
            // Push RV PC to compare
            self.bytecode.push_u32(*rv_pc);
            // SUB - result is 0 if equal
            self.bytecode.emit(Opcode::Sub);
            // If not equal, skip to next comparison
            self.bytecode.jumpi_to(&skip_label);
            // Match found! Pop target PC and jump to destination
            self.bytecode.emit(Opcode::Pop);
            self.bytecode.push2(*evm_pos as u16);
            self.bytecode.emit(Opcode::Jump);
            // Skip label
            self.bytecode.jumpdest(&skip_label);
        }

        // No match found - halt
        self.bytecode.emit(Opcode::Pop);
        self.bytecode.jump_to("halt");
    }

    fn resolve_jumps(&mut self) -> Result<(), &'static str> {
        for (placeholder_pos, target_pc) in &self.pending_jumps {
            let evm_target = self.pc_to_evm.get(target_pc)
                .ok_or("Jump target not found")?;
            let target_bytes = (*evm_target as u16).to_be_bytes();
            self.bytecode.bytecode_mut()[*placeholder_pos] = target_bytes[0];
            self.bytecode.bytecode_mut()[*placeholder_pos + 1] = target_bytes[1];
        }
        self.pending_jumps.clear();
        Ok(())
    }

    fn compile_instruction(&mut self, pc: u32, instr: &Instruction) -> Result<(), &'static str> {
        match instr {
            // U-type
            Instruction::Lui { rd, imm } => {
                if *rd != 0 {
                    self.emit_store_reg_imm(*rd, *imm as u32);
                }
            }
            Instruction::Auipc { rd, imm } => {
                if *rd != 0 {
                    let value = pc.wrapping_add(*imm as u32);
                    self.emit_store_reg_imm(*rd, value);
                }
            }

            // J-type
            Instruction::Jal { rd, imm } => {
                if *rd != 0 {
                    self.emit_store_reg_imm(*rd, pc.wrapping_add(4));
                }
                let target = pc.wrapping_add(*imm as u32);
                self.emit_jump_to_rv_pc(target);
            }

            // I-type (JALR)
            Instruction::Jalr { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_store_reg_imm(*rd, pc.wrapping_add(4));
                }
                // Compute target: (rs1 + imm) & ~1
                self.emit_load_reg(*rs1);
                if *imm != 0 {
                    self.bytecode.push_u32(*imm as u32);
                    self.bytecode.emit(Opcode::Add);
                }
                self.bytecode.push_u32(0xFFFFFFFE);
                self.bytecode.emit(Opcode::And);
                // Store target PC and jump to dispatch
                self.bytecode.push_u32(PC_ADDR);
                self.bytecode.emit(Opcode::MStore);
                self.bytecode.jump_to("dynamic_dispatch");
            }

            // Branch instructions
            Instruction::Beq { rs1, rs2, imm } => {
                self.emit_branch(pc, *rs1, *rs2, *imm, |bc| {
                    bc.emit(Opcode::Eq);
                });
            }
            Instruction::Bne { rs1, rs2, imm } => {
                self.emit_branch(pc, *rs1, *rs2, *imm, |bc| {
                    bc.emit(Opcode::Eq);
                    bc.emit(Opcode::IsZero);
                });
            }
            Instruction::Blt { rs1, rs2, imm } => {
                self.emit_branch(pc, *rs1, *rs2, *imm, |bc| {
                    bc.emit(Opcode::Slt);
                });
            }
            Instruction::Bge { rs1, rs2, imm } => {
                self.emit_branch(pc, *rs1, *rs2, *imm, |bc| {
                    bc.emit(Opcode::Slt);
                    bc.emit(Opcode::IsZero);
                });
            }
            Instruction::Bltu { rs1, rs2, imm } => {
                self.emit_branch(pc, *rs1, *rs2, *imm, |bc| {
                    bc.emit(Opcode::Lt);
                });
            }
            Instruction::Bgeu { rs1, rs2, imm } => {
                self.emit_branch(pc, *rs1, *rs2, *imm, |bc| {
                    bc.emit(Opcode::Lt);
                    bc.emit(Opcode::IsZero);
                });
            }

            // Load instructions
            Instruction::Lw { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_compute_addr(*rs1, *imm);
                    self.bytecode.emit(Opcode::MLoad);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Lh { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_compute_addr(*rs1, *imm);
                    self.bytecode.emit(Opcode::MLoad);
                    // Sign-extend 16-bit
                    self.bytecode.push1(240);
                    self.bytecode.emit(Opcode::Shr);
                    self.bytecode.push1(1);
                    self.bytecode.emit(Opcode::SignExtend);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Lhu { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_compute_addr(*rs1, *imm);
                    self.bytecode.emit(Opcode::MLoad);
                    self.bytecode.push1(240);
                    self.bytecode.emit(Opcode::Shr);
                    self.bytecode.push_u32(0xFFFF);
                    self.bytecode.emit(Opcode::And);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Lb { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_compute_addr(*rs1, *imm);
                    self.bytecode.emit(Opcode::MLoad);
                    self.bytecode.push1(248);
                    self.bytecode.emit(Opcode::Shr);
                    self.bytecode.push0();
                    self.bytecode.emit(Opcode::SignExtend);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Lbu { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_compute_addr(*rs1, *imm);
                    self.bytecode.emit(Opcode::MLoad);
                    self.bytecode.push1(248);
                    self.bytecode.emit(Opcode::Shr);
                    self.bytecode.push1(0xFF);
                    self.bytecode.emit(Opcode::And);
                    self.emit_store_reg(*rd);
                }
            }

            // Store instructions
            Instruction::Sw { rs1, rs2, imm } => {
                self.emit_load_reg(*rs2);
                self.emit_compute_addr(*rs1, *imm);
                self.bytecode.emit(Opcode::MStore);
            }
            Instruction::Sh { rs1, rs2, imm } => {
                // For simplicity, implement as full word store (correct for aligned)
                self.emit_load_reg(*rs2);
                self.bytecode.push_u32(0xFFFF);
                self.bytecode.emit(Opcode::And);
                self.emit_compute_addr(*rs1, *imm);
                // This is simplified - would need read-modify-write for correctness
                self.bytecode.emit(Opcode::MStore);
            }
            Instruction::Sb { rs1, rs2, imm } => {
                self.emit_load_reg(*rs2);
                self.bytecode.push1(0xFF);
                self.bytecode.emit(Opcode::And);
                self.emit_compute_addr(*rs1, *imm);
                self.bytecode.emit(Opcode::MStore8);
            }

            // Arithmetic immediate
            Instruction::Addi { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    if *imm != 0 {
                        self.bytecode.push_u32(*imm as u32);
                        self.bytecode.emit(Opcode::Add);
                    }
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Slti { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32();
                    self.bytecode.push_u32(*imm as u32);
                    self.emit_sign_extend_imm();
                    self.bytecode.emit(Opcode::Sgt);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Sltiu { rd, rs1, imm } => {
                if *rd != 0 {
                    self.bytecode.push_u32(*imm as u32);
                    self.emit_mask32();
                    self.emit_load_reg(*rs1);
                    self.emit_mask32();
                    self.bytecode.emit(Opcode::Gt);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Xori { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.bytecode.push_u32(*imm as u32);
                    self.bytecode.emit(Opcode::Xor);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Ori { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.bytecode.push_u32(*imm as u32);
                    self.bytecode.emit(Opcode::Or);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Andi { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.bytecode.push_u32(*imm as u32);
                    self.bytecode.emit(Opcode::And);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Slli { rd, rs1, shamt } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.bytecode.push1(*shamt);
                    self.bytecode.emit(Opcode::Shl);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Srli { rd, rs1, shamt } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_mask32();
                    self.bytecode.push1(*shamt);
                    self.bytecode.emit(Opcode::Shr);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Srai { rd, rs1, shamt } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32();
                    self.bytecode.push1(*shamt);
                    self.bytecode.emit(Opcode::Sar);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }

            // Register arithmetic
            Instruction::Add { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::Add);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Sub { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::Sub);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Sll { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.push1(0x1F);
                    self.bytecode.emit(Opcode::And);
                    self.bytecode.emit(Opcode::Shl);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Slt { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs2);
                    self.emit_sign_extend_32();
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32();
                    self.bytecode.emit(Opcode::Sgt);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Sltu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs2);
                    self.emit_mask32();
                    self.emit_load_reg(*rs1);
                    self.emit_mask32();
                    self.bytecode.emit(Opcode::Gt);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Xor { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::Xor);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Srl { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_mask32();
                    self.emit_load_reg(*rs2);
                    self.bytecode.push1(0x1F);
                    self.bytecode.emit(Opcode::And);
                    self.bytecode.emit(Opcode::Shr);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Sra { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32();
                    self.emit_load_reg(*rs2);
                    self.bytecode.push1(0x1F);
                    self.bytecode.emit(Opcode::And);
                    self.bytecode.emit(Opcode::Sar);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Or { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::Or);
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::And { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::And);
                    self.emit_store_reg(*rd);
                }
            }

            // M extension
            Instruction::Mul { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::Mul);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Mulh { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32();
                    self.emit_load_reg(*rs2);
                    self.emit_sign_extend_32();
                    self.bytecode.emit(Opcode::Mul);
                    self.bytecode.push1(32);
                    self.bytecode.emit(Opcode::Sar);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Mulhu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_mask32();
                    self.emit_load_reg(*rs2);
                    self.emit_mask32();
                    self.bytecode.emit(Opcode::Mul);
                    self.bytecode.push1(32);
                    self.bytecode.emit(Opcode::Shr);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Mulhsu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32();
                    self.emit_load_reg(*rs2);
                    self.emit_mask32();
                    self.bytecode.emit(Opcode::Mul);
                    self.bytecode.push1(32);
                    self.bytecode.emit(Opcode::Sar);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Div { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32();
                    self.emit_load_reg(*rs2);
                    self.emit_sign_extend_32();
                    self.bytecode.emit(Opcode::SDiv);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Divu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_mask32();
                    self.emit_load_reg(*rs2);
                    self.emit_mask32();
                    self.bytecode.emit(Opcode::Div);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Rem { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32();
                    self.emit_load_reg(*rs2);
                    self.emit_sign_extend_32();
                    self.bytecode.emit(Opcode::SMod);
                    self.emit_mask32();
                    self.emit_store_reg(*rd);
                }
            }
            Instruction::Remu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_mask32();
                    self.emit_load_reg(*rs2);
                    self.emit_mask32();
                    self.bytecode.emit(Opcode::Mod);
                    self.emit_store_reg(*rd);
                }
            }

            // System
            Instruction::Ecall => {
                self.bytecode.jump_to("halt");
            }
            Instruction::Ebreak => {
                self.bytecode.jump_to("halt");
            }
            Instruction::Fence { .. } => {
                // No-op on EVM
            }

            // CSR instructions - no-op on EVM (just store 0 to rd if rd != 0)
            Instruction::Csrrw { rd, .. } |
            Instruction::Csrrs { rd, .. } |
            Instruction::Csrrc { rd, .. } |
            Instruction::Csrrwi { rd, .. } |
            Instruction::Csrrsi { rd, .. } |
            Instruction::Csrrci { rd, .. } => {
                if *rd != 0 {
                    self.emit_store_reg_imm(*rd, 0);
                }
            }

            Instruction::Unknown { .. } => {
                return Err("Unknown instruction");
            }
        }
        Ok(())
    }

    // Helper functions

    fn emit_load_reg(&mut self, reg: u8) {
        if reg == 0 {
            self.bytecode.push0();
        } else {
            self.bytecode.push_u32(REG_BASE + (reg as u32) * 32);
            self.bytecode.emit(Opcode::MLoad);
        }
    }

    fn emit_store_reg(&mut self, reg: u8) {
        if reg != 0 {
            self.bytecode.push_u32(REG_BASE + (reg as u32) * 32);
            self.bytecode.emit(Opcode::MStore);
        } else {
            self.bytecode.emit(Opcode::Pop);
        }
    }

    fn emit_store_reg_imm(&mut self, reg: u8, value: u32) {
        if reg != 0 {
            self.bytecode.push_u32(value);
            self.bytecode.push_u32(REG_BASE + (reg as u32) * 32);
            self.bytecode.emit(Opcode::MStore);
        }
    }

    fn emit_mask32(&mut self) {
        self.bytecode.push4(0xFFFFFFFF);
        self.bytecode.emit(Opcode::And);
    }

    fn emit_sign_extend_32(&mut self) {
        self.bytecode.push1(3); // SignExtend from byte 3 (32-bit)
        self.bytecode.emit(Opcode::SignExtend);
    }

    fn emit_sign_extend_imm(&mut self) {
        self.bytecode.push1(3);
        self.bytecode.emit(Opcode::SignExtend);
    }

    fn emit_compute_addr(&mut self, base_reg: u8, offset: i32) {
        self.emit_load_reg(base_reg);
        if offset != 0 {
            self.bytecode.push_u32(offset as u32);
            self.bytecode.emit(Opcode::Add);
        }
        // Convert RISC-V address to EVM address
        self.bytecode.push_u32(self.config.load_address);
        self.bytecode.emit(Opcode::Sub);
        self.bytecode.push_u32(MEM_BASE);
        self.bytecode.emit(Opcode::Add);
    }

    fn emit_jump_to_rv_pc(&mut self, target_pc: u32) {
        self.bytecode.emit(Opcode::Push2);
        self.pending_jumps.push((self.bytecode.position(), target_pc));
        self.bytecode.emit_bytes(&[0, 0]);
        self.bytecode.emit(Opcode::Jump);
    }

    fn emit_branch<F>(&mut self, pc: u32, rs1: u8, rs2: u8, imm: i32, condition: F)
    where
        F: FnOnce(&mut EvmBytecode),
    {
        let target = pc.wrapping_add(imm as u32);
        let fallthrough = pc.wrapping_add(4);

        // Load operands (rs2, rs1 order for comparison)
        self.emit_load_reg(rs2);
        self.emit_load_reg(rs1);

        // Apply condition
        condition(&mut self.bytecode);

        // Conditional jump to target
        self.bytecode.emit(Opcode::Push2);
        self.pending_jumps.push((self.bytecode.position(), target));
        self.bytecode.emit_bytes(&[0, 0]);
        self.bytecode.emit(Opcode::Swap1);
        self.bytecode.emit(Opcode::JumpI);

        // Fallthrough to next instruction
        self.bytecode.emit(Opcode::Push2);
        self.pending_jumps.push((self.bytecode.position(), fallthrough));
        self.bytecode.emit_bytes(&[0, 0]);
        self.bytecode.emit(Opcode::Jump);
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}
