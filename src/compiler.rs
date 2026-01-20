//! RISC-V to EVM compiler
//!
//! Compiles RISC-V rv32im instructions into EVM bytecode.
//!
//! # Memory Layout in EVM
//!
//! The EVM memory is laid out as follows:
//! - 0x0000-0x03FF: RISC-V registers (x0-x31, 32 bytes each = 1024 bytes)
//! - 0x0400-0x041F: Program counter (PC)
//! - 0x0420-0x043F: Return value storage
//! - 0x0500+: RISC-V memory
//!
//! All values are stored in big-endian format in the high bits of 32-byte EVM words.
//!
//! # Stack Model and DUP Optimization
//!
//! The compiler maintains a stack model that tracks what register values are
//! currently on the EVM stack. When a register value is needed and it's already
//! on the stack (within DUP1-DUP16 range), we use DUP instead of MLOAD.
//!
//! ## Register Caching (Experimental)
//!
//! The compiler supports optional register caching within basic blocks, controlled
//! by `CompilerConfig::enable_register_caching`. When enabled, register values
//! are DUP'd before MSTORE so they remain on the stack for potential reuse.
//!
//! **Currently disabled by default** because the cleanup overhead at branch
//! points (SWAP + POPs) typically exceeds the savings in typical loop patterns.
//! For a loop with N stores and 1 cache hit before branch:
//! - Extra costs: N×DUP(3) + SWAP(3) + N×POP(2) = 5N+3 gas
//! - Savings: ~9 gas per cache hit
//! - Breakeven: Need 5N+3 < 9×hits, rarely achieved in practice
//!
//! The optimization may benefit longer straight-line code sequences with
//! multiple register reuses, but typical RISC-V loops don't see improvement.

use crate::decoder::{decode_instruction, Instruction};
use crate::evm::{EvmBytecode, Opcode};
use std::collections::HashMap;

/// Stack entry type for tracking what's on the EVM stack
#[derive(Debug, Clone, Copy, PartialEq)]
enum StackEntry {
    /// A RISC-V register value (0-31)
    Register(u8),
    /// Some computed/unknown value
    Unknown,
}

/// Stack model for tracking EVM stack contents
///
/// This enables cross-instruction register caching by tracking what register
/// values are currently on the stack. When a register needs to be loaded,
/// we can use DUP if the value is already on the stack (depth 1-16).
///
/// The model is cleared at basic block boundaries (JUMPDEST) since we cannot
/// know the stack state when jumping from other locations.
#[derive(Debug, Clone)]
struct StackModel {
    /// Stack entries, index 0 = bottom, last = top
    entries: Vec<StackEntry>,
    /// Maximum tracked depth (EVM DUP1-DUP16)
    max_depth: usize,
}

impl StackModel {
    fn new() -> Self {
        Self {
            entries: Vec::with_capacity(32),
            max_depth: 16,
        }
    }

    /// Push a register value onto the stack
    fn push_register(&mut self, reg: u8) {
        self.entries.push(StackEntry::Register(reg));
        self.trim();
    }

    /// Push an unknown value onto the stack
    fn push_unknown(&mut self) {
        self.entries.push(StackEntry::Unknown);
        self.trim();
    }

    /// Pop the top value from the stack
    fn pop(&mut self) -> Option<StackEntry> {
        self.entries.pop()
    }

    /// Pop n values from the stack
    fn pop_n(&mut self, n: usize) {
        for _ in 0..n {
            self.entries.pop();
        }
    }

    /// Binary operation: consumes 2 values, produces 1 unknown
    fn binary_op(&mut self) {
        self.pop_n(2);
        self.push_unknown();
    }

    /// Unary operation: consumes 1 value, produces 1 unknown
    fn unary_op(&mut self) {
        self.pop();
        self.push_unknown();
    }

    /// DUP operation: copies value at depth to top
    /// depth is 1-indexed (DUP1 copies top, DUP2 copies second from top)
    fn dup(&mut self, depth: usize) {
        if depth == 0 || depth > self.entries.len() {
            self.push_unknown();
            return;
        }
        let idx = self.entries.len() - depth;
        let entry = self.entries[idx];
        self.entries.push(entry);
        self.trim();
    }

    /// SWAP operation: exchanges top with element at depth
    /// depth is 1-indexed (SWAP1 exchanges top with second)
    fn swap(&mut self, depth: usize) {
        if depth == 0 || depth >= self.entries.len() {
            return;
        }
        let len = self.entries.len();
        self.entries.swap(len - 1, len - 1 - depth);
    }

    /// Find a register on the stack and return its depth (1 = top)
    /// Returns None if not found or depth > 16
    fn find_register(&self, reg: u8) -> Option<usize> {
        if reg == 0 {
            // x0 is always 0, handled separately
            return None;
        }
        for (i, entry) in self.entries.iter().rev().enumerate() {
            let depth = i + 1;
            if depth > self.max_depth {
                break;
            }
            if *entry == StackEntry::Register(reg) {
                return Some(depth);
            }
        }
        None
    }

    /// Invalidate a specific register (when it's written to memory)
    fn invalidate_register(&mut self, reg: u8) {
        for entry in &mut self.entries {
            if *entry == StackEntry::Register(reg) {
                *entry = StackEntry::Unknown;
            }
        }
    }

    /// Clear the entire stack model (at basic block boundaries)
    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Trim stack to max tracked depth
    fn trim(&mut self) {
        // Keep slightly more than max_depth for better tracking
        let max_track = self.max_depth + 8;
        if self.entries.len() > max_track {
            let remove = self.entries.len() - max_track;
            self.entries.drain(0..remove);
        }
    }

    /// Get current stack depth
    #[allow(dead_code)]
    fn depth(&self) -> usize {
        self.entries.len()
    }
}

/// Memory layout constants
/// Each register gets a full 32-byte EVM word to avoid overlap issues
pub const REG_BASE: u32 = 0x0000;     // Base address for registers
pub const REG_SIZE: u32 = 32;         // 32 bytes per register (EVM word size)
pub const PC_ADDR: u32 = 0x0400;      // Program counter address (after 32 registers * 32 bytes)
pub const RETVAL_ADDR: u32 = 0x0420;  // Return value storage
pub const MEM_BASE: u32 = 0x0500;     // Base address for RISC-V memory

/// Compiler configuration
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Base address where the RISC-V program is loaded
    pub load_address: u32,
    /// Stack pointer initial value
    pub stack_pointer: u32,
    /// Memory size in bytes
    pub memory_size: u32,
    /// Enable basic block register caching
    /// When true, caches register values on the EVM stack for potential reuse.
    /// Currently disabled by default as the cleanup overhead at branch points
    /// exceeds the savings in typical loop patterns.
    pub enable_register_caching: bool,
    /// Initial memory contents: (RISC-V address, data bytes)
    /// These are written to memory during initialization before program execution.
    pub initial_memory: Vec<(u32, Vec<u8>)>,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x20000, // 128KB
            enable_register_caching: false, // Disabled: overhead exceeds savings
            initial_memory: Vec::new(),
        }
    }
}

/// Compiler for RISC-V to EVM translation
pub struct Compiler {
    config: CompilerConfig,
    bytecode: EvmBytecode,
    /// Maps RISC-V PC to EVM bytecode offset
    pc_to_evm: HashMap<u32, usize>,
    /// Maps EVM placeholder positions to RISC-V PC targets
    pending_jumps: Vec<(usize, u32)>,
    /// Stack model for cross-instruction DUP optimization
    stack: StackModel,
    /// Number of cached register values currently on the EVM stack
    /// These need to be cleaned up at basic block boundaries
    cached_count: usize,
}

impl Compiler {
    /// Create a new compiler with default configuration
    pub fn new() -> Self {
        Self::with_config(CompilerConfig::default())
    }

    /// Create a new compiler with custom configuration
    pub fn with_config(config: CompilerConfig) -> Self {
        Self {
            config,
            bytecode: EvmBytecode::new(),
            pc_to_evm: HashMap::new(),
            pending_jumps: Vec::new(),
            stack: StackModel::new(),
            cached_count: 0,
        }
    }

    /// Compile a RISC-V program to EVM bytecode
    ///
    /// The input is raw RISC-V machine code bytes.
    /// Returns EVM bytecode that can be executed.
    pub fn compile(&mut self, program: &[u8]) -> Result<Vec<u8>, String> {
        // First pass: decode all instructions and identify jump targets
        let instructions: Vec<(u32, Instruction)> = program
            .chunks_exact(4)
            .enumerate()
            .map(|(i, chunk)| {
                let raw = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let pc = self.config.load_address + (i as u32 * 4);
                (pc, decode_instruction(raw))
            })
            .collect();

        // Identify all jump targets
        let mut jump_targets: std::collections::HashSet<u32> = std::collections::HashSet::new();
        // First instruction is always a target (entry point)
        if let Some((first_pc, _)) = instructions.first() {
            jump_targets.insert(*first_pc);
        }
        // Scan for branch/jump targets
        for (pc, instr) in &instructions {
            match instr {
                Instruction::Jal { imm, .. } => {
                    jump_targets.insert(pc.wrapping_add(*imm as u32));
                }
                Instruction::Beq { imm, .. }
                | Instruction::Bne { imm, .. }
                | Instruction::Blt { imm, .. }
                | Instruction::Bge { imm, .. }
                | Instruction::Bltu { imm, .. }
                | Instruction::Bgeu { imm, .. } => {
                    jump_targets.insert(pc.wrapping_add(*imm as u32));
                }
                Instruction::Jalr { .. } => {
                    // Dynamic jump - all instructions are potential targets
                    // For now, mark all as targets (conservative)
                    for (target_pc, _) in &instructions {
                        jump_targets.insert(*target_pc);
                    }
                }
                _ => {}
            }
        }

        // Emit initialization code
        self.emit_init();

        // Second pass: compile each instruction
        for (pc, instr) in &instructions {
            // Only emit JUMPDEST for actual jump targets
            let is_jump_target = jump_targets.contains(pc);
            if is_jump_target {
                // Clean up any cached values from previous basic block
                // (this happens BEFORE the JUMPDEST for the fallthrough path)
                self.cleanup_cached_values();
                // Record the EVM position BEFORE emitting JUMPDEST
                // This ensures jump targets point to the JUMPDEST instruction
                self.pc_to_evm.insert(*pc, self.bytecode.position());
                self.bytecode.emit(Opcode::JumpDest);
                // Clear stack model at basic block boundaries
                // We can't know stack state when jumping from other locations
                self.stack.clear();
            } else {
                // Not a jump target - record position normally
                self.pc_to_evm.insert(*pc, self.bytecode.position());
            }

            // Compile the instruction
            // Stack model carries over within basic blocks for cross-instruction caching
            self.compile_instruction(*pc, instr)?;
        }

        // Emit halt/return code
        self.emit_halt();

        // Resolve all pending jumps
        self.resolve_jumps()?;

        // Finalize and return
        self.bytecode.resolve()?;
        Ok(self.bytecode.bytecode().to_vec())
    }

    /// Emit initialization code
    fn emit_init(&mut self) {
        // Initialize x0 (zero) to 0
        self.emit_store_reg_imm(0, 0);

        // Initialize stack pointer (x2/sp)
        self.emit_store_reg_imm(2, self.config.stack_pointer);

        // Initialize memory from config
        // Clone to avoid borrow issues
        let initial_memory = self.config.initial_memory.clone();
        for (rv_addr, data) in initial_memory {
            self.emit_memory_init(rv_addr, &data);
        }

        // Jump to the main execution loop
        self.bytecode.jump_to("main_entry");
        self.bytecode.jumpdest("main_entry");
    }

    /// Emit code to initialize a memory region
    /// Writes data bytes to RISC-V memory starting at rv_addr
    fn emit_memory_init(&mut self, rv_addr: u32, data: &[u8]) {
        // Convert RISC-V address to EVM address
        let base_evm_addr = MEM_BASE + rv_addr.wrapping_sub(self.config.load_address);

        // Write 32 bytes at a time (EVM word size)
        let mut offset = 0u32;
        while offset < data.len() as u32 {
            let mut word = [0u8; 32];
            let remaining = (data.len() as u32 - offset) as usize;
            let chunk_size = remaining.min(32);

            // Copy bytes to the beginning of the word (big-endian storage)
            word[..chunk_size].copy_from_slice(&data[offset as usize..offset as usize + chunk_size]);

            // Push the 32-byte value
            self.bytecode.emit(Opcode::Push32);
            self.bytecode.emit_bytes(&word);

            // Push the EVM address
            self.bytecode.push_u32(base_evm_addr + offset);

            // MSTORE
            self.bytecode.emit(Opcode::MStore);

            offset += 32;
        }
    }

    /// Emit halt code (returns the value in a0/x10)
    fn emit_halt(&mut self) {
        self.bytecode.jumpdest("halt");

        // Load a0 (x10) - the return value
        self.emit_load_reg(10);

        // Store at return position in memory for RETURN
        self.bytecode.push_u32(RETVAL_ADDR);
        self.bytecode.emit(Opcode::MStore);

        // RETURN(offset=RETVAL_ADDR, size=32)
        self.bytecode.push1(32);
        self.bytecode.push_u32(RETVAL_ADDR);
        self.bytecode.emit(Opcode::Return);
    }

    /// Compile a single instruction
    fn compile_instruction(&mut self, pc: u32, instr: &Instruction) -> Result<(), String> {
        match instr {
            // U-type instructions
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

            // Jump instructions
            Instruction::Jal { rd, imm } => {
                // Store return address (pc + 4) in rd
                if *rd != 0 {
                    self.emit_store_reg_imm(*rd, pc.wrapping_add(4));
                }
                // Jump to target
                let target = pc.wrapping_add(*imm as u32);
                self.emit_jump_to_rv_pc(target);
            }

            Instruction::Jalr { rd, rs1, imm } => {
                // Store return address in rd (if rd != 0)
                if *rd != 0 {
                    self.emit_store_reg_imm(*rd, pc.wrapping_add(4));
                }
                // Compute target: (rs1 + imm) & ~1
                self.emit_load_reg(*rs1);
                self.t_push_u32(*imm as u32);
                self.t_binary_op(Opcode::Add);
                self.t_push_u32(0xFFFFFFFE);
                self.t_binary_op(Opcode::And);
                // Dynamic jump dispatch
                self.emit_dynamic_jump();
            }

            // Branch instructions
            // Load operands first (may benefit from cache), then cleanup, then branch
            Instruction::Beq { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                self.emit_load_reg_pair(*rs1, *rs2);
                self.t_binary_op(Opcode::Eq);
                self.emit_branch_with_cleanup(target, fallthrough);
            }

            Instruction::Bne { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                self.emit_load_reg_pair(*rs1, *rs2);
                self.t_binary_op(Opcode::Eq);
                self.t_unary_op(Opcode::IsZero); // NOT equal
                self.emit_branch_with_cleanup(target, fallthrough);
            }

            Instruction::Blt { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Signed comparison: rs1 < rs2
                self.emit_signed_lt(*rs1, *rs2);
                self.emit_branch_with_cleanup(target, fallthrough);
            }

            Instruction::Bge { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Signed comparison: rs1 >= rs2 (i.e., NOT rs1 < rs2)
                self.emit_signed_lt(*rs1, *rs2);
                self.t_unary_op(Opcode::IsZero);
                self.emit_branch_with_cleanup(target, fallthrough);
            }

            Instruction::Bltu { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Load rs2 first (bottom), then rs1 (top)
                self.emit_load_reg_pair(*rs2, *rs1);
                // LT: returns 1 if s[0] < s[1], i.e., rs1 < rs2
                self.t_binary_op(Opcode::Lt);
                self.emit_branch_with_cleanup(target, fallthrough);
            }

            Instruction::Bgeu { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Load rs2 first (bottom), then rs1 (top)
                self.emit_load_reg_pair(*rs2, *rs1);
                // LT: returns 1 if rs1 < rs2
                self.t_binary_op(Opcode::Lt);
                self.t_unary_op(Opcode::IsZero); // NOT less than = >=
                self.emit_branch_with_cleanup(target, fallthrough);
            }

            // Load instructions
            Instruction::Lb { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_byte_signed(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Lh { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_half_signed(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Lw { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_word(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Lbu { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_byte_unsigned(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Lhu { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_half_unsigned(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            // Store instructions
            Instruction::Sb { rs1, rs2, imm } => {
                self.emit_store_byte(*rs1, *rs2, *imm);
            }

            Instruction::Sh { rs1, rs2, imm } => {
                self.emit_store_half(*rs1, *rs2, *imm);
            }

            Instruction::Sw { rs1, rs2, imm } => {
                self.emit_store_word(*rs1, *rs2, *imm);
            }

            // Arithmetic immediate instructions
            Instruction::Addi { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.t_push_u32(*imm as u32);
                    self.t_binary_op(Opcode::Add);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Slti { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_signed_lt_imm(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sltiu { rd, rs1, imm } => {
                if *rd != 0 {
                    // Push imm first (bottom), then rs1 (top)
                    self.t_push_u32(*imm as u32);
                    self.emit_load_reg(*rs1);
                    // LT: returns 1 if s[0] < s[1], i.e., rs1 < imm
                    self.t_binary_op(Opcode::Lt);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Xori { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.t_push_u32(*imm as u32);
                    self.t_binary_op(Opcode::Xor);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Ori { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.t_push_u32(*imm as u32);
                    self.t_binary_op(Opcode::Or);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Andi { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.t_push_u32(*imm as u32);
                    self.t_binary_op(Opcode::And);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Slli { rd, rs1, shamt } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.t_push1(*shamt);
                    self.t_binary_op(Opcode::Shl);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Srli { rd, rs1, shamt } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.t_push1(*shamt);
                    self.t_binary_op(Opcode::Shr);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Srai { rd, rs1, shamt } => {
                if *rd != 0 {
                    // Sign-extend to 256 bits, then arithmetic shift right
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32_to_256();
                    self.t_push1(*shamt);
                    self.t_binary_op(Opcode::Sar);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            // Register arithmetic instructions
            Instruction::Add { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Add);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sub { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Load rs2 first (bottom), then rs1 (top)
                    // EVM SUB: s[0] - s[1] = rs1 - rs2
                    self.emit_load_reg_pair(*rs2, *rs1);
                    self.t_binary_op(Opcode::Sub);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sll { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_push1(0x1F);
                    self.t_binary_op(Opcode::And); // Only low 5 bits of rs2
                    self.t_binary_op(Opcode::Shl);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Slt { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_signed_lt(*rs1, *rs2);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sltu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Load rs2 first (bottom), then rs1 (top)
                    self.emit_load_reg_pair(*rs2, *rs1);
                    // LT: returns 1 if s[0] < s[1], i.e., rs1 < rs2
                    self.t_binary_op(Opcode::Lt);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Xor { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Xor);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Srl { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_push1(0x1F);
                    self.t_binary_op(Opcode::And);
                    self.t_binary_op(Opcode::Shr);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sra { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32_to_256();
                    self.emit_load_reg(*rs2);
                    self.t_push1(0x1F);
                    self.t_binary_op(Opcode::And);
                    self.t_binary_op(Opcode::Sar);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Or { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Or);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::And { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::And);
                    self.emit_store_reg(*rd);
                }
            }

            // System instructions
            Instruction::Fence { .. } => {
                // No-op in EVM (single-threaded)
            }

            Instruction::Ecall => {
                // For testing, ECALL triggers program termination
                // Return value is in a0 (x10)
                self.cleanup_cached_values();
                self.bytecode.jump_to("halt");
                self.stack.clear();
            }

            Instruction::Ebreak => {
                // For debugging, treat as halt
                self.cleanup_cached_values();
                self.bytecode.jump_to("halt");
                self.stack.clear();
            }

            // M extension (multiply/divide)
            Instruction::Mul { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Mul);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Mulh { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Signed * Signed, return high 32 bits
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32_to_256();
                    self.emit_load_reg(*rs2);
                    self.emit_sign_extend_32_to_256();
                    self.t_binary_op(Opcode::Mul);
                    self.t_push1(32);
                    self.t_binary_op(Opcode::Sar); // Arithmetic shift for signed
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Mulhsu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Signed * Unsigned, return high 32 bits
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32_to_256();
                    self.emit_load_reg(*rs2);
                    // rs2 is unsigned, no sign extension needed
                    self.t_binary_op(Opcode::Mul);
                    self.t_push1(32);
                    self.t_binary_op(Opcode::Sar);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Mulhu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Unsigned * Unsigned, return high 32 bits
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Mul);
                    self.t_push1(32);
                    self.t_binary_op(Opcode::Shr);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Div { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Signed division
                    self.emit_signed_div(*rs1, *rs2);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Divu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Unsigned division
                    self.emit_unsigned_div(*rs1, *rs2);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Rem { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Signed remainder
                    self.emit_signed_rem(*rs1, *rs2);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Remu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Unsigned remainder
                    self.emit_unsigned_rem(*rs1, *rs2);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Unknown { raw } => {
                return Err(format!("Unknown instruction at PC {:#x}: {:#010x}", pc, raw));
            }
        }

        Ok(())
    }

    // ============================================================
    // Cache management
    // ============================================================

    /// Clean up all cached register values from the EVM stack
    /// Called at basic block boundaries (before JUMPDEST)
    fn cleanup_cached_values(&mut self) {
        for _ in 0..self.cached_count {
            self.bytecode.emit(Opcode::Pop);
            self.stack.pop();
        }
        self.cached_count = 0;
    }

    // ============================================================
    // Tracked bytecode emission methods
    // ============================================================
    // These methods emit bytecode AND update the stack model

    /// Push 0 onto the stack (tracked)
    fn t_push0(&mut self) {
        self.bytecode.push0();
        self.stack.push_unknown();
    }

    /// Push a 1-byte value onto the stack (tracked)
    fn t_push1(&mut self, value: u8) {
        self.bytecode.push1(value);
        self.stack.push_unknown();
    }

    /// Push a 2-byte value onto the stack (tracked)
    fn t_push2(&mut self, value: u16) {
        self.bytecode.push2(value);
        self.stack.push_unknown();
    }

    /// Push a 4-byte value onto the stack (tracked)
    fn t_push4(&mut self, value: u32) {
        self.bytecode.push4(value);
        self.stack.push_unknown();
    }

    /// Push a u32 value onto the stack (tracked)
    fn t_push_u32(&mut self, value: u32) {
        self.bytecode.push_u32(value);
        self.stack.push_unknown();
    }

    /// Push a u256 value onto the stack (tracked)
    fn t_push_u256(&mut self, bytes: &[u8; 32]) {
        self.bytecode.push_u256(bytes);
        self.stack.push_unknown();
    }

    /// Emit a binary operation (consumes 2, produces 1 unknown)
    fn t_binary_op(&mut self, opcode: Opcode) {
        self.bytecode.emit(opcode);
        self.stack.binary_op();
    }

    /// Emit a unary operation (consumes 1, produces 1 unknown)
    fn t_unary_op(&mut self, opcode: Opcode) {
        self.bytecode.emit(opcode);
        self.stack.unary_op();
    }

    /// Emit MLOAD (consumes address, produces value)
    fn t_mload(&mut self) {
        self.bytecode.emit(Opcode::MLoad);
        self.stack.pop();
        self.stack.push_unknown();
    }

    /// Emit MSTORE (consumes value and address)
    fn t_mstore(&mut self) {
        self.bytecode.emit(Opcode::MStore);
        self.stack.pop_n(2);
    }

    /// Emit MSTORE8 (consumes value and address)
    fn t_mstore8(&mut self) {
        self.bytecode.emit(Opcode::MStore8);
        self.stack.pop_n(2);
    }

    /// Emit a DUP opcode for the given depth (1-16) and track it
    fn t_dup(&mut self, depth: usize) {
        let opcode = match depth {
            1 => Opcode::Dup1,
            2 => Opcode::Dup2,
            3 => Opcode::Dup3,
            4 => Opcode::Dup4,
            5 => Opcode::Dup5,
            6 => Opcode::Dup6,
            7 => Opcode::Dup7,
            8 => Opcode::Dup8,
            9 => Opcode::Dup9,
            10 => Opcode::Dup10,
            11 => Opcode::Dup11,
            12 => Opcode::Dup12,
            13 => Opcode::Dup13,
            14 => Opcode::Dup14,
            15 => Opcode::Dup15,
            16 => Opcode::Dup16,
            _ => panic!("Invalid DUP depth: {}", depth),
        };
        self.bytecode.emit(opcode);
        self.stack.dup(depth);
    }

    /// Emit a SWAP opcode for the given depth (1-16) and track it
    fn t_swap(&mut self, depth: usize) {
        let opcode = match depth {
            1 => Opcode::Swap1,
            2 => Opcode::Swap2,
            3 => Opcode::Swap3,
            4 => Opcode::Swap4,
            5 => Opcode::Swap5,
            6 => Opcode::Swap6,
            7 => Opcode::Swap7,
            8 => Opcode::Swap8,
            9 => Opcode::Swap9,
            10 => Opcode::Swap10,
            11 => Opcode::Swap11,
            12 => Opcode::Swap12,
            13 => Opcode::Swap13,
            14 => Opcode::Swap14,
            15 => Opcode::Swap15,
            16 => Opcode::Swap16,
            _ => panic!("Invalid SWAP depth: {}", depth),
        };
        self.bytecode.emit(opcode);
        self.stack.swap(depth);
    }

    /// Emit POP and track it
    fn t_pop(&mut self) {
        self.bytecode.emit(Opcode::Pop);
        self.stack.pop();
    }

    // ============================================================
    // Helper methods for register access
    // ============================================================

    /// Load a register value onto the EVM stack
    /// Uses DUP when the register value is already on the stack
    fn emit_load_reg(&mut self, reg: u8) {
        if reg == 0 {
            // x0 is always 0
            self.t_push0();
        } else if let Some(depth) = self.stack.find_register(reg) {
            // Register is already on the stack - use DUP instead of MLOAD
            self.t_dup(depth);
        } else {
            // Load from memory and track as register value
            self.emit_load_reg_from_memory(reg);
        }
    }

    /// Load a register value from memory (always loads, no cache check)
    fn emit_load_reg_from_memory(&mut self, reg: u8) {
        let addr = REG_BASE + (reg as u32) * REG_SIZE;
        self.bytecode.push_u32(addr);
        self.bytecode.emit(Opcode::MLoad);
        // Update stack: address popped, value pushed
        self.stack.pop();
        // Value is in the high bits, shift right to get it
        self.bytecode.push1(224); // 256 - 32 = 224
        self.bytecode.emit(Opcode::Shr);
        // Update stack: binary op (shift)
        self.stack.binary_op();
        // Mark the result as the register value
        // We need to manually set the top entry to Register(reg)
        self.stack.pop();
        self.stack.push_register(reg);
    }

    /// Load two registers onto the stack (rs1 first/bottom, rs2 second/top)
    /// Uses DUP optimization when possible
    fn emit_load_reg_pair(&mut self, rs1: u8, rs2: u8) {
        self.emit_load_reg(rs1);
        self.emit_load_reg(rs2);
    }

    /// Store the top of stack value into a register
    ///
    /// When register caching is enabled, DUPs the value before storing so it
    /// remains on the stack for potential reuse within the same basic block.
    fn emit_store_reg(&mut self, reg: u8) {
        if reg == 0 {
            // Writing to x0 is a no-op, just pop the value
            self.t_pop();
        } else {
            // Invalidate any old cached copies of this register
            self.stack.invalidate_register(reg);

            if self.config.enable_register_caching {
                // Basic block caching: mark the value as this register and DUP it
                // so a copy remains on the stack for potential reuse
                self.stack.pop();  // Remove the Unknown entry
                self.stack.push_register(reg);  // Mark as Register(reg)

                // DUP the value - keeps the original for reuse
                self.t_dup(1);
                self.cached_count += 1;

                // Shift the top copy and store it
                self.t_push1(224);
                self.t_binary_op(Opcode::Shl);
                let addr = REG_BASE + (reg as u32) * REG_SIZE;
                self.t_push_u32(addr);
                self.t_mstore();
                // After MSTORE, the original Register(reg) value remains on stack
            } else {
                // Standard behavior: just store the value
                self.t_push1(224);
                self.t_binary_op(Opcode::Shl);
                let addr = REG_BASE + (reg as u32) * REG_SIZE;
                self.t_push_u32(addr);
                self.t_mstore();
            }
        }
    }

    /// Store an immediate value into a register
    fn emit_store_reg_imm(&mut self, reg: u8, value: u32) {
        // Invalidate any cached copies of this register
        self.stack.invalidate_register(reg);
        if reg != 0 {
            self.t_push_u32(value);
            self.t_push1(224);
            self.t_binary_op(Opcode::Shl);
            let addr = REG_BASE + (reg as u32) * REG_SIZE;
            self.t_push_u32(addr);
            self.t_mstore();
        }
    }

    // ============================================================
    // Memory access helpers
    // ============================================================

    /// Compute RISC-V memory address to EVM address
    /// Stack effect: consumes addr, produces EVM addr
    fn emit_rv_to_evm_addr(&mut self) {
        // EVM addr = (rv_addr - load_address) + MEM_BASE
        // Stack: [rv_addr]
        let load_addr = self.config.load_address;
        self.t_push_u32(load_addr);
        // Stack: [rv_addr, load_address] with load_address on top
        self.t_swap(1);
        // Stack: [load_address, rv_addr] with rv_addr on top
        // EVM SUB: s[0] - s[1] = rv_addr - load_address
        self.t_binary_op(Opcode::Sub);
        self.t_push_u32(MEM_BASE);
        self.t_binary_op(Opcode::Add);
    }

    /// Helper to compute EVM address from base register and offset, leaves address on stack
    fn emit_compute_evm_addr(&mut self, base_reg: u8, offset: i32) {
        self.emit_load_reg(base_reg);
        self.t_push_u32(offset as u32);
        self.t_binary_op(Opcode::Add);
        self.emit_rv_to_evm_addr();
    }

    /// Load a byte (signed) from memory
    fn emit_load_byte_signed(&mut self, base_reg: u8, offset: i32) {
        self.emit_compute_evm_addr(base_reg, offset);
        self.t_mload();
        self.t_push1(248); // 256 - 8
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        // Sign extend from 8 bits
        self.emit_sign_extend_8_to_32();
    }

    /// Load a byte (unsigned) from memory
    fn emit_load_byte_unsigned(&mut self, base_reg: u8, offset: i32) {
        self.emit_compute_evm_addr(base_reg, offset);
        self.t_mload();
        self.t_push1(248);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
    }

    /// Load a halfword (signed) from little-endian memory
    fn emit_load_half_signed(&mut self, base_reg: u8, offset: i32) {
        // Load low byte
        self.emit_compute_evm_addr(base_reg, offset);
        self.t_mload();
        self.t_push1(248);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);

        // Load high byte
        self.emit_compute_evm_addr(base_reg, offset.wrapping_add(1));
        self.t_mload();
        self.t_push1(248);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);

        // Combine: high_byte << 8 | low_byte
        self.t_push1(8);
        self.t_binary_op(Opcode::Shl);
        self.t_binary_op(Opcode::Or);

        // Sign extend from 16 bits
        self.emit_sign_extend_16_to_32();
    }

    /// Load a halfword (unsigned) from little-endian memory
    fn emit_load_half_unsigned(&mut self, base_reg: u8, offset: i32) {
        // Load low byte
        self.emit_compute_evm_addr(base_reg, offset);
        self.t_mload();
        self.t_push1(248);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);

        // Load high byte
        self.emit_compute_evm_addr(base_reg, offset.wrapping_add(1));
        self.t_mload();
        self.t_push1(248);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);

        // Combine: high_byte << 8 | low_byte
        self.t_push1(8);
        self.t_binary_op(Opcode::Shl);
        self.t_binary_op(Opcode::Or);
    }

    /// Load a word from little-endian memory
    fn emit_load_word(&mut self, base_reg: u8, offset: i32) {
        // Load byte 0 (LSB)
        self.emit_compute_evm_addr(base_reg, offset);
        self.t_mload();
        self.t_push1(248);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);

        // Load byte 1
        self.emit_compute_evm_addr(base_reg, offset.wrapping_add(1));
        self.t_mload();
        self.t_push1(248);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        self.t_push1(8);
        self.t_binary_op(Opcode::Shl);
        self.t_binary_op(Opcode::Or);

        // Load byte 2
        self.emit_compute_evm_addr(base_reg, offset.wrapping_add(2));
        self.t_mload();
        self.t_push1(248);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        self.t_push1(16);
        self.t_binary_op(Opcode::Shl);
        self.t_binary_op(Opcode::Or);

        // Load byte 3 (MSB)
        self.emit_compute_evm_addr(base_reg, offset.wrapping_add(3));
        self.t_mload();
        self.t_push1(248);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        self.t_push1(24);
        self.t_binary_op(Opcode::Shl);
        self.t_binary_op(Opcode::Or);
    }

    /// Store a byte to memory
    fn emit_store_byte(&mut self, base_reg: u8, src_reg: u8, offset: i32) {
        // Load source byte and mask
        self.emit_load_reg(src_reg);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);

        // Compute address
        self.emit_load_reg(base_reg);
        self.t_push_u32(offset as u32);
        self.t_binary_op(Opcode::Add);
        self.emit_rv_to_evm_addr();

        // MSTORE8
        self.t_mstore8();
    }

    /// Store a halfword to memory (little-endian)
    fn emit_store_half(&mut self, base_reg: u8, src_reg: u8, offset: i32) {
        // Store low byte
        self.emit_load_reg(src_reg);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        self.emit_load_reg(base_reg);
        self.t_push_u32(offset as u32);
        self.t_binary_op(Opcode::Add);
        self.emit_rv_to_evm_addr();
        self.t_mstore8();

        // Store high byte
        self.emit_load_reg(src_reg);
        self.t_push1(8);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        self.emit_load_reg(base_reg);
        self.t_push_u32(offset.wrapping_add(1) as u32);
        self.t_binary_op(Opcode::Add);
        self.emit_rv_to_evm_addr();
        self.t_mstore8();
    }

    /// Store a word to memory (little-endian)
    fn emit_store_word(&mut self, base_reg: u8, src_reg: u8, offset: i32) {
        for i in 0..4 {
            self.emit_load_reg(src_reg);
            self.t_push1((i * 8) as u8);
            self.t_binary_op(Opcode::Shr);
            self.t_push1(0xFF);
            self.t_binary_op(Opcode::And);
            self.emit_load_reg(base_reg);
            self.t_push_u32(offset.wrapping_add(i) as u32);
            self.t_binary_op(Opcode::Add);
            self.emit_rv_to_evm_addr();
            self.t_mstore8();
        }
    }

    // ============================================================
    // Sign extension helpers
    // ============================================================

    /// Sign-extend from 8 bits to 32 bits
    fn emit_sign_extend_8_to_32(&mut self) {
        // EVM SIGNEXTEND(byte_pos, value) where byte_pos is s[0] (top)
        // Stack: [value] -> [value, 0] -> SIGNEXTEND(0, value)
        self.t_push0();
        self.t_binary_op(Opcode::SignExtend);
        self.emit_mask_to_32bit();
    }

    /// Sign-extend from 16 bits to 32 bits
    fn emit_sign_extend_16_to_32(&mut self) {
        // Stack: [value] -> [value, 1] -> SIGNEXTEND(1, value)
        self.t_push1(1);
        self.t_binary_op(Opcode::SignExtend);
        self.emit_mask_to_32bit();
    }

    /// Sign-extend from 32 bits to 256 bits
    fn emit_sign_extend_32_to_256(&mut self) {
        // Stack: [value] -> [value, 3] -> SIGNEXTEND(3, value)
        self.t_push1(3);
        self.t_binary_op(Opcode::SignExtend);
    }

    /// Mask value to 32 bits
    fn emit_mask_to_32bit(&mut self) {
        self.t_push4(0xFFFFFFFF);
        self.t_binary_op(Opcode::And);
    }

    // ============================================================
    // Comparison helpers
    // ============================================================

    /// Signed less-than comparison: rs1 < rs2
    fn emit_signed_lt(&mut self, rs1: u8, rs2: u8) {
        // Load rs2 first (will be at bottom of stack)
        self.emit_load_reg(rs2);
        self.emit_sign_extend_32_to_256();
        // Load rs1 second (will be on top of stack)
        self.emit_load_reg(rs1);
        self.emit_sign_extend_32_to_256();
        // SLT: returns 1 if s[0] < s[1], i.e., rs1 < rs2
        self.t_binary_op(Opcode::Slt);
    }

    /// Signed less-than comparison with immediate
    fn emit_signed_lt_imm(&mut self, rs1: u8, imm: i32) {
        // Push the immediate first (will be at bottom of stack)
        if imm >= 0 {
            self.t_push_u32(imm as u32);
        } else {
            // Negative immediate - need to sign-extend to 256 bits
            let mut bytes = [0xFFu8; 32];
            let imm_bytes = (imm as u32).to_be_bytes();
            bytes[28] = imm_bytes[0];
            bytes[29] = imm_bytes[1];
            bytes[30] = imm_bytes[2];
            bytes[31] = imm_bytes[3];
            self.t_push_u256(&bytes);
        }
        // Load rs1 second (will be on top of stack)
        self.emit_load_reg(rs1);
        self.emit_sign_extend_32_to_256();
        // SLT: returns 1 if s[0] < s[1], i.e., rs1 < imm
        self.t_binary_op(Opcode::Slt);
    }

    // ============================================================
    // Division helpers (handle division by zero)
    // ============================================================

    /// Signed division
    fn emit_signed_div(&mut self, rs1: u8, rs2: u8) {
        // Clean up cached values - division has complex internal control flow
        self.cleanup_cached_values();

        let div_label = format!("div_{}", self.bytecode.position());
        let end_label = format!("div_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.t_unary_op(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", div_label));
        self.stack.pop(); // JUMPI consumes condition

        // Check for overflow (MIN_INT / -1)
        self.emit_load_reg(rs1);
        self.t_push4(0x80000000); // MIN_INT
        self.t_binary_op(Opcode::Eq);
        self.emit_load_reg(rs2);
        self.t_push4(0xFFFFFFFF); // -1
        self.t_binary_op(Opcode::Eq);
        self.t_binary_op(Opcode::And);
        self.bytecode.jumpi_to(&format!("{}_overflow", div_label));
        self.stack.pop(); // JUMPI consumes condition

        // Normal signed division
        // Load rs2 first (bottom), then rs1 (top)
        // EVM SDIV: s[0] / s[1] = rs1 / rs2
        self.emit_load_reg(rs2);
        self.emit_sign_extend_32_to_256();
        self.emit_load_reg(rs1);
        self.emit_sign_extend_32_to_256();
        self.t_binary_op(Opcode::SDiv);
        self.emit_mask_to_32bit();
        self.bytecode.jump_to(&end_label);
        self.stack.clear(); // After jump, stack state unknown

        // Division by zero: return -1
        self.bytecode.jumpdest(&format!("{}_zero", div_label));
        self.stack.clear(); // Jump target - clear stack model
        self.t_push4(0xFFFFFFFF);
        self.bytecode.jump_to(&end_label);
        self.stack.clear();

        // Overflow: return MIN_INT
        self.bytecode.jumpdest(&format!("{}_overflow", div_label));
        self.stack.clear();
        self.t_push4(0x80000000);

        self.bytecode.jumpdest(&end_label);
        self.stack.clear();
        self.stack.push_unknown(); // Result is on stack
    }

    /// Unsigned division
    fn emit_unsigned_div(&mut self, rs1: u8, rs2: u8) {
        // Clean up cached values - division has complex internal control flow
        self.cleanup_cached_values();

        let div_label = format!("divu_{}", self.bytecode.position());
        let end_label = format!("divu_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.t_unary_op(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", div_label));
        self.stack.pop(); // JUMPI consumes condition

        // Normal unsigned division
        // Load rs2 first (bottom), then rs1 (top)
        // EVM DIV: s[0] / s[1] = rs1 / rs2
        self.emit_load_reg(rs2);
        self.emit_load_reg(rs1);
        self.t_binary_op(Opcode::Div);
        self.bytecode.jump_to(&end_label);
        self.stack.clear();

        // Division by zero: return MAX_UINT
        self.bytecode.jumpdest(&format!("{}_zero", div_label));
        self.stack.clear();
        self.t_push4(0xFFFFFFFF);

        self.bytecode.jumpdest(&end_label);
        self.stack.clear();
        self.stack.push_unknown(); // Result is on stack
    }

    /// Signed remainder
    fn emit_signed_rem(&mut self, rs1: u8, rs2: u8) {
        // Clean up cached values - remainder has complex internal control flow
        self.cleanup_cached_values();

        let rem_label = format!("rem_{}", self.bytecode.position());
        let end_label = format!("rem_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.t_unary_op(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", rem_label));
        self.stack.pop(); // JUMPI consumes condition

        // Normal signed remainder
        // Load rs2 first (bottom), then rs1 (top)
        // EVM SMOD: s[0] % s[1] = rs1 % rs2
        self.emit_load_reg(rs2);
        self.emit_sign_extend_32_to_256();
        self.emit_load_reg(rs1);
        self.emit_sign_extend_32_to_256();
        self.t_binary_op(Opcode::SMod);
        self.emit_mask_to_32bit();
        self.bytecode.jump_to(&end_label);
        self.stack.clear();

        // Division by zero: return rs1
        self.bytecode.jumpdest(&format!("{}_zero", rem_label));
        self.stack.clear();
        self.emit_load_reg(rs1);

        self.bytecode.jumpdest(&end_label);
        self.stack.clear();
        self.stack.push_unknown(); // Result is on stack
    }

    /// Unsigned remainder
    fn emit_unsigned_rem(&mut self, rs1: u8, rs2: u8) {
        // Clean up cached values - remainder has complex internal control flow
        self.cleanup_cached_values();

        let rem_label = format!("remu_{}", self.bytecode.position());
        let end_label = format!("remu_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.t_unary_op(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", rem_label));
        self.stack.pop(); // JUMPI consumes condition

        // Normal unsigned remainder
        // Load rs2 first (bottom), then rs1 (top)
        // EVM MOD: s[0] % s[1] = rs1 % rs2
        self.emit_load_reg(rs2);
        self.emit_load_reg(rs1);
        self.t_binary_op(Opcode::Mod);
        self.bytecode.jump_to(&end_label);
        self.stack.clear();

        // Division by zero: return rs1
        self.bytecode.jumpdest(&format!("{}_zero", rem_label));
        self.stack.clear();
        self.emit_load_reg(rs1);

        self.bytecode.jumpdest(&end_label);
        self.stack.clear();
        self.stack.push_unknown(); // Result is on stack
    }

    // ============================================================
    // Jump helpers
    // ============================================================

    /// Emit a jump to a RISC-V PC address
    fn emit_jump_to_rv_pc(&mut self, target_pc: u32) {
        // Clean up cached values before jumping to another basic block
        self.cleanup_cached_values();
        // Record placeholder for later resolution
        self.bytecode.emit(Opcode::Push2);
        self.stack.push_unknown();
        self.pending_jumps.push((self.bytecode.position(), target_pc));
        self.bytecode.emit_bytes(&[0, 0]); // Placeholder
        self.bytecode.emit(Opcode::Jump);
        self.stack.pop(); // JUMP consumes destination
        self.stack.clear(); // Control flow transfer - stack state unknown at target
    }

    /// Emit a conditional branch with proper cleanup
    /// Stack has condition on top, possibly with cached values below
    fn emit_branch_with_cleanup(&mut self, target_pc: u32, _fallthrough_pc: u32) {
        // Stack: [...cached..., condition]
        // Need to cleanup cached values while preserving condition

        if self.cached_count > 0 && self.cached_count <= 16 {
            // Swap condition below cached values
            // Stack: [...cached..., condition] -> [condition, ...cached...]
            self.t_swap(self.cached_count);

            // Pop cached values
            for _ in 0..self.cached_count {
                self.bytecode.emit(Opcode::Pop);
                self.stack.pop();
            }
            self.cached_count = 0;
            // Stack: [condition]
        } else if self.cached_count > 16 {
            // Too many cached values, just clear them all
            // This is rare and not worth optimizing
            for _ in 0..self.cached_count {
                self.bytecode.emit(Opcode::Pop);
                self.stack.pop();
            }
            self.cached_count = 0;
        }

        // JUMPI: s[0] = destination, s[1] = condition
        // Push target addr, then we have [condition, target] with target on top
        self.bytecode.emit(Opcode::Push2);
        self.stack.push_unknown();
        self.pending_jumps.push((self.bytecode.position(), target_pc));
        self.bytecode.emit_bytes(&[0, 0]); // Placeholder
        // Stack is now [condition, target_addr] - correct for JUMPI
        self.bytecode.emit(Opcode::JumpI);
        self.stack.pop_n(2); // JUMPI consumes condition and destination
        // Stack is now empty for fallthrough path
    }

    /// Emit dynamic jump based on computed address (for JALR)
    fn emit_dynamic_jump(&mut self) {
        // Stack has target RISC-V PC on top
        // We need to find the corresponding EVM address
        // This requires a dispatch table

        // For simplicity, we'll emit a series of comparisons
        // A more efficient approach would use a jump table

        // Store the target PC temporarily
        self.t_push_u32(PC_ADDR);
        self.t_mstore();

        // Clean up cached values before dynamic jump
        self.cleanup_cached_values();

        // Jump to dispatch routine
        self.bytecode.jump_to("dynamic_dispatch");
        self.stack.clear(); // Dynamic dispatch - stack state unknown
    }

    /// Resolve all pending jump targets
    fn resolve_jumps(&mut self) -> Result<(), String> {
        for (placeholder_pos, target_pc) in &self.pending_jumps {
            let evm_target = self.pc_to_evm.get(target_pc)
                .ok_or_else(|| format!("Jump target PC {:#x} not found", target_pc))?;
            let target_bytes = (*evm_target as u16).to_be_bytes();
            self.bytecode.bytecode_mut()[*placeholder_pos] = target_bytes[0];
            self.bytecode.bytecode_mut()[*placeholder_pos + 1] = target_bytes[1];
        }
        self.pending_jumps.clear();

        // Now emit the dynamic dispatch routine
        self.emit_dynamic_dispatch();

        Ok(())
    }

    /// Emit the dynamic dispatch routine for JALR
    fn emit_dynamic_dispatch(&mut self) {
        self.bytecode.jumpdest("dynamic_dispatch");

        // Load the target PC from memory
        self.bytecode.push_u32(PC_ADDR);
        self.bytecode.emit(Opcode::MLoad);

        // Generate dispatch table
        for (rv_pc, evm_pos) in self.pc_to_evm.iter() {
            // DUP1 (duplicate target PC)
            self.bytecode.emit(Opcode::Dup1);
            // Push RV PC to compare
            self.bytecode.push_u32(*rv_pc);
            self.bytecode.push1(224);
            self.bytecode.emit(Opcode::Shl);
            // EQ
            self.bytecode.emit(Opcode::Eq);
            // If equal, jump to EVM position
            self.bytecode.push2(*evm_pos as u16);
            self.bytecode.emit(Opcode::Swap1);
            self.bytecode.emit(Opcode::JumpI);
        }

        // If no match found, halt
        self.bytecode.emit(Opcode::Pop); // Pop the target PC
        self.bytecode.jump_to("halt");
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}
