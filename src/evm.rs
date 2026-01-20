//! EVM bytecode builder
//!
//! Provides utilities for constructing EVM bytecode.

use std::collections::HashMap;

/// EVM opcodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    // Stop and Arithmetic
    Stop = 0x00,
    Add = 0x01,
    Mul = 0x02,
    Sub = 0x03,
    Div = 0x04,
    SDiv = 0x05,
    Mod = 0x06,
    SMod = 0x07,
    AddMod = 0x08,
    MulMod = 0x09,
    Exp = 0x0a,
    SignExtend = 0x0b,

    // Comparison & Bitwise Logic
    Lt = 0x10,
    Gt = 0x11,
    Slt = 0x12,
    Sgt = 0x13,
    Eq = 0x14,
    IsZero = 0x15,
    And = 0x16,
    Or = 0x17,
    Xor = 0x18,
    Not = 0x19,
    Byte = 0x1a,
    Shl = 0x1b,
    Shr = 0x1c,
    Sar = 0x1d,

    // SHA3
    Keccak256 = 0x20,

    // Environmental Information
    Address = 0x30,
    Balance = 0x31,
    Origin = 0x32,
    Caller = 0x33,
    CallValue = 0x34,
    CallDataLoad = 0x35,
    CallDataSize = 0x36,
    CallDataCopy = 0x37,
    CodeSize = 0x38,
    CodeCopy = 0x39,
    GasPrice = 0x3a,
    ExtCodeSize = 0x3b,
    ExtCodeCopy = 0x3c,
    ReturnDataSize = 0x3d,
    ReturnDataCopy = 0x3e,
    ExtCodeHash = 0x3f,

    // Block Information
    BlockHash = 0x40,
    Coinbase = 0x41,
    Timestamp = 0x42,
    Number = 0x43,
    Difficulty = 0x44,
    GasLimit = 0x45,
    ChainId = 0x46,
    SelfBalance = 0x47,
    BaseFee = 0x48,

    // Stack, Memory, Storage and Flow
    Pop = 0x50,
    MLoad = 0x51,
    MStore = 0x52,
    MStore8 = 0x53,
    SLoad = 0x54,
    SStore = 0x55,
    Jump = 0x56,
    JumpI = 0x57,
    Pc = 0x58,
    MSize = 0x59,
    Gas = 0x5a,
    JumpDest = 0x5b,

    // Push operations
    Push0 = 0x5f,
    Push1 = 0x60,
    Push2 = 0x61,
    Push3 = 0x62,
    Push4 = 0x63,
    Push5 = 0x64,
    Push6 = 0x65,
    Push7 = 0x66,
    Push8 = 0x67,
    Push9 = 0x68,
    Push10 = 0x69,
    Push11 = 0x6a,
    Push12 = 0x6b,
    Push13 = 0x6c,
    Push14 = 0x6d,
    Push15 = 0x6e,
    Push16 = 0x6f,
    Push17 = 0x70,
    Push18 = 0x71,
    Push19 = 0x72,
    Push20 = 0x73,
    Push21 = 0x74,
    Push22 = 0x75,
    Push23 = 0x76,
    Push24 = 0x77,
    Push25 = 0x78,
    Push26 = 0x79,
    Push27 = 0x7a,
    Push28 = 0x7b,
    Push29 = 0x7c,
    Push30 = 0x7d,
    Push31 = 0x7e,
    Push32 = 0x7f,

    // Dup operations
    Dup1 = 0x80,
    Dup2 = 0x81,
    Dup3 = 0x82,
    Dup4 = 0x83,
    Dup5 = 0x84,
    Dup6 = 0x85,
    Dup7 = 0x86,
    Dup8 = 0x87,
    Dup9 = 0x88,
    Dup10 = 0x89,
    Dup11 = 0x8a,
    Dup12 = 0x8b,
    Dup13 = 0x8c,
    Dup14 = 0x8d,
    Dup15 = 0x8e,
    Dup16 = 0x8f,

    // Swap operations
    Swap1 = 0x90,
    Swap2 = 0x91,
    Swap3 = 0x92,
    Swap4 = 0x93,
    Swap5 = 0x94,
    Swap6 = 0x95,
    Swap7 = 0x96,
    Swap8 = 0x97,
    Swap9 = 0x98,
    Swap10 = 0x99,
    Swap11 = 0x9a,
    Swap12 = 0x9b,
    Swap13 = 0x9c,
    Swap14 = 0x9d,
    Swap15 = 0x9e,
    Swap16 = 0x9f,

    // Log operations
    Log0 = 0xa0,
    Log1 = 0xa1,
    Log2 = 0xa2,
    Log3 = 0xa3,
    Log4 = 0xa4,

    // System operations
    Create = 0xf0,
    Call = 0xf1,
    CallCode = 0xf2,
    Return = 0xf3,
    DelegateCall = 0xf4,
    Create2 = 0xf5,
    StaticCall = 0xfa,
    Revert = 0xfd,
    Invalid = 0xfe,
    SelfDestruct = 0xff,
}

/// Placeholder for forward jumps that need to be resolved later
#[derive(Debug, Clone)]
pub struct JumpPlaceholder {
    /// Position in bytecode where the jump target needs to be written
    pub position: usize,
    /// Label to jump to
    pub label: String,
}

/// EVM bytecode builder
#[derive(Debug, Default)]
pub struct EvmBytecode {
    /// The bytecode being built
    bytecode: Vec<u8>,
    /// Label positions
    labels: HashMap<String, usize>,
    /// Unresolved jump placeholders
    placeholders: Vec<JumpPlaceholder>,
}

impl EvmBytecode {
    /// Create a new empty bytecode builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the current position in bytecode
    pub fn position(&self) -> usize {
        self.bytecode.len()
    }

    /// Emit a single opcode
    pub fn emit(&mut self, opcode: Opcode) -> &mut Self {
        self.bytecode.push(opcode as u8);
        self
    }

    /// Emit raw bytes
    pub fn emit_bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.bytecode.extend_from_slice(bytes);
        self
    }

    /// Patch bytes at a specific position (for resolving forward references)
    pub fn patch(&mut self, position: usize, bytes: &[u8]) {
        for (i, &byte) in bytes.iter().enumerate() {
            if position + i < self.bytecode.len() {
                self.bytecode[position + i] = byte;
            }
        }
    }

    /// Emit a raw byte
    pub fn emit_byte(&mut self, byte: u8) -> &mut Self {
        self.bytecode.push(byte);
        self
    }

    /// Emit PUSH0
    pub fn push0(&mut self) -> &mut Self {
        self.emit(Opcode::Push0)
    }

    /// Emit PUSH1 with value
    pub fn push1(&mut self, value: u8) -> &mut Self {
        self.emit(Opcode::Push1);
        self.bytecode.push(value);
        self
    }

    /// Emit PUSH2 with value
    pub fn push2(&mut self, value: u16) -> &mut Self {
        self.emit(Opcode::Push2);
        self.bytecode.extend_from_slice(&value.to_be_bytes());
        self
    }

    /// Emit PUSH4 with value
    pub fn push4(&mut self, value: u32) -> &mut Self {
        self.emit(Opcode::Push4);
        self.bytecode.extend_from_slice(&value.to_be_bytes());
        self
    }

    /// Emit PUSH32 with value
    pub fn push32(&mut self, value: [u8; 32]) -> &mut Self {
        self.emit(Opcode::Push32);
        self.bytecode.extend_from_slice(&value);
        self
    }

    /// Emit a push instruction for the smallest size that fits the value
    pub fn push_u32(&mut self, value: u32) -> &mut Self {
        if value == 0 {
            self.push0()
        } else if value <= 0xFF {
            self.push1(value as u8)
        } else if value <= 0xFFFF {
            self.push2(value as u16)
        } else if value <= 0xFFFFFF {
            self.emit(Opcode::Push3);
            self.bytecode.push((value >> 16) as u8);
            self.bytecode.push((value >> 8) as u8);
            self.bytecode.push(value as u8);
            self
        } else {
            self.push4(value)
        }
    }

    /// Emit a push instruction for a 256-bit value (as big-endian bytes)
    pub fn push_u256(&mut self, value: &[u8; 32]) -> &mut Self {
        // Find the first non-zero byte
        let first_nonzero = value.iter().position(|&b| b != 0);

        match first_nonzero {
            None => self.push0(),
            Some(idx) => {
                let len = 32 - idx;
                let opcode = (Opcode::Push1 as u8 + len as u8 - 1) as u8;
                self.bytecode.push(opcode);
                self.bytecode.extend_from_slice(&value[idx..]);
                self
            }
        }
    }

    /// Define a label at the current position
    pub fn label(&mut self, name: &str) -> &mut Self {
        self.labels.insert(name.to_string(), self.bytecode.len());
        self
    }

    /// Emit a JUMPDEST with a label
    pub fn jumpdest(&mut self, name: &str) -> &mut Self {
        self.label(name);
        self.emit(Opcode::JumpDest)
    }

    /// Emit a JUMP to a label (may be forward reference)
    pub fn jump_to(&mut self, label: &str) -> &mut Self {
        if let Some(&target) = self.labels.get(label) {
            // Label is already defined
            self.push2(target as u16);
        } else {
            // Forward reference - emit placeholder
            self.emit(Opcode::Push2);
            self.placeholders.push(JumpPlaceholder {
                position: self.bytecode.len(),
                label: label.to_string(),
            });
            self.bytecode.extend_from_slice(&[0, 0]); // Placeholder bytes
        }
        self.emit(Opcode::Jump)
    }

    /// Emit a conditional JUMPI to a label
    pub fn jumpi_to(&mut self, label: &str) -> &mut Self {
        if let Some(&target) = self.labels.get(label) {
            self.push2(target as u16);
        } else {
            self.emit(Opcode::Push2);
            self.placeholders.push(JumpPlaceholder {
                position: self.bytecode.len(),
                label: label.to_string(),
            });
            self.bytecode.extend_from_slice(&[0, 0]);
        }
        self.emit(Opcode::JumpI)
    }

    /// Push the address of a label
    pub fn push_label(&mut self, label: &str) -> &mut Self {
        if let Some(&target) = self.labels.get(label) {
            self.push2(target as u16);
        } else {
            self.emit(Opcode::Push2);
            self.placeholders.push(JumpPlaceholder {
                position: self.bytecode.len(),
                label: label.to_string(),
            });
            self.bytecode.extend_from_slice(&[0, 0]);
        }
        self
    }

    /// Resolve all forward references
    pub fn resolve(&mut self) -> Result<(), String> {
        for placeholder in &self.placeholders {
            let target = self.labels.get(&placeholder.label)
                .ok_or_else(|| format!("Undefined label: {}", placeholder.label))?;
            let target_bytes = (*target as u16).to_be_bytes();
            self.bytecode[placeholder.position] = target_bytes[0];
            self.bytecode[placeholder.position + 1] = target_bytes[1];
        }
        self.placeholders.clear();
        Ok(())
    }

    /// Finalize and return the bytecode
    pub fn finalize(mut self) -> Result<Vec<u8>, String> {
        self.resolve()?;
        Ok(self.bytecode)
    }

    /// Get the bytecode without resolving (for inspection)
    pub fn bytecode(&self) -> &[u8] {
        &self.bytecode
    }

    /// Get the raw bytecode mutably
    pub fn bytecode_mut(&mut self) -> &mut Vec<u8> {
        &mut self.bytecode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_values() {
        let mut builder = EvmBytecode::new();
        builder.push0();
        assert_eq!(builder.bytecode(), &[0x5f]);

        let mut builder = EvmBytecode::new();
        builder.push1(0x42);
        assert_eq!(builder.bytecode(), &[0x60, 0x42]);

        let mut builder = EvmBytecode::new();
        builder.push2(0x1234);
        assert_eq!(builder.bytecode(), &[0x61, 0x12, 0x34]);
    }

    #[test]
    fn test_labels() {
        let mut builder = EvmBytecode::new();
        builder.jumpdest("start");
        builder.push1(1);
        builder.jump_to("start");
        let bytecode = builder.finalize().unwrap();
        // JUMPDEST, PUSH1 1, PUSH2 0x0000, JUMP
        assert_eq!(bytecode, &[0x5b, 0x60, 0x01, 0x61, 0x00, 0x00, 0x56]);
    }
}
