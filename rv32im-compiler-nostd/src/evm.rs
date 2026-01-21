//! EVM bytecode builder (no_std version)

use alloc::{vec::Vec, string::String, collections::BTreeMap};

/// EVM opcodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
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
    Keccak256 = 0x20,
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
    Push0 = 0x5f,
    Push1 = 0x60,
    Push2 = 0x61,
    Push3 = 0x62,
    Push4 = 0x63,
    Push32 = 0x7f,
    Dup1 = 0x80,
    Dup2 = 0x81,
    Dup3 = 0x82,
    Dup4 = 0x83,
    Dup5 = 0x84,
    Dup6 = 0x85,
    Dup7 = 0x86,
    Dup8 = 0x87,
    Swap1 = 0x90,
    Swap2 = 0x91,
    Swap3 = 0x92,
    Swap4 = 0x93,
    Swap5 = 0x94,
    Swap6 = 0x95,
    Swap7 = 0x96,
    Swap8 = 0x97,
    Return = 0xf3,
    Revert = 0xfd,
    Invalid = 0xfe,
}

/// Placeholder for forward jumps
pub struct JumpPlaceholder {
    pub position: usize,
    pub label: String,
}

/// EVM bytecode builder
pub struct EvmBytecode {
    bytecode: Vec<u8>,
    labels: BTreeMap<String, usize>,
    placeholders: Vec<JumpPlaceholder>,
}

impl EvmBytecode {
    pub fn new() -> Self {
        Self {
            bytecode: Vec::new(),
            labels: BTreeMap::new(),
            placeholders: Vec::new(),
        }
    }

    pub fn position(&self) -> usize {
        self.bytecode.len()
    }

    pub fn emit(&mut self, opcode: Opcode) -> &mut Self {
        self.bytecode.push(opcode as u8);
        self
    }

    pub fn emit_bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.bytecode.extend_from_slice(bytes);
        self
    }

    pub fn emit_byte(&mut self, byte: u8) -> &mut Self {
        self.bytecode.push(byte);
        self
    }

    pub fn push0(&mut self) -> &mut Self {
        self.emit(Opcode::Push0)
    }

    pub fn push1(&mut self, value: u8) -> &mut Self {
        self.emit(Opcode::Push1);
        self.bytecode.push(value);
        self
    }

    pub fn push2(&mut self, value: u16) -> &mut Self {
        self.emit(Opcode::Push2);
        self.bytecode.extend_from_slice(&value.to_be_bytes());
        self
    }

    pub fn push4(&mut self, value: u32) -> &mut Self {
        self.emit(Opcode::Push4);
        self.bytecode.extend_from_slice(&value.to_be_bytes());
        self
    }

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

    pub fn label(&mut self, name: &str) -> &mut Self {
        self.labels.insert(String::from(name), self.bytecode.len());
        self
    }

    pub fn jumpdest(&mut self, name: &str) -> &mut Self {
        self.label(name);
        self.emit(Opcode::JumpDest)
    }

    pub fn jump_to(&mut self, label: &str) -> &mut Self {
        if let Some(&target) = self.labels.get(label) {
            self.push2(target as u16);
        } else {
            self.emit(Opcode::Push2);
            self.placeholders.push(JumpPlaceholder {
                position: self.bytecode.len(),
                label: String::from(label),
            });
            self.bytecode.extend_from_slice(&[0, 0]);
        }
        self.emit(Opcode::Jump)
    }

    pub fn jumpi_to(&mut self, label: &str) -> &mut Self {
        if let Some(&target) = self.labels.get(label) {
            self.push2(target as u16);
        } else {
            self.emit(Opcode::Push2);
            self.placeholders.push(JumpPlaceholder {
                position: self.bytecode.len(),
                label: String::from(label),
            });
            self.bytecode.extend_from_slice(&[0, 0]);
        }
        self.emit(Opcode::JumpI)
    }

    pub fn resolve(&mut self) -> Result<(), &'static str> {
        for placeholder in &self.placeholders {
            let target = self.labels.get(&placeholder.label)
                .ok_or("Undefined label")?;
            let target_bytes = (*target as u16).to_be_bytes();
            self.bytecode[placeholder.position] = target_bytes[0];
            self.bytecode[placeholder.position + 1] = target_bytes[1];
        }
        self.placeholders.clear();
        Ok(())
    }

    pub fn finalize(mut self) -> Result<Vec<u8>, &'static str> {
        self.resolve()?;
        Ok(self.bytecode)
    }

    pub fn bytecode(&self) -> &[u8] {
        &self.bytecode
    }

    pub fn bytecode_mut(&mut self) -> &mut Vec<u8> {
        &mut self.bytecode
    }
}

impl Default for EvmBytecode {
    fn default() -> Self {
        Self::new()
    }
}
