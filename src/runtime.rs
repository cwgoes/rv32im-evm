//! EVM runtime for executing compiled RISC-V programs
//!
//! Uses revm (reth's EVM implementation) to execute the compiled bytecode.

use revm::{
    primitives::{
        Address, Bytes, ExecutionResult, Output, U256, Bytecode,
        AccountInfo, TxKind, SpecId,
    },
    interpreter::analysis::to_analysed,
    inspectors::CustomPrintTracer,
    inspector_handle_register,
    Evm, InMemoryDB,
};

/// Runtime for executing compiled RISC-V programs on the EVM
#[allow(dead_code)]
pub struct Runtime {
    /// The compiled EVM bytecode
    bytecode: Vec<u8>,
    /// Initial memory state (RISC-V memory)
    initial_memory: Vec<u8>,
    /// Memory base offset in EVM
    mem_base: u32,
    /// Calldata to pass to the contract
    calldata: Vec<u8>,
}

impl Runtime {
    /// Create a new runtime with compiled bytecode
    pub fn new(bytecode: Vec<u8>) -> Self {
        Self {
            bytecode,
            initial_memory: Vec::new(),
            mem_base: 0x0100,
            calldata: Vec::new(),
        }
    }

    /// Set the initial memory state
    pub fn with_memory(mut self, memory: Vec<u8>) -> Self {
        self.initial_memory = memory;
        self
    }

    /// Set calldata to pass to the contract
    pub fn with_calldata(mut self, calldata: Vec<u8>) -> Self {
        self.calldata = calldata;
        self
    }

    /// Create analyzed bytecode with properly built jump table
    fn create_analyzed_bytecode(&self) -> Bytecode {
        // Create LegacyRaw bytecode
        let raw = Bytecode::new_legacy(Bytes::from(self.bytecode.clone()));

        // Explicitly analyze to build jump table
        // This converts LegacyRaw -> LegacyAnalyzed with proper JumpTable
        to_analysed(raw)
    }

    /// Execute the program and return the result
    pub fn execute(&self) -> Result<ExecutionOutput, ExecutionError> {
        self.execute_internal(false)
    }

    /// Execute with step-by-step tracing
    pub fn execute_traced(&self) -> Result<ExecutionOutput, ExecutionError> {
        self.execute_internal(true)
    }

    /// Internal execute with optional tracing
    fn execute_internal(&self, traced: bool) -> Result<ExecutionOutput, ExecutionError> {
        // Create in-memory database
        let mut db = InMemoryDB::default();

        // Contract address
        let contract_addr = Address::from_slice(&[0x42; 20]);

        // Analyze bytecode to build jump table
        let bytecode = self.create_analyzed_bytecode();

        let account = AccountInfo {
            balance: U256::ZERO,
            nonce: 0,
            code_hash: bytecode.hash_slow(),
            code: Some(bytecode),
        };
        db.insert_account_info(contract_addr, account);

        // Prepare calldata
        let calldata = Bytes::from(self.calldata.clone());

        // Execute with or without tracing
        let result = if traced {
            let mut evm = Evm::builder()
                .with_db(db)
                .with_spec_id(SpecId::CANCUN)
                .modify_tx_env(|tx| {
                    tx.transact_to = TxKind::Call(contract_addr);
                    tx.data = calldata.clone();
                    tx.gas_limit = 1_000_000_000;
                    tx.gas_price = U256::from(0);
                })
                .with_external_context(CustomPrintTracer::default())
                .append_handler_register(inspector_handle_register)
                .build();
            evm.transact_commit()
        } else {
            let mut evm = Evm::builder()
                .with_db(db)
                .with_spec_id(SpecId::CANCUN)
                .modify_tx_env(|tx| {
                    tx.transact_to = TxKind::Call(contract_addr);
                    tx.data = calldata.clone();
                    tx.gas_limit = 1_000_000_000;
                    tx.gas_price = U256::from(0);
                })
                .build();
            evm.transact_commit()
        };

        let result = result.map_err(|e| ExecutionError::EvmError(format!("{:?}", e)))?;

        match result {
            ExecutionResult::Success { output, gas_used, .. } => {
                match output {
                    Output::Call(bytes) => {
                        // Parse return value (32 bytes, big-endian)
                        let return_value = if bytes.len() >= 32 {
                            u32::from_be_bytes([bytes[28], bytes[29], bytes[30], bytes[31]])
                        } else if bytes.len() >= 4 {
                            u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                        } else {
                            0
                        };
                        Ok(ExecutionOutput {
                            return_value,
                            gas_used,
                        })
                    }
                    Output::Create(_, _) => {
                        Err(ExecutionError::UnexpectedCreate)
                    }
                }
            }
            ExecutionResult::Revert { output, .. } => {
                Err(ExecutionError::Revert(hex::encode(&output)))
            }
            ExecutionResult::Halt { reason, .. } => {
                Err(ExecutionError::Halt(format!("{:?}", reason)))
            }
        }
    }
}

/// Output from program execution
#[derive(Debug, Clone)]
pub struct ExecutionOutput {
    /// Return value (from a0/x10 register)
    pub return_value: u32,
    /// Gas used during execution
    pub gas_used: u64,
}

/// Errors that can occur during execution
#[derive(Debug, Clone)]
pub enum ExecutionError {
    EvmError(String),
    Revert(String),
    Halt(String),
    UnexpectedCreate,
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::EvmError(e) => write!(f, "EVM error: {}", e),
            ExecutionError::Revert(e) => write!(f, "Revert: {}", e),
            ExecutionError::Halt(e) => write!(f, "Halt: {}", e),
            ExecutionError::UnexpectedCreate => write!(f, "Unexpected CREATE output"),
        }
    }
}

impl std::error::Error for ExecutionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{Compiler, CompilerConfig};

    #[test]
    fn test_simple_return() {
        // Simple program: li a0, 42; ecall
        // li a0, 42 = addi a0, zero, 42 = 0x02a00513
        // ecall = 0x00000073
        let program = vec![
            0x13, 0x05, 0xa0, 0x02, // addi a0, zero, 42
            0x73, 0x00, 0x00, 0x00, // ecall
        ];

        let config = CompilerConfig {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x20000,
            ..Default::default()
        };

        let mut compiler = Compiler::with_config(config);
        let bytecode = compiler.compile(&program).expect("Compilation failed");

        let runtime = Runtime::new(bytecode);
        let output = runtime.execute().expect("Execution failed");

        assert_eq!(output.return_value, 42);
    }
}
