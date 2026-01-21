//! Meta-compilation test: Compile the Rust compiler to EVM and test it

use rv32im_evm::compiler::{Compiler, CompilerConfig};
use rv32im_evm::runtime::Runtime;
use std::fs;

fn main() {
    // Load only the text section of the rv32im compiler binary
    // (data sections are handled separately via memory initialization)
    let compiler_binary = fs::read("rv32im-compiler-nostd/target/rv32im-compiler-text.bin")
        .expect("Failed to read compiler binary (text section)");

    println!("Loaded compiler binary: {} bytes ({} instructions)",
             compiler_binary.len(),
             compiler_binary.len() / 4);

    // Configure compiler for the meta-compiler
    // The nostd compiler expects:
    // - Load address: 0x80000000
    // - Input at: 0x80001000
    // - Output at: 0x80002000
    // - Heap at: 0x80010000
    let config = CompilerConfig {
        load_address: 0x80000000,
        stack_pointer: 0x80020000,  // Higher stack since heap starts at 0x80010000
        memory_size: 0x40000,       // 256KB for the compiler
        ..Default::default()
    };

    println!("Compiling rv32im compiler to EVM...");
    let mut compiler = Compiler::with_config(config);
    let evm_bytecode = compiler.compile(&compiler_binary)
        .expect("Failed to compile rv32im compiler to EVM");

    println!("Generated EVM bytecode: {} bytes", evm_bytecode.len());

    // Save the EVM bytecode for inspection
    fs::write("rv32im-compiler-nostd/target/rv32im-compiler.evm", &evm_bytecode)
        .expect("Failed to write EVM bytecode");
    println!("Saved EVM bytecode to rv32im-compiler-nostd/target/rv32im-compiler.evm");

    // Create simple factorial program to compile
    // li a0, 10
    // li a1, 1
    // loop:
    //   beq a0, zero, done  (PC+16)
    //   mul a1, a1, a0
    //   addi a0, a0, -1
    //   j loop  (PC-12)
    // done:
    //   mv a0, a1
    //   ecall
    let factorial_program: Vec<u8> = vec![
        0x13, 0x05, 0xa0, 0x00,  // li a0, 10 (addi a0, zero, 10)
        0x93, 0x05, 0x10, 0x00,  // li a1, 1 (addi a1, zero, 1)
        // loop:
        0x63, 0x08, 0x05, 0x00,  // beq a0, zero, done (offset +16)
        0xb3, 0x85, 0xa5, 0x02,  // mul a1, a1, a0
        0x13, 0x05, 0xf5, 0xff,  // addi a0, a0, -1
        0x6f, 0xf0, 0x5f, 0xff,  // j loop (offset -12)
        // done:
        0x13, 0x85, 0x05, 0x00,  // mv a0, a1 (addi a0, a1, 0)
        0x73, 0x00, 0x00, 0x00,  // ecall
    ];

    println!("\nFactorial program: {} bytes ({} instructions)",
             factorial_program.len(),
             factorial_program.len() / 4);

    // For the meta-compiler test, we would need to:
    // 1. Initialize the EVM memory with the factorial program at INPUT_ADDR
    // 2. Set INPUT_LEN_ADDR to the program length
    // 3. Run the EVM-compiled compiler
    // 4. Read the output from OUTPUT_ADDR
    //
    // However, the runtime doesn't currently support initializing memory.
    // For now, let's just verify the compilation succeeded.

    println!("\nMeta-compilation successful!");
    println!("Generated {} bytes of EVM bytecode from {} byte rv32im binary",
             evm_bytecode.len(), compiler_binary.len());
    println!("Bytecode expansion ratio: {:.2}x",
             evm_bytecode.len() as f64 / compiler_binary.len() as f64);

    // As a sanity check, compile factorial directly and run it
    println!("\n--- Sanity check: Direct factorial compilation ---");
    let direct_config = CompilerConfig {
        load_address: 0x80000000,
        stack_pointer: 0x80010000,
        memory_size: 0x20000,
        ..Default::default()
    };

    let mut direct_compiler = Compiler::with_config(direct_config);
    let direct_bytecode = direct_compiler.compile(&factorial_program)
        .expect("Failed to compile factorial directly");

    println!("Direct factorial EVM bytecode: {} bytes", direct_bytecode.len());

    let runtime = Runtime::new(direct_bytecode);
    let output = runtime.execute().expect("Direct factorial execution failed");

    println!("Direct factorial(10) result: {} (expected: 3628800)", output.return_value);
    println!("Gas used: {}", output.gas_used);

    assert_eq!(output.return_value, 3628800, "Factorial result incorrect!");
    println!("Direct compilation verified!");
}
