//! RISC-V rv32im to EVM compiler CLI

use rv32im_evm::{Compiler, compiler::CompilerConfig};
use std::fs;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.bin> [output.bin]", args[0]);
        eprintln!("  Compiles RISC-V rv32im binary to EVM bytecode");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(&args[1]);
    let output_path = if args.len() > 2 {
        PathBuf::from(&args[2])
    } else {
        input_path.with_extension("evm")
    };

    // Read input file
    let program = fs::read(&input_path).unwrap_or_else(|e| {
        eprintln!("Error reading input file: {}", e);
        std::process::exit(1);
    });

    // Compile
    let config = CompilerConfig::default();
    let mut compiler = Compiler::with_config(config);

    let bytecode = compiler.compile(&program).unwrap_or_else(|e| {
        eprintln!("Compilation error: {}", e);
        std::process::exit(1);
    });

    // Write output
    fs::write(&output_path, &bytecode).unwrap_or_else(|e| {
        eprintln!("Error writing output file: {}", e);
        std::process::exit(1);
    });

    println!("Compiled {} bytes of RISC-V to {} bytes of EVM bytecode",
             program.len(), bytecode.len());
    println!("Output written to: {}", output_path.display());
}
