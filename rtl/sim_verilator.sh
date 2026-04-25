#!/bin/bash
# W-bit Verilator Simulation Script

# Ensure Verilator is installed
if ! command -v verilator &> /dev/null; then
    echo "Verilator not found. Installing..."
    sudo apt-get install -y verilator
fi

echo "Compiling W-bit RTL..."
# Note: we compile wbit_top.sv which includes the others via instantiation,
# but we must list them all or rely on include paths. Listing is safer.
verilator -Wall --trace --cc wbit_top.sv wbit_accelerator.sv wbit_tile.sv wbit_if.sv --exe sim_main.cpp

echo "Building Simulation..."
make -j -C obj_dir -f Vwbit_top.mk

echo "Running Simulation..."
./obj_dir/Vwbit_top

echo "Done. Waveform saved to waveform.vcd"