#include "Vwbit_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <iostream>

vluint64_t main_time = 0;

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    Vwbit_top* top = new Vwbit_top;
    VerilatedVcdC* tfp = new VerilatedVcdC;
    
    top->trace(tfp, 99);
    tfp->open("waveform.vcd");

    top->clk = 0;
    top->rst_n = 0;
    
    // 1. Reset
    while (main_time < 20) {
        if ((main_time % 5) == 0) top->clk = !top->clk;
        if (main_time == 10) top->rst_n = 1;
        top->eval();
        tfp->dump(main_time);
        main_time++;
    }
    
    // 2. CONFIG: Write W[0][0] = 20
    printf("Writing W=20...\n");
    top->bus_cfg_req_valid = 1;
    top->bus_cfg_req_rw = 1; // Write
    top->bus_cfg_req_addr = 0; // Row 0, Col 0
    top->bus_cfg_req_data = 20;
    
    // Clock it in
    for (int i=0; i<20; i++) {
        if ((main_time % 5) == 0) top->clk = !top->clk;
        top->eval(); tfp->dump(main_time++);
    }
    top->bus_cfg_req_valid = 0;
    
    // 3. COMPUTE: Input X[0] = 1000
    printf("Computing X=1000...\n");
    top->bus_op_start = 1;
    top->bus_data_in_valid = 1;
    top->bus_data_in_last = 1; // Finish loading immediately
    top->bus_data_in_payload[0] = 1000; 
    
    // Run compute cycles (Give it 50 cycles to be safe)
    for (int i=0; i<500; i++) {
        if ((main_time % 5) == 0) top->clk = !top->clk;
        if (i==20) top->bus_op_start = 0; 
        if (i==20) top->bus_data_in_valid = 0; // Clear input valid too
        top->eval(); tfp->dump(main_time++);
    }
    
    // 4. LEARN: Trigger Update
    printf("Triggering Learn...\n");
    top->bus_op_learn = 1;
    for (int i=0; i<40; i++) {
        if ((main_time % 5) == 0) top->clk = !top->clk;
        top->eval(); tfp->dump(main_time++);
    }
    top->bus_op_learn = 0;
    
    // Run update cycles (needs 256 cycles + margin)
    for (int i=0; i<3000; i++) {
        if ((main_time % 5) == 0) top->clk = !top->clk;
        top->eval(); tfp->dump(main_time++);
    }
    
    // 5. VERIFY: Read W[0][0]
    printf("Reading back W...\n");
    top->bus_cfg_req_valid = 1;
    top->bus_cfg_req_rw = 0; // Read
    top->bus_cfg_req_addr = 0;
    
    // Wait for read
    for (int i=0; i<40; i++) {
        if ((main_time % 5) == 0) top->clk = !top->clk;
        top->eval(); tfp->dump(main_time++);
    }
    
    int read_val = top->bus_cfg_resp_data & 0xFF;
    printf("Read W = %d\n", read_val);
    
    // Since we write 8-bit signed, 39 fits.
    if (read_val == 39) {
        printf("SUCCESS: W updated correctly (20 -> 39)\n");
    } else {
        printf("FAILURE: Expected 39, got %d\n", read_val);
    }

    top->final();
    tfp->close();
    delete top;
    return 0;
}