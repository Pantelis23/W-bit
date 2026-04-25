// W-bit Update Engine (Row-Parallel Hebbian Learning)
// Computes W_new[j] = W[j] + (x * y[j] >>> shift) for j=0..255
// Uses flattened ports for Yosys compatibility.

module wbit_update_engine #(
    parameter COLS = 256,
    parameter CELL_BITS = 8,
    parameter ACCUM_BITS = 32
) (
    input  logic signed [15:0] x_val,        
    input  logic signed [COLS*ACCUM_BITS-1:0] y_vec_flat, 
    input  logic signed [COLS*CELL_BITS-1:0]  w_row_in_flat,
    input  logic [4:0] lr_shift,             
    
    output logic [COLS*CELL_BITS-1:0] w_row_out_flat
);

    // Unpack for usage
    logic signed [ACCUM_BITS-1:0] y_vec [COLS];
    logic signed [CELL_BITS-1:0] w_row_in [COLS];
    logic [CELL_BITS-1:0] w_row_out [COLS];
    
    always_comb begin
        for (int k=0; k<COLS; k++) begin
            y_vec[k] = y_vec_flat[k*ACCUM_BITS +: ACCUM_BITS];
            w_row_in[k] = w_row_in_flat[k*CELL_BITS +: CELL_BITS];
        end
    end
    
    // Repack output
    always_comb begin
        for (int k=0; k<COLS; k++) begin
            w_row_out_flat[k*CELL_BITS +: CELL_BITS] = w_row_out[k];
        end
    end

    // Parallel Update Units
    genvar j;
    generate
        for (j = 0; j < COLS; j++) begin : g_update
            logic signed [47:0] prod;
            logic signed [CELL_BITS-1:0] delta;
            logic signed [CELL_BITS+1:0] w_sum;
            
            always_comb begin
                // 1. Calculate Delta: x * y
                prod = x_val * y_vec[j];
                
                // 2. Apply Learning Rate (Shift)
                delta = prod >>> lr_shift;
                
                // 3. Update Weight
                w_sum = w_row_in[j] + delta;
                
                // 4. Saturate
                if (w_sum > 127)       w_row_out[j] = 8'sd127;
                else if (w_sum < -128) w_row_out[j] = -8'sd128;
                else                   w_row_out[j] = w_sum[7:0];
            end
        end
    endgenerate

endmodule