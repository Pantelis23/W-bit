// 16-bit Linear Feedback Shift Register
// Generates pseudo-random noise for Analog simulation.

module wbit_lfsr (
    input  logic clk,
    input  logic rst_n,
    input  logic enable,
    output logic [15:0] rand_out
);

    logic [15:0] lfsr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr <= 16'hACE1; // Non-zero seed
        end else if (enable) begin
            // Xnor taps for 16-bit: 16, 14, 13, 11
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
        end
    end
    
    assign rand_out = lfsr;

endmodule
