// W-bit Top Level Wrapper for Verilator
// Flattens the interface to standard ports.

module wbit_top #(
    parameter NUM_TILES = 1,
    parameter TILE_ROWS = 256,
    parameter TILE_COLS = 256
) (
    input logic clk,
    input logic rst_n,
    
    // Flattened Interface Signals
    input  logic        bus_cfg_req_valid,
    input  logic        bus_cfg_req_rw,
    input  logic [31:0] bus_cfg_req_addr,
    input  logic [31:0] bus_cfg_req_data,
    
    output logic        bus_cfg_resp_valid,
    output logic [31:0] bus_cfg_resp_data,
    
    input  logic        bus_data_in_valid,
    input  logic        bus_data_in_last,
    input  logic [255:0] bus_data_in_payload,
    
    output logic        bus_data_out_valid,
    output logic        bus_data_out_last,
    output logic [255:0] bus_data_out_payload,
    
    input  logic        bus_op_start,
    input  logic        bus_op_learn,    // New Signal
    input  logic [15:0] bus_op_patch_id,
    input  logic [15:0] bus_op_layer_id,
    output logic        bus_op_busy,
    output logic        bus_op_done
);

    // Interface Instance
    wbit_if intf(clk, rst_n);
    
    // Map Ports to Interface
    assign intf.cfg_req_valid   = bus_cfg_req_valid;
    assign intf.cfg_req_rw      = bus_cfg_req_rw;
    assign intf.cfg_req_addr    = bus_cfg_req_addr;
    assign intf.cfg_req_data    = bus_cfg_req_data;
    
    assign bus_cfg_resp_valid   = intf.cfg_resp_valid;
    assign bus_cfg_resp_data    = intf.cfg_resp_data;
    
    assign intf.data_in_valid   = bus_data_in_valid;
    assign intf.data_in_last    = bus_data_in_last;
    assign intf.data_in_payload = bus_data_in_payload;
    
    assign bus_data_out_valid   = intf.data_out_valid;
    assign bus_data_out_last    = intf.data_out_last;
    assign bus_data_out_payload = intf.data_out_payload;
    
    assign intf.op_start        = bus_op_start;
    assign intf.op_learn        = bus_op_learn; // Map it
    assign intf.op_patch_id     = bus_op_patch_id;
    assign intf.op_layer_id     = bus_op_layer_id;
    
    assign bus_op_busy          = intf.op_busy;
    assign bus_op_done          = intf.op_done;

    // Instantiate Accelerator
    wbit_accelerator #(
        .NUM_TILES(NUM_TILES),
        .TILE_ROWS(TILE_ROWS),
        .TILE_COLS(TILE_COLS)
    ) u_core (
        .clk(clk),
        .rst_n(rst_n),
        .bus(intf.device)
    );

endmodule