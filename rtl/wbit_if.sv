// W-bit Accelerator Interface
// Defines the protocol between the Aeternum Core and the W-bit Fabric.

/* verilator lint_off UNUSEDSIGNAL */
interface wbit_if (input logic clk, input logic rst_n);
    // -------------------------------------------------------------------------
    // Control Plane (Configuration & Patch Loading)
    // -------------------------------------------------------------------------
    logic        cfg_req_valid;
    logic        cfg_req_rw;      // 0=Read, 1=Write
    logic [31:0] cfg_req_addr;    // Address in PatchDB/Config space
    logic [31:0] cfg_req_data;    // Data to write
    
    logic        cfg_resp_valid;
    logic [31:0] cfg_resp_data;   // Data read
    
    // -------------------------------------------------------------------------
    // Data Plane (Vector Operations)
    // -------------------------------------------------------------------------
    // Input Vector (x) Broadcast
    
    logic        data_in_valid;
    logic        data_in_last;    // Indicates last chunk of vector
    logic [255:0] data_in_payload; // 16 x 16-bit
    
    // Output Vector (y) Reduction
    logic        data_out_valid;
    logic        data_out_last;
    logic [255:0] data_out_payload;
    
    // -------------------------------------------------------------------------
    // Operation Control
    // -------------------------------------------------------------------------
    // Command to trigger compute (e.g., "Run Layer 1 with Patch ID 5")
    logic        op_start;
    logic        op_learn;        // TRIGGER SELF-ADJUSTMENT (Update weights)
    logic [15:0] op_patch_id;     // Active patch ID (WZMA)
    logic [15:0] op_layer_id;     // Target layer index
    logic        op_busy;         // Accelerator is busy
    logic        op_done;         // Computation finished

    // Modports
    modport host (
        output cfg_req_valid, cfg_req_rw, cfg_req_addr, cfg_req_data,
        input  cfg_resp_valid, cfg_resp_data,
        
        output data_in_valid, data_in_last, data_in_payload,
        input  data_out_valid, data_out_last, data_out_payload,
        
        output op_start, op_learn, op_patch_id, op_layer_id,
        input  op_busy, op_done
    );

    modport device (
        input  cfg_req_valid, cfg_req_rw, cfg_req_addr, cfg_req_data,
        output cfg_resp_valid, cfg_resp_data,
        
        input  data_in_valid, data_in_last, data_in_payload,
        output data_out_valid, data_out_last, data_out_payload,
        
        input  op_start, op_learn, op_patch_id, op_layer_id,
        output op_busy, op_done
    );

endinterface
/* verilator lint_on UNUSEDSIGNAL */