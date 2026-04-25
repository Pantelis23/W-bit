// W-bit Accelerator Top Level
// Connects Aeternum Core (via wbit_if) to the W-bit Fabric.
// Supports Forward (Compute) and Backward (Self-Adjusting) modes.

module wbit_accelerator #(
    /* verilator lint_off UNUSEDPARAM */
    parameter NUM_TILES = 1, 
    /* verilator lint_on UNUSEDPARAM */
    parameter TILE_ROWS = 256,
    parameter TILE_COLS = 256
) (
    input logic clk,
    input logic rst_n,
    wbit_if.device bus
);

    // -------------------------------------------------------------------------
    // Control Plane (Configuration)
    // -------------------------------------------------------------------------
    
    logic        tile_cfg_en;
    logic        tile_cfg_we;
    logic [7:0]  tile_cfg_row;
    logic [7:0]  tile_cfg_col;
    logic [7:0]  tile_cfg_wdata;
    logic [7:0]  tile_cfg_rdata;
    
    always_comb begin
        bus.cfg_resp_valid = bus.cfg_req_valid; 
        
        tile_cfg_en    = bus.cfg_req_valid;
        tile_cfg_we    = bus.cfg_req_rw;
        tile_cfg_row   = bus.cfg_req_addr[15:8];
        tile_cfg_col   = bus.cfg_req_addr[7:0];
        tile_cfg_wdata = bus.cfg_req_data[7:0];
        
        bus.cfg_resp_data = {24'b0, tile_cfg_rdata};
    end

    // -------------------------------------------------------------------------
    // Data Plane (Buffers)
    // -------------------------------------------------------------------------
    
    logic [TILE_ROWS-1:0][15:0] input_buffer; // Stores x for compute AND update
    
    // -------------------------------------------------------------------------
    // State Machine
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE,
        LOAD_INPUT,
        COMPUTE,
        SEND_OUTPUT,
        LEARN_UPDATE
    } state_t;
    
    state_t state;
    logic [7:0] update_row_counter;
    logic [3:0] input_chunk_idx;
    logic [4:0] output_chunk_idx;
    
    // Tile Signals
    logic tile_compute_en;
    logic tile_compute_ready; // NEW: tile idle and safe to trigger
    logic tile_update_en;
    logic tile_result_valid;
    logic [TILE_COLS-1:0][31:0] tile_result_y;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            input_chunk_idx <= 0;
            output_chunk_idx <= 0;
            update_row_counter <= 0;
            bus.op_busy <= 0;
            bus.op_done <= 0;
            bus.data_out_valid <= 0;
            bus.data_out_last <= 0;
            
            tile_compute_en <= 0;
            tile_update_en <= 0;
        end else begin
            case (state)
                IDLE: begin
                    bus.op_done <= 0;
                    if (bus.op_start) begin
                        state <= LOAD_INPUT;
                        bus.op_busy <= 1;
                        input_chunk_idx <= 0;
                    end else if (bus.op_learn) begin
                        state <= LEARN_UPDATE;
                        bus.op_busy <= 1;
                        update_row_counter <= 0;
                        tile_update_en <= 1;
                    end
                end
                
                LOAD_INPUT: begin
                    if (bus.data_in_valid) begin
                        input_buffer[input_chunk_idx * 16 +: 16] <= bus.data_in_payload;
                        if (bus.data_in_last || input_chunk_idx == 15) begin
                            // Guard: only trigger if tile is idle (compute_ready HIGH).
                            // In normal sequencing this is always true; the guard prevents
                            // a second MVM from corrupting an in-flight pipeline.
                            if (tile_compute_ready) begin
                                state           <= COMPUTE;
                                tile_compute_en <= 1;
                            end
                            // else: stall in LOAD_INPUT until tile is free
                        end else begin
                            input_chunk_idx <= input_chunk_idx + 1;
                        end
                    end
                end

                COMPUTE: begin
                    tile_compute_en <= 0; // Pulse — tile latches on the next rising edge
                    // Tile pipeline latency: ROWS+2 = 258 cycles until result_valid.
                    //   1 cycle  S_IDLE  -> S_RUNNING transition
                    //   256 cycles S_RUNNING (rows 0..255 through multiply stage)
                    //   1 cycle  S_DRAIN (flush last product into accumulator)
                    //   1 cycle  S_LATCH (capture + pulse result_valid)
                    // State machine simply waits here; no polling needed.
                    if (tile_result_valid) begin
                        state <= SEND_OUTPUT;
                        output_chunk_idx <= 0;
                    end
                end
                
                SEND_OUTPUT: begin
                    bus.data_out_valid <= 1;
                    bus.data_out_payload <= tile_result_y[output_chunk_idx * 8 +: 8];
                    
                    if (output_chunk_idx == 31) begin
                        bus.data_out_last <= 1;
                        state <= IDLE;
                        bus.op_done <= 1;
                        bus.op_busy <= 0;
                    end else begin
                        bus.data_out_last <= 0;
                        output_chunk_idx <= output_chunk_idx + 1;
                    end
                end
                
                LEARN_UPDATE: begin
                    if (update_row_counter == 255) begin
                        state <= IDLE;
                        bus.op_done <= 1;
                        bus.op_busy <= 0;
                        tile_update_en <= 0;
                        update_row_counter <= 0;
                    end else begin
                        update_row_counter <= update_row_counter + 1;
                    end
                end
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Tile Instance
    // -------------------------------------------------------------------------
    
    // Casting for flattened ports
    // SystemVerilog treats [A][B] as [A*B-1:0] implicitly for packed arrays
    
    wbit_tile #(
        .ROWS(TILE_ROWS),
        .COLS(TILE_COLS)
    ) u_tile (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_en(tile_cfg_en),
        .cfg_we(tile_cfg_we),
        .cfg_row_addr(tile_cfg_row),
        .cfg_col_addr(tile_cfg_col),
        .cfg_wdata(tile_cfg_wdata),
        .cfg_rdata(tile_cfg_rdata),
        
        .compute_en   (tile_compute_en),
        .compute_ready(tile_compute_ready),
        .vector_x_flat(input_buffer), // Implicit Cast
        
        .update_en(tile_update_en),
        .update_row_idx(update_row_counter),
        .update_x_val(input_buffer[update_row_counter]),
        .update_y_flat(tile_result_y), // Implicit Cast
        .update_lr_shift(5'd20),
        
        .result_valid(tile_result_valid),
        .vector_y_flat(tile_result_y)  // Implicit Cast
    );

endmodule
