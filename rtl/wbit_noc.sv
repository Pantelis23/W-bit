// W-bit Network-on-Chip (Ring Topology)
// Wraps wbit_accelerator to allow scaling.
// Features: Ingress Routing, Egress Arbitration (Forwarding > Local).

module wbit_noc #(
    parameter NODE_ID = 0,
    parameter NUM_NODES = 4
) (
    input logic clk,
    input logic rst_n,
    
    // Ring Interface (Inbound)
    input  logic [255:0] ring_in_data,
    input  logic         ring_in_valid,
    input  logic [7:0]   ring_in_dest,
    
    // Ring Interface (Outbound)
    output logic [255:0] ring_out_data,
    output logic         ring_out_valid,
    output logic [7:0]   ring_out_dest,
    
    // Host Injection (Optional, usually for Node 0)
    input  logic         host_inject_valid,
    input  logic [255:0] host_inject_data,
    input  logic [7:0]   host_inject_dest
);

    // -------------------------------------------------------------------------
    // Local Accelerator
    // -------------------------------------------------------------------------
    wbit_if local_bus(clk, rst_n);
    
    wbit_accelerator #(
        .NUM_TILES(1)
    ) u_accel (
        .clk(clk),
        .rst_n(rst_n),
        .bus(local_bus.device)
    );

    // -------------------------------------------------------------------------
    // Buffers & Arbitration
    // -------------------------------------------------------------------------
    // Packet types: 
    // 1. Forwarding (transit) - Priority High
    // 2. Local Egress (result) - Priority Low
    // 3. Host Injection - Priority Medium
    
    logic        forward_req;
    logic [255:0] forward_data;
    logic [7:0]   forward_dest;
    
    logic        local_req;
    logic [255:0] local_data;
    logic [7:0]   local_dest;
    
    // Ingress Logic
    always_comb begin
        forward_req = 0;
        forward_data = '0;
        forward_dest = '0;
        
        local_bus.data_in_valid = 0;
        local_bus.data_in_payload = '0;
        local_bus.op_start = 0;
        
        if (ring_in_valid) begin
            if (ring_in_dest == NODE_ID) begin
                // Consumed by us
                local_bus.data_in_valid = 1;
                local_bus.data_in_payload = ring_in_data;
                local_bus.op_start = 1; // Auto-trigger (simplified)
            end else begin
                // Forward it
                forward_req = 1;
                forward_data = ring_in_data;
                forward_dest = ring_in_dest;
            end
        end
    end
    
    // Egress Logic (Buffer local result)
    // The accelerator streams data out. We must buffer it if the ring is busy.
    // For simplicity, we assume we can stall the accelerator (bus.op_busy logic needs update for backpressure).
    // Here we just capture valid output.
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            local_req <= 0;
            local_data <= '0;
            local_dest <= 0;
        end else begin
            if (local_bus.data_out_valid) begin
                local_req <= 1;
                local_data <= local_bus.data_out_payload;
                local_dest <= 0; // Default: Send results back to Host (Node 0)
            end else if (ring_out_valid && !forward_req && !host_inject_valid) begin
                // We sent it
                local_req <= 0; 
            end
        end
    end

    // Arbiter (Strict Priority: Forward > Host > Local)
    always_comb begin
        ring_out_valid = 0;
        ring_out_data = '0;
        ring_out_dest = '0;
        
        if (forward_req) begin
            ring_out_valid = 1;
            ring_out_data = forward_data;
            ring_out_dest = forward_dest;
        end else if (host_inject_valid) begin
            ring_out_valid = 1;
            ring_out_data = host_inject_data;
            ring_out_dest = host_inject_dest;
        end else if (local_req) begin
            ring_out_valid = 1;
            ring_out_data = local_data;
            ring_out_dest = local_dest;
        end
    end

endmodule