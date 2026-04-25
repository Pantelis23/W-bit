// =============================================================================
// wbit_copro_wrapper.sv  —  Aeternum Coprocessor -> W-bit Accelerator Bridge
// =============================================================================
//
// BUS GEOMETRY  (why 16 chunks are required)
// -------------------------------------------
//   Aeternum VRF bus width    : W = 256 bits = 16 x 16-bit activations / txn
//   W-bit tile input vector   : TILE_ROWS x 16 bits = 256 x 16 = 4096 bits
//   Required transactions     : N_CHUNKS = 4096 / 256 = 16
//
//   The original single-transaction shortcut (data_in_last=1 on cycle 0)
//   delivered only 1/16 of the input data.  Worse, it fired data_in_valid
//   while the accelerator was still in IDLE (before it had transitioned to
//   LOAD_INPUT), so in practice zero data was ever loaded.
//
// FORWARD PASS PROTOCOL  (F7_WBIT_FWD = 0x30)
// ---------------------------------------------
//   Step 1 — Issue command   : cmd_valid=1, funct7=F7_WBIT_FWD
//     Wrapper: asserts op_start (1 cycle, START state).
//     Accelerator: IDLE -> LOAD_INPUT.
//
//   Steps 2-17 — Deliver data: cmd_valid=1 (x16, any funct7)
//     Wrapper: asserts data_in_valid + data_in_payload for each chunk.
//     cmd_ready=1 throughout LOAD_CHUNKS so Aeternum can insert wait cycles.
//     data_in_last is asserted ONLY on chunk 15.
//
//   Step 18-onwards — Compute: 258 cycles (tile pipeline, WAIT_COMPUTE state)
//     Wrapper: cmd_ready=0, waiting for bus.op_done.
//
// CRITICAL TIMING FIX: START state
// ----------------------------------
//   op_start and data_in_valid must NOT be asserted simultaneously.
//   op_start causes accelerator IDLE->LOAD_INPUT at the END of that cycle.
//   data_in_valid is only meaningful when the accelerator IS in LOAD_INPUT,
//   i.e., from the NEXT cycle onward.
//
//   The 1-cycle START state is the register boundary that enforces this.
//
// OUTPUT LIMITATION  (known, deferred to Phase A3)
// -------------------------------------------------
//   vd_data is W=256 bits wide, holding 8 x 32-bit results.
//   The tile produces 256 x 32-bit = 8192-bit results over 32 output chunks.
//   Only the final chunk (results 248..255) is captured here, coincident
//   with bus.op_done (the accelerator asserts op_done on the last chunk).
//   Full result readout requires an output streaming protocol.
//
// =============================================================================

module wbit_copro_wrapper #(
    parameter int W         = 256, // VRF bus width in bits (16 x 16-bit per txn)
    parameter int TILE_ROWS = 256, // must match wbit_accelerator TILE_ROWS
    parameter int TILE_COLS = 256  // must match wbit_accelerator TILE_COLS
) (
    input  logic clk,
    input  logic rst_n,

    // ── Aeternum coprocessor interface (simplified subset) ─────────────────
    input  logic         cmd_valid,   // command or data-chunk strobe from Aeternum
    input  logic [6:0]   cmd_funct7,  // operation selector
    input  logic [2:0]   cmd_vd,      // destination VRF register (unused here)
    input  logic [W-1:0] vs1_data,    // activation chunk from VRF output port
    output logic         cmd_ready,   // HIGH: wrapper can accept cmd_valid this cycle
    output logic         done_pulse,  // 1-cycle pulse when operation completes
    output logic [W-1:0] vd_data      // result chunk (see output limitation above)
);

    // =========================================================================
    // Chunking constants
    // =========================================================================

    // Number of bus transactions to deliver one full activation vector.
    localparam int CHUNKS   = (TILE_ROWS * 16) / W; // 16 for default parameters
    localparam int CHK_BITS = $clog2(CHUNKS);        // 4

    // =========================================================================
    // W-bit accelerator bus + instance
    // =========================================================================

    /* verilator lint_off UNUSEDSIGNAL */
    wbit_if bus(clk, rst_n);
    /* verilator lint_on UNUSEDSIGNAL */

    wbit_accelerator #(
        .NUM_TILES(1),
        .TILE_ROWS(TILE_ROWS),
        .TILE_COLS(TILE_COLS)
    ) u_wbit (
        .clk  (clk),
        .rst_n(rst_n),
        .bus  (bus.device)
    );

    // =========================================================================
    // Opcodes
    // =========================================================================

    localparam logic [6:0] F7_WBIT_FWD = 7'h30; // forward MVM pass
    localparam logic [6:0] F7_WBIT_LRN = 7'h31; // Hebbian weight update
    /* verilator lint_off UNUSEDPARAM */
    localparam logic [6:0] F7_WBIT_CFG = 7'h32; // config write (reserved for Phase A3)
    /* verilator lint_on UNUSEDPARAM */

    // =========================================================================
    // FSM
    // =========================================================================
    //
    //   FWD path:
    //     IDLE ---(cmd_valid, FWD)---> START ---> LOAD_CHUNKS(x16) ---> WAIT_COMPUTE
    //      ^                                                                  |
    //      +------------------------------------------------------------------+
    //
    //   LRN path:
    //     IDLE ---(cmd_valid, LRN)---> WAIT_COMPUTE ---> IDLE
    //
    // =========================================================================

    typedef enum logic [1:0] {
        IDLE         = 2'd0,
        START        = 2'd1, // 1 cycle: op_start pulse; no data
        LOAD_CHUNKS  = 2'd2, // 16 cycles (+ optional wait): stream activation chunks
        WAIT_COMPUTE = 2'd3  // up to 258+ cycles: wait for tile + capture result
    } state_t;

    state_t state;

    logic [CHK_BITS-1:0] chunk_cnt; // next chunk index to send, 0..CHUNKS-1

    // =========================================================================
    // Datapath
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state               <= IDLE;
            chunk_cnt           <= '0;
            cmd_ready           <= 1'b0;
            done_pulse          <= 1'b0;
            vd_data             <= '0;
            // Host-driven bus signals — explicit reset to avoid X-prop
            bus.op_start        <= 1'b0;
            bus.op_learn        <= 1'b0;
            bus.op_patch_id     <= '0;
            bus.op_layer_id     <= '0;
            bus.data_in_valid   <= 1'b0;
            bus.data_in_last    <= 1'b0;
            bus.data_in_payload <= '0;
            bus.cfg_req_valid   <= 1'b0;
            bus.cfg_req_rw      <= 1'b0;
            bus.cfg_req_addr    <= '0;
            bus.cfg_req_data    <= '0;
        end else begin

            // -- Defaults: de-assert all single-cycle strobes ----------------
            // Case branches below override as needed; last NBA wins.
            done_pulse          <= 1'b0;
            bus.op_start        <= 1'b0;
            bus.op_learn        <= 1'b0;
            bus.data_in_valid   <= 1'b0;
            bus.data_in_last    <= 1'b0;

            case (state)

                // ── IDLE ─────────────────────────────────────────────────
                // Quiescent: waiting for a command from the Aeternum core.
                IDLE: begin
                    cmd_ready <= 1'b1;
                    chunk_cnt <= '0;

                    if (cmd_valid) begin
                        cmd_ready <= 1'b0; // hold off until we're ready again

                        if (cmd_funct7 == F7_WBIT_FWD) begin
                            state <= START;

                        end else if (cmd_funct7 == F7_WBIT_LRN) begin
                            bus.op_learn <= 1'b1;
                            state        <= WAIT_COMPUTE;
                        end
                        // F7_WBIT_CFG: reserved for future use
                    end
                end

                // ── START ────────────────────────────────────────────────
                // Assert op_start for exactly one clock cycle.
                //
                // Physical meaning: the accelerator sees op_start=1, latches
                // it at the rising edge, and transitions IDLE -> LOAD_INPUT
                // at the END of this cycle.
                //
                // data_in_valid is NOT asserted here.  The accelerator is
                // still in IDLE this cycle and would ignore it.  Data begins
                // in LOAD_CHUNKS, where the accelerator is already in LOAD_INPUT.
                START: begin
                    bus.op_start <= 1'b1; // 1-cycle pulse (default de-asserts next cycle)
                    cmd_ready    <= 1'b0; // NOT ready — no capture logic in this state.
                    // cmd_ready goes HIGH on the first cycle of LOAD_CHUNKS, after the
                    // accelerator has completed its IDLE->LOAD_INPUT transition.
                    // A host responding to cmd_ready=1 in START would present chunk 0
                    // with no one listening — data would be silently lost.
                    state        <= LOAD_CHUNKS;
                end

                // ── LOAD_CHUNKS ──────────────────────────────────────────
                // Stream CHUNKS=16 activation chunks through the accelerator's
                // LOAD_INPUT state.
                //
                // cmd_ready=1: Aeternum may present vs1_data with cmd_valid
                // each cycle.  Inserting idle cycles (cmd_valid=0) is safe —
                // the accelerator waits in LOAD_INPUT until data arrives.
                //
                // Chunk delivery schedule:
                //   chunk 0  : activations x[0..15]   (bus bytes 0..31)
                //   chunk 1  : activations x[16..31]  (bus bytes 32..63)
                //   ...
                //   chunk 15 : activations x[240..255] (bus bytes 480..511)
                //
                // The accelerator's internal input_chunk_idx tracks position
                // independently — the wrapper only needs to count and flag last.
                LOAD_CHUNKS: begin
                    cmd_ready <= 1'b1; // remain open for the next chunk

                    if (cmd_valid) begin
                        bus.data_in_valid   <= 1'b1;
                        bus.data_in_payload <= vs1_data;
                        bus.data_in_last    <= (chunk_cnt == CHK_BITS'(CHUNKS - 1));

                        if (chunk_cnt == CHK_BITS'(CHUNKS - 1)) begin
                            // Last chunk delivered.  Accelerator will trigger compute.
                            cmd_ready <= 1'b0;
                            state     <= WAIT_COMPUTE;
                        end else begin
                            chunk_cnt <= chunk_cnt + 1'b1;
                        end
                    end
                end

                // ── WAIT_COMPUTE ─────────────────────────────────────────
                // The tile is running its 258-cycle pipelined MVM.
                // The accelerator emits 32 x 256-bit result chunks, then
                // asserts op_done on the last chunk.
                //
                // We capture the result payload coincident with op_done
                // (chunk 31 of 32 — output results y[248..255]).
                // Full output streaming (all 256 results) is deferred to Phase A3.
                WAIT_COMPUTE: begin
                    cmd_ready <= 1'b0;

                    if (bus.op_done) begin
                        vd_data    <= bus.data_out_valid ? bus.data_out_payload : '0;
                        done_pulse <= 1'b1;
                        cmd_ready  <= 1'b1;
                        state      <= IDLE;
                    end
                end

                default: state <= IDLE;

            endcase
        end
    end

endmodule
