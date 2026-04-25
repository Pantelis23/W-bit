// =============================================================================
// wbit_tile.sv  —  Weight-Stationary Pipelined MAC Array
// =============================================================================
//
// ARCHITECTURE
// ------------
// Row-sliced rolling accumulator — the exact compute primitive of a physical
// ReRAM CIM macro:
//
//   Each cycle, the row_cnt word-line is asserted and all COLS column
//   conductances are sensed simultaneously (Kirchhoff's current law).
//   The row counter IS the wavefront propagating from row 0 to ROWS-1.
//
// TWO-STAGE PIPELINE  (original 26-level critical path → 2 stages)
// -----------------------------------------------------------------
//
//   Cycle r  ──► Stage 1 (MULTIPLY, registered in g_pe.always_ff)
//                  mul_pipe[j] ← vector_x[r] × w[r][j]
//                  Critical path: one (8b × 16b) multiply ≈ 4 LUT levels
//
//   Cycle r+1 ──► Stage 2 (ACCUMULATE, registered in g_pe.always_ff)
//                  acc[j] ← acc[j] + sign_ext(mul_pipe[j])
//                  Critical path: one 32b add ≈ 2 LUT levels
//
//   Both stages run simultaneously every cycle in a pipelined overlap.
//   Stage 2 is gated off on the very first RUNNING cycle (row_cnt==0)
//   because mul_pipe has not yet been filled.
//
// COMPUTE LATENCY  (from compute_en rising edge to result_valid rising edge)
// --------------------------------------------------------------------------
//   Cycle  0     : FSM transitions S_IDLE → S_RUNNING; acc seeded with noise
//   Cycles 1..256: S_RUNNING — Stage 1 fires rows 0..255; Stage 2 accumulates
//                  rows 0..254 (lags one cycle behind Stage 1)
//   Cycle  257   : S_DRAIN — Stage 2 absorbs row 255's product
//   Cycle  258   : S_LATCH — result_reg ← acc; result_valid pulsed
//   Total        : ROWS + 2 = 258 cycles
//
// STRUCTURAL NOTE — BLKLOOPINIT compliance
// -----------------------------------------
//   IEEE-1800 synthesis and strict lint tools do not allow delayed
//   assignments (<=) to arrays inside for loops within always_ff.
//   Per-column pipeline registers (mul_pipe, acc, result_reg slices) are
//   therefore instantiated via a generate block (g_pe), each iteration
//   containing its own always_ff.  The FSM always_ff owns only scalar
//   control signals.  This is fully legal IEEE-1800 and synthesises
//   identically to the equivalent for-loop version.
//
// INTERFACE CHANGES vs. BEHAVIOURAL MODEL
// ----------------------------------------
//   ADDED:   compute_ready (output) — HIGH when idle, safe to pulse compute_en.
//   REMOVED: nothing — all existing ports preserved.
//   wbit_accelerator.sv requires only the addition of the compute_ready wire.
//
// SYNTHESIS NOTES
// ---------------
//   • (* ram_style = "block" *) targets Xilinx/Intel BRAM; Yosys ignores.
//   • cfg_en should only be asserted while compute_ready is HIGH.
//   • update_en is independently safe (serialised by the accelerator FSM).
// =============================================================================

module wbit_tile #(
    parameter int ROWS         = 256,
    parameter int COLS         = 256,
    parameter int CELL_BITS    = 8,    // weight precision (ReRAM conductance bits)
    parameter int ACCUM_BITS   = 32,   // accumulator precision
    parameter int NOISE_ENABLE = 1     // 0 = ideal, 1 = LFSR read-noise floor
) (
    input  logic clk,
    input  logic rst_n,

    // ── Configuration  (weight-cell read / write) ──────────────────────────
    input  logic                  cfg_en,
    input  logic                  cfg_we,
    input  logic [7:0]            cfg_row_addr,
    input  logic [7:0]            cfg_col_addr,
    input  logic [CELL_BITS-1:0]  cfg_wdata,
    output logic [CELL_BITS-1:0]  cfg_rdata,

    // ── Compute  ───────────────────────────────────────────────────────────
    input  logic                  compute_en,    // 1-cycle pulse starts MVM
    output logic                  compute_ready, // HIGH = idle, safe to trigger
    input  logic [ROWS*16-1:0]    vector_x_flat, // full activation vector (held stable)

    // ── Online weight update  ──────────────────────────────────────────────
    input  logic                        update_en,
    input  logic [7:0]                  update_row_idx,
    input  logic [15:0]                 update_x_val,
    input  logic [COLS*ACCUM_BITS-1:0]  update_y_flat,
    input  logic [4:0]                  update_lr_shift,

    // ── Result  ────────────────────────────────────────────────────────────
    output logic                        result_valid,   // 1-cycle pulse
    output logic [COLS*ACCUM_BITS-1:0]  vector_y_flat
);

// =============================================================================
// Local parameters
// =============================================================================

localparam int ROW_BITS = $clog2(ROWS);         // 8 for ROWS=256
localparam int MUL_BITS = CELL_BITS + 16;        // 24: exact (8×16) product width
localparam int EXT_BITS = ACCUM_BITS - MUL_BITS; // 8: sign-extension padding

localparam logic [ROW_BITS-1:0] ROW_LAST = ROW_BITS'(ROWS - 1); // 8'd255

// =============================================================================
// FSM state encoding
// =============================================================================

typedef enum logic [1:0] {
    S_IDLE    = 2'd0,
    S_RUNNING = 2'd1,
    S_DRAIN   = 2'd2,
    S_LATCH   = 2'd3
} state_t;

state_t state;

// =============================================================================
// Weight memory  (synthesis infers block RAM via attribute)
// =============================================================================

(* ram_style = "block" *)
reg [COLS*CELL_BITS-1:0] mem [ROWS];

// =============================================================================
// Control registers  (driven by FSM always_ff only)
// =============================================================================

logic [ROW_BITS-1:0] row_cnt; // wavefront row pointer 0..ROWS-1

// =============================================================================
// Input vector unpack  (combinational mux; stable for full 258-cycle window)
// =============================================================================

logic signed [15:0] vector_x [ROWS];

always_comb
    for (int i = 0; i < ROWS; i++)
        vector_x[i] = signed'(vector_x_flat[i*16 +: 16]);

// =============================================================================
// Active-row weight unpack  (combinational mux on registered row_cnt)
// Corresponds to asserting the word-line and reading column conductances.
// =============================================================================

logic signed [CELL_BITS-1:0] w_cur [COLS];

always_comb
    for (int j = 0; j < COLS; j++)
        w_cur[j] = signed'(mem[row_cnt][j*CELL_BITS +: CELL_BITS]);

// =============================================================================
// Noise generator  (LFSR — free-running, sampled at each compute_en)
// =============================================================================

logic [15:0] rand_val;
wbit_lfsr u_noise_gen (
    .clk     (clk),
    .rst_n   (rst_n),
    .enable  (1'b1),
    .rand_out(rand_val)
);

// Per-column noise seed: XOR tap provides decorrelated phases from one LFSR word.
// Produces an 8-bit unsigned offset in [0, 31] — models ADC read-noise pedestal.
// Repeats with period 16 across columns (adequate for a noise-floor model).
logic [7:0] noise_seed [COLS];
generate
    for (genvar j = 0; j < COLS; j++) begin : g_noise_seed
        // j & 15 is compile-time constant per genvar iteration → constant bit-select
        // 4-bit XOR zero-padded to 8 bits (top nibble = 0, value ∈ [0,15])
        assign noise_seed[j] = {4'b0000,
                                 rand_val[3:0] ^ {4{rand_val[j & 15]}}};
    end
endgenerate

// =============================================================================
// Update engine  (Hebbian row update — combinational, unchanged)
// =============================================================================

logic [COLS*CELL_BITS-1:0] current_row_flat;
logic [COLS*CELL_BITS-1:0] updated_row_flat;

assign current_row_flat = mem[update_row_idx];

wbit_update_engine #(
    .COLS      (COLS),
    .CELL_BITS (CELL_BITS),
    .ACCUM_BITS(ACCUM_BITS)
) u_update (
    .x_val         (update_x_val),
    .y_vec_flat    (update_y_flat),
    .w_row_in_flat (current_row_flat),
    .lr_shift      (update_lr_shift),
    .w_row_out_flat(updated_row_flat)
);

// =============================================================================
// Per-column pipeline registers  (g_pe generate block)
// =============================================================================
//
//   Each iteration of g_pe is one Processing Element (PE) for column j.
//   Two always_ff blocks per PE — one per pipeline stage — give the
//   synthesiser maximum scheduling freedom and satisfy Verilator's
//   BLKLOOPINIT rule (no delayed assignments to arrays inside for loops).
//
//   Sign-extension (combinational, g_sign_ext) sits between Stage 1 and
//   Stage 2 and costs no register resources.
//
// =============================================================================

// Stage 1 outputs — registered 24-bit signed products
logic signed [MUL_BITS-1:0]  mul_pipe  [COLS];

// Sign-extension of Stage 1 output for the 32-bit accumulate
logic signed [ACCUM_BITS-1:0] mul_pipe_ext [COLS];

// Stage 2 — column partial-sum accumulators
logic signed [ACCUM_BITS-1:0] acc [COLS];

// Output result register (packed flat bus)
logic [COLS*ACCUM_BITS-1:0] result_reg;
assign vector_y_flat = result_reg;

generate
    for (genvar j = 0; j < COLS; j++) begin : g_pe

        // ------------------------------------------------------------------
        // Sign extension of Stage 1 product (combinational)
        // ------------------------------------------------------------------
        assign mul_pipe_ext[j] = {{EXT_BITS{mul_pipe[j][MUL_BITS-1]}},
                                            mul_pipe[j]};

        // ------------------------------------------------------------------
        // Stage 1 — Multiply  (fires every cycle in S_RUNNING)
        //
        //   PE[j].Stage1: mul_pipe[j] ← vector_x[row_cnt] × w_cur[j]
        //
        //   Physical analogue: column j's sense-amp output sampled for the
        //   currently selected word-line.
        // ------------------------------------------------------------------
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                mul_pipe[j] <= '0;
            end else if (state == S_RUNNING) begin
                mul_pipe[j] <= vector_x[row_cnt] * w_cur[j];
            end
        end

        // ------------------------------------------------------------------
        // Stage 2 — Accumulate  (fires in S_RUNNING from cycle 2 onward,
        //                        plus once in S_DRAIN to flush last product)
        //
        //   PE[j].Stage2: acc[j] ← acc[j] + mul_pipe_ext[j]
        //
        //   Gate on row_cnt != 0 during S_RUNNING: Stage 1 has not yet
        //   produced a valid product on the very first RUNNING cycle.
        //
        //   S_IDLE + compute_en: seed acc with LFSR noise (models the
        //   ADC pedestal present at the start of every crossbar sense).
        //
        //   S_LATCH: pack acc into the flat output register.
        // ------------------------------------------------------------------
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                acc[j]                               <= '0;
                result_reg[j*ACCUM_BITS +: ACCUM_BITS] <= '0;
            end else begin
                case (state)
                    // Seed accumulators when a new compute begins
                    S_IDLE: begin
                        if (compute_en) begin
                            acc[j] <= (NOISE_ENABLE != 0)
                                      ? ACCUM_BITS'(noise_seed[j])
                                      : '0;
                        end
                    end

                    // Accumulate Stage 2 (skip cycle 0 — mul_pipe not yet valid)
                    S_RUNNING: begin
                        if (row_cnt != '0)
                            acc[j] <= acc[j] + mul_pipe_ext[j];
                    end

                    // Flush the last multiply result (row ROWS-1)
                    S_DRAIN: begin
                        acc[j] <= acc[j] + mul_pipe_ext[j];
                    end

                    // Capture final accumulator into output register
                    S_LATCH: begin
                        result_reg[j*ACCUM_BITS +: ACCUM_BITS] <= acc[j];
                    end

                    default: ; // S_LATCH handled above; no action on default
                endcase
            end
        end

    end
endgenerate

// =============================================================================
// FSM + scalar control  (owns: state, row_cnt, compute_ready,
//                               result_valid, cfg_rdata, mem writes)
// =============================================================================
//
//   S_IDLE ──(compute_en)──► S_RUNNING ──(row_cnt==ROW_LAST)──► S_DRAIN
//     ▲                                                              │
//     └──────────────────── S_LATCH ◄──────────────────────────────┘
//
// =============================================================================

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state         <= S_IDLE;
        row_cnt       <= '0;
        compute_ready <= 1'b1;
        result_valid  <= 1'b0;
        cfg_rdata     <= '0;
    end else begin

        // result_valid is a single-cycle pulse — de-assert by default
        result_valid <= 1'b0;

        // -- Configuration port (safe while compute_ready is HIGH) ----------
        if (cfg_en) begin
            if (cfg_we)
                mem[cfg_row_addr][cfg_col_addr*CELL_BITS +: CELL_BITS] <= cfg_wdata;
            cfg_rdata <= mem[cfg_row_addr][cfg_col_addr*CELL_BITS +: CELL_BITS];
        end

        // -- Online weight update (serialised by accelerator FSM) -----------
        if (update_en)
            mem[update_row_idx] <= updated_row_flat;

        // -- Compute FSM ----------------------------------------------------
        case (state)

            // ----------------------------------------------------------------
            // S_IDLE: quiescent — tile ready to accept a new MVM request
            // ----------------------------------------------------------------
            S_IDLE: begin
                compute_ready <= 1'b1;
                row_cnt       <= '0;
                if (compute_en) begin
                    compute_ready <= 1'b0;
                    state         <= S_RUNNING;
                    // (acc seeding handled in g_pe S_IDLE branch above)
                end
            end

            // ----------------------------------------------------------------
            // S_RUNNING: the multiply wavefront traverses rows 0 → ROWS-1
            //
            //   Every cycle advances row_cnt, firing Stage 1 in all COLS PEs.
            //   Stage 2 in each PE absorbs the previous cycle's Stage 1 output
            //   (gated off on row_cnt==0 — the pipeline fill cycle).
            // ----------------------------------------------------------------
            S_RUNNING: begin
                if (row_cnt == ROW_LAST) begin
                    // All rows have passed through Stage 1.
                    // row ROWS-1's product is in mul_pipe — flush in S_DRAIN.
                    state   <= S_DRAIN;
                    row_cnt <= '0;
                end else begin
                    row_cnt <= row_cnt + 1'b1;
                end
            end

            // ----------------------------------------------------------------
            // S_DRAIN: one extra cycle to flush the last multiply into acc
            // ----------------------------------------------------------------
            S_DRAIN: begin
                state <= S_LATCH;
            end

            // ----------------------------------------------------------------
            // S_LATCH: signal host; tile returns to IDLE next cycle
            // ----------------------------------------------------------------
            S_LATCH: begin
                result_valid  <= 1'b1;
                compute_ready <= 1'b1;
                state         <= S_IDLE;
            end

            default: state <= S_IDLE;

        endcase
    end
end

endmodule
