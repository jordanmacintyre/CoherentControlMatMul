//-----------------------------------------------------------------------------
// pkg_types.sv
// Type definitions and parameters for coherent photonic matrix multiply
//-----------------------------------------------------------------------------

package pkg_types;

    //-------------------------------------------------------------------------
    // Data width parameters
    //-------------------------------------------------------------------------
    parameter int DATA_WIDTH   = 16;    // Q1.15 for weights and inputs
    parameter int PHASE_WIDTH  = 16;    // Q2.14 for phase DAC codes
    parameter int ADC_WIDTH    = 12;    // ADC resolution
    parameter int ACC_WIDTH    = 32;    // Accumulator width for error
    parameter int NUM_PHASES   = 4;     // Number of phase shifters
    parameter int NUM_OUTPUTS  = 2;     // Number of output channels

    //-------------------------------------------------------------------------
    // Fixed-point format constants
    //-------------------------------------------------------------------------
    // Q1.15: 1 sign bit, 0 integer bits, 15 fractional bits
    // Range: [-1, 1), Resolution: 2^-15 = 3.05e-5
    parameter int Q1_15_FRAC_BITS = 15;
    parameter logic [15:0] Q1_15_ONE     = 16'h7FFF;  // +0.99997 (max positive)
    parameter logic [15:0] Q1_15_NEG_ONE = 16'h8000;  // -1.0

    // Q2.14: 1 sign bit, 1 integer bit, 14 fractional bits
    // For phase: [0, 2*pi) mapped to [0, 2^16)
    parameter int Q2_14_FRAC_BITS = 14;

    //-------------------------------------------------------------------------
    // Calibration parameters
    //-------------------------------------------------------------------------
    parameter int unsigned CAL_SETTLE_CYCLES  = 16;     // Cycles to wait after phase change
    parameter int unsigned CAL_AVG_SAMPLES    = 8;      // Samples to average per measurement
    parameter int unsigned CAL_MAX_ITERATIONS = 1000;   // Max calibration iterations
    parameter int unsigned CAL_LOCK_THRESHOLD = 32'h0000_0100;  // Error threshold for lock
    parameter int unsigned CAL_LOCK_COUNT     = 4;      // Consecutive iterations below threshold

    // Phase update step sizes (16-bit for PHASE_WIDTH)
    parameter logic [15:0] PHASE_STEP_INITIAL = 16'h1000;  // ~0.39 rad initial step
    parameter logic [15:0] PHASE_STEP_MIN     = 16'h0040;  // ~0.015 rad minimum step
    parameter int unsigned PHASE_STEP_DECAY   = 2;         // Shift right by 1 = divide by 2

    //-------------------------------------------------------------------------
    // Calibration FSM states
    //-------------------------------------------------------------------------
    typedef enum logic [3:0] {
        CAL_IDLE           = 4'd0,
        CAL_VALIDATE       = 4'd1,   // Validate input weights
        CAL_APPLY_BASIS0   = 4'd2,   // Apply x = [1, 0]
        CAL_SETTLE0        = 4'd3,   // Wait for thermal settling
        CAL_SAMPLE_COL0    = 4'd4,   // Sample column 0
        CAL_APPLY_BASIS1   = 4'd5,   // Apply x = [0, 1]
        CAL_SETTLE1        = 4'd6,   // Wait for thermal settling
        CAL_SAMPLE_COL1    = 4'd7,   // Sample column 1
        CAL_COMPUTE_ERROR  = 4'd8,   // Compute error metric
        CAL_PROBE_PLUS     = 4'd9,   // Try phi + delta
        CAL_PROBE_MINUS    = 4'd10,  // Try phi - delta
        CAL_UPDATE_PHASE   = 4'd11,  // Update phase based on probes
        CAL_CHECK_CONVERGE = 4'd12,  // Check convergence criteria
        CAL_LOCKED         = 4'd13,  // Calibration complete
        CAL_ERROR          = 4'd14   // Error state (invalid weights)
    } cal_state_t;

    //-------------------------------------------------------------------------
    // Top-level FSM states
    //-------------------------------------------------------------------------
    typedef enum logic [2:0] {
        TOP_IDLE      = 3'd0,
        TOP_CAL       = 3'd1,  // Calibration mode
        TOP_LOCKED    = 3'd2,  // Locked, ready for eval
        TOP_EVAL      = 3'd3,  // Evaluation mode
        TOP_ERROR     = 3'd4   // Error state
    } top_state_t;

    //-------------------------------------------------------------------------
    // Plant interface mode
    //-------------------------------------------------------------------------
    typedef enum logic [1:0] {
        PLANT_IDLE    = 2'd0,
        PLANT_CAL     = 2'd1,
        PLANT_EVAL    = 2'd2
    } plant_mode_t;

    //-------------------------------------------------------------------------
    // Status flags
    //-------------------------------------------------------------------------
    typedef struct packed {
        logic cal_locked;       // [7] Calibration locked
        logic cal_in_progress;  // [6] Calibration in progress
        logic eval_done;        // [5] Evaluation complete
        logic error_weights;    // [4] Invalid weights error
        logic error_timeout;    // [3] Calibration timeout
        logic error_saturated;  // [2] ADC saturation detected
        logic reserved1;        // [1] Reserved
        logic reserved0;        // [0] Reserved
    } status_flags_t;

endpackage
