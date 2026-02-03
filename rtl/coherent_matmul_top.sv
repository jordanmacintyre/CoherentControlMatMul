//-----------------------------------------------------------------------------
// coherent_matmul_top.sv
// Top-level module for coherent photonic 2x2 matrix multiply
//-----------------------------------------------------------------------------

module coherent_matmul_top
    import pkg_types::*;
#(
    parameter int DATA_WIDTH_P  = DATA_WIDTH,
    parameter int PHASE_WIDTH_P = PHASE_WIDTH,
    parameter int NUM_PHASES_P  = NUM_PHASES,
    parameter int ADC_WIDTH_P   = ADC_WIDTH,
    parameter int VOA_WIDTH_P   = VOA_WIDTH
)(
    input  logic                           clk,
    input  logic                           rst_n,

    // Target weights (Q1.15 fixed-point, range [-1, 1))
    input  logic signed [DATA_WIDTH_P-1:0] w0,
    input  logic signed [DATA_WIDTH_P-1:0] w1,
    input  logic signed [DATA_WIDTH_P-1:0] w2,
    input  logic signed [DATA_WIDTH_P-1:0] w3,

    // Input data (Q1.15 fixed-point, range [-1, 1))
    input  logic signed [DATA_WIDTH_P-1:0] x0,
    input  logic signed [DATA_WIDTH_P-1:0] x1,

    // Control signals
    input  logic                           start_cal,   // Start calibration
    input  logic                           start_eval,  // Start evaluation
    input  logic                           svd_mode,    // SVD mode (1) or unitary mode (0)

    // SVD mode inputs (pre-computed singular values as DAC codes)
    input  logic [VOA_WIDTH_P-1:0]         sigma_dac0,  // σ₀ VOA code
    input  logic [VOA_WIDTH_P-1:0]         sigma_dac1,  // σ₁ VOA code

    // Status outputs
    output logic                           cal_done,
    output logic                           cal_locked,
    output logic                           eval_done,
    output logic [7:0]                     status_flags,

    // Evaluation outputs (Q1.15 fixed-point)
    output logic signed [DATA_WIDTH_P-1:0] y0_out,
    output logic signed [DATA_WIDTH_P-1:0] y1_out,
    output logic                           y_valid,

    // Plant interface - outputs to plant (unitary mode / legacy)
    output logic [PHASE_WIDTH_P-1:0]       phi_dac [NUM_PHASES_P],
    output logic signed [DATA_WIDTH_P-1:0] x_drive0,
    output logic signed [DATA_WIDTH_P-1:0] x_drive1,
    output logic                           plant_enable,
    output logic [1:0]                     plant_mode,

    // SVD plant interface - extended outputs
    output logic [PHASE_WIDTH_P-1:0]       phi_dac_v [NUM_PHASES_V],  // V† mesh phases
    output logic [VOA_WIDTH_P-1:0]         voa_dac [NUM_VOAS],        // Σ attenuators
    output logic [PHASE_WIDTH_P-1:0]       phi_dac_u [NUM_PHASES_U],  // U mesh phases

    // Plant interface - inputs from plant
    input  logic signed [ADC_WIDTH_P-1:0]  adc_i0,
    input  logic signed [ADC_WIDTH_P-1:0]  adc_q0,
    input  logic signed [ADC_WIDTH_P-1:0]  adc_i1,
    input  logic signed [ADC_WIDTH_P-1:0]  adc_q1,
    input  logic                           adc_valid
);

    //-------------------------------------------------------------------------
    // Internal signals
    //-------------------------------------------------------------------------

    // State machines
    top_state_t top_state, top_state_next;
    cal_state_t cal_state, cal_state_next;
    svd_cal_state_t svd_cal_state, svd_cal_state_next;

    // Status flags structure
    status_flags_t status;

    // Phase registers (unitary mode)
    logic [PHASE_WIDTH_P-1:0] phi_reg [NUM_PHASES_P];
    logic [PHASE_WIDTH_P-1:0] phi_step;  // Current step size

    // SVD mode registers
    logic [PHASE_WIDTH_P-1:0] phi_reg_v [NUM_PHASES_V];  // V† mesh phases
    logic [PHASE_WIDTH_P-1:0] phi_reg_u [NUM_PHASES_U];  // U mesh phases
    logic [VOA_WIDTH_P-1:0]   voa_reg [NUM_VOAS];        // VOA codes
    svd_component_t           svd_component;             // Current component being calibrated
    logic                     v_locked_reg;
    logic                     u_locked_reg;

    // Measured matrix elements (from I/Q measurements)
    // m_hat[0] = M00, m_hat[1] = M01, m_hat[2] = M10, m_hat[3] = M11
    logic signed [DATA_WIDTH_P-1:0] m_hat_real [4];
    logic signed [DATA_WIDTH_P-1:0] m_hat_imag [4];

    // Error computation
    logic [ACC_WIDTH-1:0] error_current;
    logic [ACC_WIDTH-1:0] error_best;
    logic [ACC_WIDTH-1:0] error_probe_plus;
    logic [ACC_WIDTH-1:0] error_probe_minus;

    // Calibration counters and indices
    logic [15:0] settle_counter;
    logic [3:0]  sample_counter;
    logic [15:0] iteration_counter;
    logic [3:0]  lock_counter;
    logic [1:0]  phase_index;  // Which phase we're optimizing (0-3 for each mesh)

    // Sample accumulators (15-bit signed: 8 samples × 12-bit ADC)
    logic signed [ADC_WIDTH_P+3:0] i0_accum, q0_accum;
    logic signed [ADC_WIDTH_P+3:0] i1_accum, q1_accum;

    // Scaled Q1.15 outputs from ADC accumulators
    logic signed [DATA_WIDTH_P-1:0] i0_scaled, q0_scaled;
    logic signed [DATA_WIDTH_P-1:0] i1_scaled, q1_scaled;

    // Control flags
    logic weights_valid;
    logic cal_start_pending;
    logic eval_start_pending;

    // Mode latch (captured at calibration start)
    logic svd_mode_latched;

    //-------------------------------------------------------------------------
    // Momentum-based optimization state
    // Velocity accumulators for each phase - enables gradient descent with momentum
    // Update rule: v = β*v + step_contribution, then φ += v
    //-------------------------------------------------------------------------
    logic signed [PHASE_WIDTH_P-1:0] phi_velocity [NUM_PHASES_P];
    logic signed [PHASE_WIDTH_P-1:0] phi_velocity_v [NUM_PHASES_V];
    logic signed [PHASE_WIDTH_P-1:0] phi_velocity_u [NUM_PHASES_U];

    //-------------------------------------------------------------------------
    // ADC to Q1.15 Scalers
    //
    // Interface contract: ADC full-scale (±2047) = ±1.0 optical amplitude
    // Scaling: (accumulated / 8) * 16 = accumulated * 2 = accumulated << 1
    // This converts 8-sample ADC accumulation to Q1.15 fixed-point
    //-------------------------------------------------------------------------
    adc_scaler #(
        .ADC_WIDTH(ADC_WIDTH_P),
        .ACC_SAMPLES(CAL_AVG_SAMPLES),
        .Q15_WIDTH(DATA_WIDTH_P),
        .SCALE_SHIFT(1)
    ) scaler_i0 (
        .accumulator(i0_accum),
        .q15_out(i0_scaled)
    );

    adc_scaler #(
        .ADC_WIDTH(ADC_WIDTH_P),
        .ACC_SAMPLES(CAL_AVG_SAMPLES),
        .Q15_WIDTH(DATA_WIDTH_P),
        .SCALE_SHIFT(1)
    ) scaler_q0 (
        .accumulator(q0_accum),
        .q15_out(q0_scaled)
    );

    adc_scaler #(
        .ADC_WIDTH(ADC_WIDTH_P),
        .ACC_SAMPLES(CAL_AVG_SAMPLES),
        .Q15_WIDTH(DATA_WIDTH_P),
        .SCALE_SHIFT(1)
    ) scaler_i1 (
        .accumulator(i1_accum),
        .q15_out(i1_scaled)
    );

    adc_scaler #(
        .ADC_WIDTH(ADC_WIDTH_P),
        .ACC_SAMPLES(CAL_AVG_SAMPLES),
        .Q15_WIDTH(DATA_WIDTH_P),
        .SCALE_SHIFT(1)
    ) scaler_q1 (
        .accumulator(q1_accum),
        .q15_out(q1_scaled)
    );

    //-------------------------------------------------------------------------
    // Input weight validation
    // Q1.15 format inherently represents [-1, 1), so any value is valid
    //-------------------------------------------------------------------------
    assign weights_valid = 1'b1;

    //-------------------------------------------------------------------------
    // Status flags mapping
    //-------------------------------------------------------------------------
    assign status_flags = status;
    assign cal_done     = (top_state == TOP_LOCKED) || (top_state == TOP_ERROR);
    assign cal_locked   = status.cal_locked;
    assign eval_done    = status.eval_done;

    // SVD calibration done detection
    wire svd_cal_done   = (svd_cal_state == SVD_CAL_LOCKED) || (svd_cal_state == SVD_CAL_ERROR);

    //-------------------------------------------------------------------------
    // Top-level state machine
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            top_state <= TOP_IDLE;
        end else begin
            top_state <= top_state_next;
        end
    end

    always_comb begin
        top_state_next = top_state;

        case (top_state)
            TOP_IDLE: begin
                if (start_cal)
                    top_state_next = TOP_CAL;
            end

            TOP_CAL: begin
                if (svd_mode_latched) begin
                    // SVD mode: check SVD FSM
                    if (svd_cal_state == SVD_CAL_LOCKED)
                        top_state_next = TOP_LOCKED;
                    else if (svd_cal_state == SVD_CAL_ERROR)
                        top_state_next = TOP_ERROR;
                end else begin
                    // Unitary mode: check original FSM
                    if (cal_state == CAL_LOCKED)
                        top_state_next = TOP_LOCKED;
                    else if (cal_state == CAL_ERROR)
                        top_state_next = TOP_ERROR;
                end
            end

            TOP_LOCKED: begin
                if (start_eval)
                    top_state_next = TOP_EVAL;
                else if (start_cal)
                    top_state_next = TOP_CAL;
            end

            TOP_EVAL: begin
                if (status.eval_done)
                    top_state_next = TOP_LOCKED;
            end

            TOP_ERROR: begin
                if (start_cal)
                    top_state_next = TOP_CAL;
            end

            default: top_state_next = TOP_IDLE;
        endcase
    end

    //-------------------------------------------------------------------------
    // Calibration FSM
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cal_state <= CAL_IDLE;
        end else begin
            cal_state <= cal_state_next;
        end
    end

    always_comb begin
        cal_state_next = cal_state;

        case (cal_state)
            CAL_IDLE: begin
                if (top_state == TOP_CAL)
                    cal_state_next = CAL_VALIDATE;
            end

            CAL_VALIDATE: begin
                if (weights_valid)
                    cal_state_next = CAL_APPLY_BASIS0;
                else
                    cal_state_next = CAL_ERROR;
            end

            CAL_APPLY_BASIS0: begin
                cal_state_next = CAL_SETTLE0;
            end

            CAL_SETTLE0: begin
                if (settle_counter >= CAL_SETTLE_CYCLES)
                    cal_state_next = CAL_SAMPLE_COL0;
            end

            CAL_SAMPLE_COL0: begin
                if (sample_counter >= CAL_AVG_SAMPLES)
                    cal_state_next = CAL_APPLY_BASIS1;
            end

            CAL_APPLY_BASIS1: begin
                cal_state_next = CAL_SETTLE1;
            end

            CAL_SETTLE1: begin
                if (settle_counter >= CAL_SETTLE_CYCLES)
                    cal_state_next = CAL_SAMPLE_COL1;
            end

            CAL_SAMPLE_COL1: begin
                if (sample_counter >= CAL_AVG_SAMPLES)
                    cal_state_next = CAL_COMPUTE_ERROR;
            end

            CAL_COMPUTE_ERROR: begin
                cal_state_next = CAL_PROBE_PLUS;
            end

            CAL_PROBE_PLUS: begin
                // After settling and measuring with phi+delta
                if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES)
                    cal_state_next = CAL_PROBE_MINUS;
            end

            CAL_PROBE_MINUS: begin
                // After settling and measuring with phi-delta
                if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES)
                    cal_state_next = CAL_UPDATE_PHASE;
            end

            CAL_UPDATE_PHASE: begin
                cal_state_next = CAL_CHECK_CONVERGE;
            end

            CAL_CHECK_CONVERGE: begin
                if (lock_counter >= CAL_LOCK_COUNT)
                    cal_state_next = CAL_LOCKED;
                else if (iteration_counter >= CAL_MAX_ITERATIONS)
                    cal_state_next = CAL_ERROR;
                else if (phase_index < NUM_PHASES_P - 1)
                    // Move to next phase
                    cal_state_next = CAL_PROBE_PLUS;
                else
                    // Full sweep done, start new measurement
                    cal_state_next = CAL_APPLY_BASIS0;
            end

            CAL_LOCKED: begin
                if (top_state != TOP_CAL && top_state != TOP_LOCKED)
                    cal_state_next = CAL_IDLE;
            end

            CAL_ERROR: begin
                if (top_state == TOP_IDLE)
                    cal_state_next = CAL_IDLE;
            end

            default: cal_state_next = CAL_IDLE;
        endcase
    end

    //-------------------------------------------------------------------------
    // SVD Calibration FSM
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            svd_cal_state <= SVD_CAL_IDLE;
            svd_mode_latched <= 1'b0;
        end else begin
            svd_cal_state <= svd_cal_state_next;
            // Latch SVD mode at calibration start
            if (top_state == TOP_IDLE && start_cal)
                svd_mode_latched <= svd_mode;
        end
    end

    always_comb begin
        svd_cal_state_next = svd_cal_state;

        case (svd_cal_state)
            SVD_CAL_IDLE: begin
                if (top_state == TOP_CAL && svd_mode_latched)
                    svd_cal_state_next = SVD_CAL_VALIDATE;
            end

            SVD_CAL_VALIDATE: begin
                if (weights_valid)
                    svd_cal_state_next = SVD_CAL_APPLY_BASIS0_V;
                else
                    svd_cal_state_next = SVD_CAL_ERROR;
            end

            // V† mesh calibration sequence
            SVD_CAL_APPLY_BASIS0_V: svd_cal_state_next = SVD_CAL_SETTLE0_V;

            SVD_CAL_SETTLE0_V: begin
                if (settle_counter >= CAL_SETTLE_CYCLES)
                    svd_cal_state_next = SVD_CAL_SAMPLE_COL0_V;
            end

            SVD_CAL_SAMPLE_COL0_V: begin
                if (sample_counter >= CAL_AVG_SAMPLES)
                    svd_cal_state_next = SVD_CAL_APPLY_BASIS1_V;
            end

            SVD_CAL_APPLY_BASIS1_V: svd_cal_state_next = SVD_CAL_SETTLE1_V;

            SVD_CAL_SETTLE1_V: begin
                if (settle_counter >= CAL_SETTLE_CYCLES)
                    svd_cal_state_next = SVD_CAL_SAMPLE_COL1_V;
            end

            SVD_CAL_SAMPLE_COL1_V: begin
                if (sample_counter >= CAL_AVG_SAMPLES)
                    svd_cal_state_next = SVD_CAL_COMPUTE_ERR_V;
            end

            SVD_CAL_COMPUTE_ERR_V: svd_cal_state_next = SVD_CAL_PROBE_PLUS_V;

            SVD_CAL_PROBE_PLUS_V: begin
                if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES)
                    svd_cal_state_next = SVD_CAL_PROBE_MINUS_V;
            end

            SVD_CAL_PROBE_MINUS_V: begin
                if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES)
                    svd_cal_state_next = SVD_CAL_UPDATE_V;
            end

            SVD_CAL_UPDATE_V: svd_cal_state_next = SVD_CAL_CHECK_V;

            SVD_CAL_CHECK_V: begin
                if (v_locked_reg)
                    svd_cal_state_next = SVD_CAL_SET_SIGMA;
                else if (iteration_counter >= CAL_MAX_ITERATIONS)
                    svd_cal_state_next = SVD_CAL_ERROR;
                else if (phase_index < NUM_PHASES_V - 1)
                    svd_cal_state_next = SVD_CAL_PROBE_PLUS_V;
                else
                    svd_cal_state_next = SVD_CAL_APPLY_BASIS0_V;
            end

            // Sigma assignment (single cycle)
            SVD_CAL_SET_SIGMA: svd_cal_state_next = SVD_CAL_APPLY_BASIS0_U;

            // U mesh calibration sequence
            SVD_CAL_APPLY_BASIS0_U: svd_cal_state_next = SVD_CAL_SETTLE0_U;

            SVD_CAL_SETTLE0_U: begin
                if (settle_counter >= CAL_SETTLE_CYCLES)
                    svd_cal_state_next = SVD_CAL_SAMPLE_COL0_U;
            end

            SVD_CAL_SAMPLE_COL0_U: begin
                if (sample_counter >= CAL_AVG_SAMPLES)
                    svd_cal_state_next = SVD_CAL_APPLY_BASIS1_U;
            end

            SVD_CAL_APPLY_BASIS1_U: svd_cal_state_next = SVD_CAL_SETTLE1_U;

            SVD_CAL_SETTLE1_U: begin
                if (settle_counter >= CAL_SETTLE_CYCLES)
                    svd_cal_state_next = SVD_CAL_SAMPLE_COL1_U;
            end

            SVD_CAL_SAMPLE_COL1_U: begin
                if (sample_counter >= CAL_AVG_SAMPLES)
                    svd_cal_state_next = SVD_CAL_COMPUTE_ERR_U;
            end

            SVD_CAL_COMPUTE_ERR_U: svd_cal_state_next = SVD_CAL_PROBE_PLUS_U;

            SVD_CAL_PROBE_PLUS_U: begin
                if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES)
                    svd_cal_state_next = SVD_CAL_PROBE_MINUS_U;
            end

            SVD_CAL_PROBE_MINUS_U: begin
                if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES)
                    svd_cal_state_next = SVD_CAL_UPDATE_U;
            end

            SVD_CAL_UPDATE_U: svd_cal_state_next = SVD_CAL_CHECK_U;

            SVD_CAL_CHECK_U: begin
                if (u_locked_reg)
                    svd_cal_state_next = SVD_CAL_LOCKED;
                else if (iteration_counter >= CAL_MAX_ITERATIONS)
                    svd_cal_state_next = SVD_CAL_ERROR;
                else if (phase_index < NUM_PHASES_U - 1)
                    svd_cal_state_next = SVD_CAL_PROBE_PLUS_U;
                else
                    svd_cal_state_next = SVD_CAL_APPLY_BASIS0_U;
            end

            SVD_CAL_LOCKED: begin
                if (top_state != TOP_CAL && top_state != TOP_LOCKED)
                    svd_cal_state_next = SVD_CAL_IDLE;
            end

            SVD_CAL_ERROR: begin
                if (top_state == TOP_IDLE)
                    svd_cal_state_next = SVD_CAL_IDLE;
            end

            default: svd_cal_state_next = SVD_CAL_IDLE;
        endcase
    end

    //-------------------------------------------------------------------------
    // Calibration datapath
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset phase registers to mid-scale
            for (int i = 0; i < NUM_PHASES_P; i++) begin
                phi_reg[i] <= 16'h8000;
            end
            // Reset SVD registers
            for (int i = 0; i < NUM_PHASES_V; i++) begin
                phi_reg_v[i] <= 16'h8000;
            end
            for (int i = 0; i < NUM_PHASES_U; i++) begin
                phi_reg_u[i] <= 16'h8000;
            end
            for (int i = 0; i < NUM_VOAS; i++) begin
                voa_reg[i] <= VOA_FULL_TRANSMISSION;
            end
            svd_component <= SVD_COMP_V;
            v_locked_reg <= 1'b0;
            u_locked_reg <= 1'b0;

            phi_step <= PHASE_STEP_INITIAL;
            settle_counter <= '0;
            sample_counter <= '0;
            iteration_counter <= '0;
            lock_counter <= '0;
            phase_index <= '0;
            error_best <= '1;  // Max value
            error_current <= '0;
            i0_accum <= '0;
            q0_accum <= '0;
            i1_accum <= '0;
            q1_accum <= '0;
            for (int i = 0; i < 4; i++) begin
                m_hat_real[i] <= '0;
                m_hat_imag[i] <= '0;
            end
            // Reset velocity registers for momentum optimization
            for (int i = 0; i < NUM_PHASES_P; i++) begin
                phi_velocity[i] <= '0;
            end
            for (int i = 0; i < NUM_PHASES_V; i++) begin
                phi_velocity_v[i] <= '0;
            end
            for (int i = 0; i < NUM_PHASES_U; i++) begin
                phi_velocity_u[i] <= '0;
            end
            status <= '0;

        end else begin
            // Default: clear single-cycle flags
            status.eval_done <= 1'b0;

            case (cal_state)
                CAL_IDLE: begin
                    status.cal_locked <= 1'b0;
                    status.cal_in_progress <= 1'b0;
                    iteration_counter <= '0;
                    lock_counter <= '0;
                    phi_step <= PHASE_STEP_INITIAL;
                    error_best <= '1;
                    // Reset velocity registers for new calibration
                    for (int i = 0; i < NUM_PHASES_P; i++) begin
                        phi_velocity[i] <= '0;
                    end
                end

                CAL_VALIDATE: begin
                    status.cal_in_progress <= 1'b1;
                    status.error_weights <= ~weights_valid;
                end

                CAL_APPLY_BASIS0, CAL_APPLY_BASIS1: begin
                    settle_counter <= '0;
                    sample_counter <= '0;
                    i0_accum <= '0;
                    q0_accum <= '0;
                    i1_accum <= '0;
                    q1_accum <= '0;
                end

                CAL_SETTLE0, CAL_SETTLE1: begin
                    settle_counter <= settle_counter + 1'b1;
                end

                CAL_SAMPLE_COL0: begin
                    if (adc_valid) begin
                        sample_counter <= sample_counter + 1'b1;
                        i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                        q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                        i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                        q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                    end

                    // Store averaged measurements as column 0 of M_hat
                    if (sample_counter >= CAL_AVG_SAMPLES) begin
                        // Use adc_scaler output for proper ADC→Q1.15 conversion
                        m_hat_real[0] <= i0_scaled;  // M00_real
                        m_hat_imag[0] <= q0_scaled;  // M00_imag
                        m_hat_real[2] <= i1_scaled;  // M10_real
                        m_hat_imag[2] <= q1_scaled;  // M10_imag
                    end
                end

                CAL_SAMPLE_COL1: begin
                    if (adc_valid) begin
                        sample_counter <= sample_counter + 1'b1;
                        i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                        q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                        i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                        q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                    end

                    // Store averaged measurements as column 1 of M_hat
                    if (sample_counter >= CAL_AVG_SAMPLES) begin
                        // Use adc_scaler output for proper ADC→Q1.15 conversion
                        m_hat_real[1] <= i0_scaled;  // M01_real
                        m_hat_imag[1] <= q0_scaled;  // M01_imag
                        m_hat_real[3] <= i1_scaled;  // M11_real
                        m_hat_imag[3] <= q1_scaled;  // M11_imag
                    end
                end

                CAL_COMPUTE_ERROR: begin
                    // Compute error = sum of squared differences (real + imaginary)
                    // Includes imaginary error to penalize phase rotation
                    error_current <= compute_error(m_hat_real, m_hat_imag, {w0, w1, w2, w3});
                    phase_index <= '0;
                    settle_counter <= '0;
                    sample_counter <= '0;
                end

                CAL_PROBE_PLUS: begin
                    // Apply phi + delta and measure column 0
                    settle_counter <= settle_counter + 1'b1;
                    if (settle_counter == 0) begin
                        phi_reg[phase_index] <= phi_reg[phase_index] + phi_step;
                        // Reset accumulators for new measurement
                        sample_counter <= '0;
                        i0_accum <= '0;
                        q0_accum <= '0;
                        i1_accum <= '0;
                        q1_accum <= '0;
                    end
                    // ADC sampling after settle period
                    if (settle_counter >= CAL_SETTLE_CYCLES && adc_valid) begin
                        sample_counter <= sample_counter + 1'b1;
                        i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                        q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                        i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                        q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                    end
                    // Compute partial error (column 0 only) after settling and sampling
                    if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES) begin
                        // Use column 0 measurements for partial error: compare M[0,0] vs w0, M[1,0] vs w2
                        // Keep column 1 from previous full measurement
                        // Use adc_scaler output for proper ADC→Q1.15 conversion
                        error_probe_plus <= compute_error(
                            '{i0_scaled, m_hat_real[1],
                              i1_scaled, m_hat_real[3]},
                            '{q0_scaled, m_hat_imag[1],
                              q1_scaled, m_hat_imag[3]},
                            {w0, w1, w2, w3}
                        );
                        // Restore original phase for minus probe
                        phi_reg[phase_index] <= phi_reg[phase_index] - phi_step;
                        settle_counter <= '0;
                        sample_counter <= '0;
                    end
                end

                CAL_PROBE_MINUS: begin
                    // Apply phi - delta and measure column 0
                    settle_counter <= settle_counter + 1'b1;
                    if (settle_counter == 0) begin
                        phi_reg[phase_index] <= phi_reg[phase_index] - phi_step;
                        // Reset accumulators for new measurement
                        sample_counter <= '0;
                        i0_accum <= '0;
                        q0_accum <= '0;
                        i1_accum <= '0;
                        q1_accum <= '0;
                    end
                    // ADC sampling after settle period
                    if (settle_counter >= CAL_SETTLE_CYCLES && adc_valid) begin
                        sample_counter <= sample_counter + 1'b1;
                        i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                        q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                        i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                        q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                    end
                    // Compute partial error (column 0 only) after settling and sampling
                    if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES) begin
                        // Use column 0 measurements for partial error
                        // Use adc_scaler output for proper ADC→Q1.15 conversion
                        error_probe_minus <= compute_error(
                            '{i0_scaled, m_hat_real[1],
                              i1_scaled, m_hat_real[3]},
                            '{q0_scaled, m_hat_imag[1],
                              q1_scaled, m_hat_imag[3]},
                            {w0, w1, w2, w3}
                        );
                    end
                end

                CAL_UPDATE_PHASE: begin
                    // Momentum-based gradient descent with direction reversal detection
                    // Gradient sign: if error_probe_plus > error_probe_minus, move negative
                    // Velocity update: v = β*v + step_contribution (β ≈ 0.3125)
                    // Key improvement: Reset velocity on direction change to prevent overshoot

                    // Compute velocity decay: v_decay = (v >> 2) + (v >> 4) ≈ 0.3125 * v
                    automatic logic signed [PHASE_WIDTH_P-1:0] v_decayed;
                    automatic logic signed [PHASE_WIDTH_P-1:0] v_new;  // New velocity computed combinationally
                    automatic logic gradient_positive;  // True if should move positive
                    automatic logic velocity_positive;  // True if velocity is positive

                    v_decayed = (phi_velocity[phase_index] >>> MOMENTUM_DECAY_SHIFT1) +
                                (phi_velocity[phase_index] >>> MOMENTUM_DECAY_SHIFT2);

                    gradient_positive = (error_probe_plus < error_probe_minus);
                    velocity_positive = !phi_velocity[phase_index][PHASE_WIDTH_P-1];

                    // Compute new velocity combinationally so it can be used immediately
                    // Check for direction reversal (gradient and velocity have opposite signs)
                    // If reversing direction, reset velocity to prevent overshoot
                    if (gradient_positive != velocity_positive && phi_velocity[phase_index] != '0) begin
                        // Direction reversal - reset velocity and use step only
                        v_new = gradient_positive ? $signed(phi_step) : -$signed(phi_step);
                    end else begin
                        // Same direction - apply momentum
                        v_new = gradient_positive ? (v_decayed + $signed(phi_step))
                                                  : (v_decayed - $signed(phi_step));
                    end

                    // Store new velocity for next iteration
                    phi_velocity[phase_index] <= v_new;

                    // Update best error
                    if (error_probe_plus < error_probe_minus) begin
                        if (error_probe_plus < error_best)
                            error_best <= error_probe_plus;
                    end else begin
                        if (error_probe_minus < error_best)
                            error_best <= error_probe_minus;
                    end

                    // Apply velocity to phase (restore to original then add velocity)
                    // Current position is at phi - delta (from probe_minus)
                    // Restore to original: +delta, then apply NEW velocity (not old!)
                    phi_reg[phase_index] <= phi_reg[phase_index] + phi_step + v_new;

                    iteration_counter <= iteration_counter + 1'b1;
                end

                CAL_CHECK_CONVERGE: begin
                    // Check if error is below threshold with hysteresis
                    // Lock when below CAL_LOCK_THRESHOLD, only reset if above CAL_UNLOCK_THRESHOLD
                    if (error_best < CAL_LOCK_THRESHOLD) begin
                        lock_counter <= lock_counter + 1'b1;
                    end else if (error_best > CAL_UNLOCK_THRESHOLD) begin
                        // Only reset if significantly above threshold (hysteresis prevents oscillation)
                        lock_counter <= '0;
                        // Reduce step size when not converging (every 64 iterations instead of 32)
                        if (iteration_counter[5:0] == '0 && phi_step > PHASE_STEP_MIN) begin
                            phi_step <= phi_step >> PHASE_STEP_DECAY;
                        end
                    end
                    // If between thresholds, maintain lock_counter progress (no reset)

                    // Move to next phase
                    phase_index <= phase_index + 1'b1;
                end

                CAL_LOCKED: begin
                    status.cal_locked <= 1'b1;
                    status.cal_in_progress <= 1'b0;
                end

                CAL_ERROR: begin
                    status.error_timeout <= (iteration_counter >= CAL_MAX_ITERATIONS);
                    status.cal_in_progress <= 1'b0;
                end

                default: begin
                end
            endcase

            //---------------------------------------------------------------------
            // SVD Calibration Datapath
            //---------------------------------------------------------------------
            if (svd_mode_latched) begin
                case (svd_cal_state)
                    SVD_CAL_IDLE: begin
                        status.cal_locked <= 1'b0;
                        status.cal_in_progress <= 1'b0;
                        status.v_locked <= 1'b0;
                        status.u_locked <= 1'b0;
                        v_locked_reg <= 1'b0;
                        u_locked_reg <= 1'b0;
                        iteration_counter <= '0;
                        lock_counter <= '0;
                        phi_step <= PHASE_STEP_INITIAL;
                        error_best <= '1;
                        svd_component <= SVD_COMP_V;
                        // Reset velocity registers for new calibration
                        for (int i = 0; i < NUM_PHASES_V; i++) begin
                            phi_velocity_v[i] <= '0;
                        end
                        for (int i = 0; i < NUM_PHASES_U; i++) begin
                            phi_velocity_u[i] <= '0;
                        end
                    end

                    SVD_CAL_VALIDATE: begin
                        status.cal_in_progress <= 1'b1;
                        status.error_weights <= ~weights_valid;
                        svd_component <= SVD_COMP_V;
                    end

                    // V† calibration states
                    SVD_CAL_APPLY_BASIS0_V, SVD_CAL_APPLY_BASIS1_V: begin
                        settle_counter <= '0;
                        sample_counter <= '0;
                        i0_accum <= '0;
                        q0_accum <= '0;
                        i1_accum <= '0;
                        q1_accum <= '0;
                    end

                    SVD_CAL_SETTLE0_V, SVD_CAL_SETTLE1_V: begin
                        settle_counter <= settle_counter + 1'b1;
                    end

                    SVD_CAL_SAMPLE_COL0_V: begin
                        if (adc_valid) begin
                            sample_counter <= sample_counter + 1'b1;
                            i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                            q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                            i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                            q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                        end
                        if (sample_counter >= CAL_AVG_SAMPLES) begin
                            // Use adc_scaler output for proper ADC→Q1.15 conversion
                            m_hat_real[0] <= i0_scaled;
                            m_hat_imag[0] <= q0_scaled;
                            m_hat_real[2] <= i1_scaled;
                            m_hat_imag[2] <= q1_scaled;
                        end
                    end

                    SVD_CAL_SAMPLE_COL1_V: begin
                        if (adc_valid) begin
                            sample_counter <= sample_counter + 1'b1;
                            i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                            q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                            i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                            q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                        end
                        if (sample_counter >= CAL_AVG_SAMPLES) begin
                            // Use adc_scaler output for proper ADC→Q1.15 conversion
                            m_hat_real[1] <= i0_scaled;
                            m_hat_imag[1] <= q0_scaled;
                            m_hat_real[3] <= i1_scaled;
                            m_hat_imag[3] <= q1_scaled;
                        end
                    end

                    SVD_CAL_COMPUTE_ERR_V: begin
                        error_current <= compute_error(m_hat_real, m_hat_imag, {w0, w1, w2, w3});
                        phase_index <= '0;
                        settle_counter <= '0;
                        sample_counter <= '0;
                    end

                    SVD_CAL_PROBE_PLUS_V: begin
                        settle_counter <= settle_counter + 1'b1;
                        if (settle_counter == 0) begin
                            phi_reg_v[phase_index] <= phi_reg_v[phase_index] + phi_step;
                            // Reset accumulators for new measurement
                            sample_counter <= '0;
                            i0_accum <= '0;
                            q0_accum <= '0;
                            i1_accum <= '0;
                            q1_accum <= '0;
                        end
                        // ADC sampling after settle period
                        if (settle_counter >= CAL_SETTLE_CYCLES && adc_valid) begin
                            sample_counter <= sample_counter + 1'b1;
                            i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                            q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                            i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                            q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                        end
                        if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES) begin
                            // Use column 0 measurements for partial error
                            // Use adc_scaler output for proper ADC→Q1.15 conversion
                            error_probe_plus <= compute_error(
                                '{i0_scaled, m_hat_real[1],
                                  i1_scaled, m_hat_real[3]},
                                '{q0_scaled, m_hat_imag[1],
                                  q1_scaled, m_hat_imag[3]},
                                {w0, w1, w2, w3}
                            );
                            phi_reg_v[phase_index] <= phi_reg_v[phase_index] - phi_step;
                            settle_counter <= '0;
                            sample_counter <= '0;
                        end
                    end

                    SVD_CAL_PROBE_MINUS_V: begin
                        settle_counter <= settle_counter + 1'b1;
                        if (settle_counter == 0) begin
                            phi_reg_v[phase_index] <= phi_reg_v[phase_index] - phi_step;
                            // Reset accumulators for new measurement
                            sample_counter <= '0;
                            i0_accum <= '0;
                            q0_accum <= '0;
                            i1_accum <= '0;
                            q1_accum <= '0;
                        end
                        // ADC sampling after settle period
                        if (settle_counter >= CAL_SETTLE_CYCLES && adc_valid) begin
                            sample_counter <= sample_counter + 1'b1;
                            i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                            q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                            i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                            q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                        end
                        if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES) begin
                            // Use column 0 measurements for partial error
                            // Use adc_scaler output for proper ADC→Q1.15 conversion
                            error_probe_minus <= compute_error(
                                '{i0_scaled, m_hat_real[1],
                                  i1_scaled, m_hat_real[3]},
                                '{q0_scaled, m_hat_imag[1],
                                  q1_scaled, m_hat_imag[3]},
                                {w0, w1, w2, w3}
                            );
                        end
                    end

                    SVD_CAL_UPDATE_V: begin
                        // Momentum-based gradient descent for V† mesh with direction reversal detection
                        automatic logic signed [PHASE_WIDTH_P-1:0] v_decayed;
                        automatic logic signed [PHASE_WIDTH_P-1:0] v_new;  // New velocity computed combinationally
                        automatic logic gradient_positive;
                        automatic logic velocity_positive;

                        v_decayed = (phi_velocity_v[phase_index] >>> MOMENTUM_DECAY_SHIFT1) +
                                    (phi_velocity_v[phase_index] >>> MOMENTUM_DECAY_SHIFT2);

                        gradient_positive = (error_probe_plus < error_probe_minus);
                        velocity_positive = !phi_velocity_v[phase_index][PHASE_WIDTH_P-1];

                        // Compute new velocity combinationally so it can be used immediately
                        // Reset velocity on direction reversal to prevent overshoot
                        if (gradient_positive != velocity_positive && phi_velocity_v[phase_index] != '0) begin
                            v_new = gradient_positive ? $signed(phi_step) : -$signed(phi_step);
                        end else begin
                            v_new = gradient_positive ? (v_decayed + $signed(phi_step))
                                                      : (v_decayed - $signed(phi_step));
                        end

                        // Store new velocity for next iteration
                        phi_velocity_v[phase_index] <= v_new;

                        // Update best error
                        if (error_probe_plus < error_probe_minus) begin
                            if (error_probe_plus < error_best)
                                error_best <= error_probe_plus;
                        end else begin
                            if (error_probe_minus < error_best)
                                error_best <= error_probe_minus;
                        end

                        // Apply NEW velocity to phase (not old!)
                        phi_reg_v[phase_index] <= phi_reg_v[phase_index] + phi_step + v_new;
                        iteration_counter <= iteration_counter + 1'b1;
                    end

                    SVD_CAL_CHECK_V: begin
                        // Check if error is below threshold with hysteresis
                        if (error_best < CAL_LOCK_THRESHOLD) begin
                            lock_counter <= lock_counter + 1'b1;
                            if (lock_counter >= CAL_LOCK_COUNT - 1) begin
                                v_locked_reg <= 1'b1;
                                status.v_locked <= 1'b1;
                            end
                        end else if (error_best > CAL_UNLOCK_THRESHOLD) begin
                            // Only reset if significantly above threshold (hysteresis)
                            lock_counter <= '0;
                            if (iteration_counter[5:0] == '0 && phi_step > PHASE_STEP_MIN)
                                phi_step <= phi_step >> PHASE_STEP_DECAY;
                        end
                        // If between thresholds, maintain lock_counter progress
                        phase_index <= phase_index + 1'b1;
                    end

                    // Sigma assignment
                    SVD_CAL_SET_SIGMA: begin
                        voa_reg[0] <= sigma_dac0;
                        voa_reg[1] <= sigma_dac1;
                        svd_component <= SVD_COMP_U;
                        // Reset for U calibration
                        lock_counter <= '0;
                        phi_step <= PHASE_STEP_INITIAL;
                        error_best <= '1;
                        phase_index <= '0;
                    end

                    // U calibration states (same structure as V)
                    SVD_CAL_APPLY_BASIS0_U, SVD_CAL_APPLY_BASIS1_U: begin
                        settle_counter <= '0;
                        sample_counter <= '0;
                        i0_accum <= '0;
                        q0_accum <= '0;
                        i1_accum <= '0;
                        q1_accum <= '0;
                    end

                    SVD_CAL_SETTLE0_U, SVD_CAL_SETTLE1_U: begin
                        settle_counter <= settle_counter + 1'b1;
                    end

                    SVD_CAL_SAMPLE_COL0_U: begin
                        if (adc_valid) begin
                            sample_counter <= sample_counter + 1'b1;
                            i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                            q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                            i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                            q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                        end
                        if (sample_counter >= CAL_AVG_SAMPLES) begin
                            // Use adc_scaler output for proper ADC→Q1.15 conversion
                            m_hat_real[0] <= i0_scaled;
                            m_hat_imag[0] <= q0_scaled;
                            m_hat_real[2] <= i1_scaled;
                            m_hat_imag[2] <= q1_scaled;
                        end
                    end

                    SVD_CAL_SAMPLE_COL1_U: begin
                        if (adc_valid) begin
                            sample_counter <= sample_counter + 1'b1;
                            i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                            q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                            i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                            q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                        end
                        if (sample_counter >= CAL_AVG_SAMPLES) begin
                            // Use adc_scaler output for proper ADC→Q1.15 conversion
                            m_hat_real[1] <= i0_scaled;
                            m_hat_imag[1] <= q0_scaled;
                            m_hat_real[3] <= i1_scaled;
                            m_hat_imag[3] <= q1_scaled;
                        end
                    end

                    SVD_CAL_COMPUTE_ERR_U: begin
                        error_current <= compute_error(m_hat_real, m_hat_imag, {w0, w1, w2, w3});
                        phase_index <= '0;
                        settle_counter <= '0;
                        sample_counter <= '0;
                    end

                    SVD_CAL_PROBE_PLUS_U: begin
                        settle_counter <= settle_counter + 1'b1;
                        if (settle_counter == 0) begin
                            phi_reg_u[phase_index] <= phi_reg_u[phase_index] + phi_step;
                            // Reset accumulators for new measurement
                            sample_counter <= '0;
                            i0_accum <= '0;
                            q0_accum <= '0;
                            i1_accum <= '0;
                            q1_accum <= '0;
                        end
                        // ADC sampling after settle period
                        if (settle_counter >= CAL_SETTLE_CYCLES && adc_valid) begin
                            sample_counter <= sample_counter + 1'b1;
                            i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                            q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                            i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                            q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                        end
                        if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES) begin
                            // Use column 0 measurements for partial error
                            // Use adc_scaler output for proper ADC→Q1.15 conversion
                            error_probe_plus <= compute_error(
                                '{i0_scaled, m_hat_real[1],
                                  i1_scaled, m_hat_real[3]},
                                '{q0_scaled, m_hat_imag[1],
                                  q1_scaled, m_hat_imag[3]},
                                {w0, w1, w2, w3}
                            );
                            phi_reg_u[phase_index] <= phi_reg_u[phase_index] - phi_step;
                            settle_counter <= '0;
                            sample_counter <= '0;
                        end
                    end

                    SVD_CAL_PROBE_MINUS_U: begin
                        settle_counter <= settle_counter + 1'b1;
                        if (settle_counter == 0) begin
                            phi_reg_u[phase_index] <= phi_reg_u[phase_index] - phi_step;
                            // Reset accumulators for new measurement
                            sample_counter <= '0;
                            i0_accum <= '0;
                            q0_accum <= '0;
                            i1_accum <= '0;
                            q1_accum <= '0;
                        end
                        // ADC sampling after settle period
                        if (settle_counter >= CAL_SETTLE_CYCLES && adc_valid) begin
                            sample_counter <= sample_counter + 1'b1;
                            i0_accum <= i0_accum + {{4{adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                            q0_accum <= q0_accum + {{4{adc_q0[ADC_WIDTH_P-1]}}, adc_q0};
                            i1_accum <= i1_accum + {{4{adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                            q1_accum <= q1_accum + {{4{adc_q1[ADC_WIDTH_P-1]}}, adc_q1};
                        end
                        if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES) begin
                            // Use column 0 measurements for partial error
                            // Use adc_scaler output for proper ADC→Q1.15 conversion
                            error_probe_minus <= compute_error(
                                '{i0_scaled, m_hat_real[1],
                                  i1_scaled, m_hat_real[3]},
                                '{q0_scaled, m_hat_imag[1],
                                  q1_scaled, m_hat_imag[3]},
                                {w0, w1, w2, w3}
                            );
                        end
                    end

                    SVD_CAL_UPDATE_U: begin
                        // Momentum-based gradient descent for U mesh with direction reversal detection
                        automatic logic signed [PHASE_WIDTH_P-1:0] v_decayed;
                        automatic logic signed [PHASE_WIDTH_P-1:0] v_new;  // New velocity computed combinationally
                        automatic logic gradient_positive;
                        automatic logic velocity_positive;

                        v_decayed = (phi_velocity_u[phase_index] >>> MOMENTUM_DECAY_SHIFT1) +
                                    (phi_velocity_u[phase_index] >>> MOMENTUM_DECAY_SHIFT2);

                        gradient_positive = (error_probe_plus < error_probe_minus);
                        velocity_positive = !phi_velocity_u[phase_index][PHASE_WIDTH_P-1];

                        // Compute new velocity combinationally so it can be used immediately
                        // Reset velocity on direction reversal to prevent overshoot
                        if (gradient_positive != velocity_positive && phi_velocity_u[phase_index] != '0) begin
                            v_new = gradient_positive ? $signed(phi_step) : -$signed(phi_step);
                        end else begin
                            v_new = gradient_positive ? (v_decayed + $signed(phi_step))
                                                      : (v_decayed - $signed(phi_step));
                        end

                        // Store new velocity for next iteration
                        phi_velocity_u[phase_index] <= v_new;

                        // Update best error
                        if (error_probe_plus < error_probe_minus) begin
                            if (error_probe_plus < error_best)
                                error_best <= error_probe_plus;
                        end else begin
                            if (error_probe_minus < error_best)
                                error_best <= error_probe_minus;
                        end

                        // Apply NEW velocity to phase (not old!)
                        phi_reg_u[phase_index] <= phi_reg_u[phase_index] + phi_step + v_new;
                        iteration_counter <= iteration_counter + 1'b1;
                    end

                    SVD_CAL_CHECK_U: begin
                        // Check if error is below threshold with hysteresis
                        if (error_best < CAL_LOCK_THRESHOLD) begin
                            lock_counter <= lock_counter + 1'b1;
                            if (lock_counter >= CAL_LOCK_COUNT - 1) begin
                                u_locked_reg <= 1'b1;
                                status.u_locked <= 1'b1;
                            end
                        end else if (error_best > CAL_UNLOCK_THRESHOLD) begin
                            // Only reset if significantly above threshold (hysteresis)
                            lock_counter <= '0;
                            if (iteration_counter[5:0] == '0 && phi_step > PHASE_STEP_MIN)
                                phi_step <= phi_step >> PHASE_STEP_DECAY;
                        end
                        // If between thresholds, maintain lock_counter progress
                        phase_index <= phase_index + 1'b1;
                    end

                    SVD_CAL_LOCKED: begin
                        status.cal_locked <= 1'b1;
                        status.cal_in_progress <= 1'b0;
                    end

                    SVD_CAL_ERROR: begin
                        status.error_timeout <= (iteration_counter >= CAL_MAX_ITERATIONS);
                        status.cal_in_progress <= 1'b0;
                    end

                    default: begin
                    end
                endcase
            end
        end
    end

    //-------------------------------------------------------------------------
    // Error computation function
    // Includes both real and imaginary parts for proper complex optimization.
    // For real target matrices, imaginary error penalizes phase rotation.
    //-------------------------------------------------------------------------
    function automatic logic [ACC_WIDTH-1:0] compute_error(
        input logic signed [DATA_WIDTH_P-1:0] m_real [4],
        input logic signed [DATA_WIDTH_P-1:0] m_imag [4],
        input logic signed [DATA_WIDTH_P-1:0] w [4]
    );
        logic signed [DATA_WIDTH_P:0] diff_real [4];
        logic signed [DATA_WIDTH_P:0] diff_imag [4];
        logic [ACC_WIDTH-1:0] sum;
        sum = '0;
        for (int i = 0; i < 4; i++) begin
            diff_real[i] = m_real[i] - w[i];
            diff_imag[i] = m_imag[i];  // Target imag = 0 for real matrices
            sum = sum + (diff_real[i] * diff_real[i]) + (diff_imag[i] * diff_imag[i]);
        end
        return sum;
    endfunction

    //-------------------------------------------------------------------------
    // Plant interface output assignments
    //-------------------------------------------------------------------------
    always_comb begin
        // Default outputs
        plant_enable = 1'b0;
        plant_mode   = PLANT_IDLE;
        x_drive0     = '0;
        x_drive1     = '0;

        case (top_state)
            TOP_CAL: begin
                plant_enable = 1'b1;
                plant_mode   = PLANT_CAL;

                if (svd_mode_latched) begin
                    // SVD mode: drive basis vectors based on SVD FSM state
                    case (svd_cal_state)
                        SVD_CAL_APPLY_BASIS0_V, SVD_CAL_SETTLE0_V, SVD_CAL_SAMPLE_COL0_V,
                        SVD_CAL_APPLY_BASIS0_U, SVD_CAL_SETTLE0_U, SVD_CAL_SAMPLE_COL0_U,
                        SVD_CAL_PROBE_PLUS_V, SVD_CAL_PROBE_MINUS_V,
                        SVD_CAL_PROBE_PLUS_U, SVD_CAL_PROBE_MINUS_U: begin
                            // Drive basis0 [1,0] for column 0 measurements and PROBE states
                            x_drive0 = Q1_15_ONE;
                            x_drive1 = '0;
                        end
                        SVD_CAL_APPLY_BASIS1_V, SVD_CAL_SETTLE1_V, SVD_CAL_SAMPLE_COL1_V,
                        SVD_CAL_APPLY_BASIS1_U, SVD_CAL_SETTLE1_U, SVD_CAL_SAMPLE_COL1_U: begin
                            x_drive0 = '0;
                            x_drive1 = Q1_15_ONE;
                        end
                        default: begin
                            x_drive0 = '0;
                            x_drive1 = '0;
                        end
                    endcase
                end else begin
                    // Unitary mode: drive basis vectors for calibration
                    case (cal_state)
                        CAL_APPLY_BASIS0, CAL_SETTLE0, CAL_SAMPLE_COL0,
                        CAL_PROBE_PLUS, CAL_PROBE_MINUS: begin
                            // Drive basis0 [1,0] for column 0 measurements and PROBE states
                            x_drive0 = Q1_15_ONE;  // 1.0
                            x_drive1 = '0;          // 0.0
                        end
                        CAL_APPLY_BASIS1, CAL_SETTLE1, CAL_SAMPLE_COL1: begin
                            x_drive0 = '0;          // 0.0
                            x_drive1 = Q1_15_ONE;  // 1.0
                        end
                        default: begin
                            x_drive0 = '0;
                            x_drive1 = '0;
                        end
                    endcase
                end
            end

            TOP_EVAL: begin
                plant_enable = 1'b1;
                plant_mode   = PLANT_EVAL;
                x_drive0     = x0;
                x_drive1     = x1;
            end

            default: begin
                plant_enable = 1'b0;
                plant_mode   = PLANT_IDLE;
            end
        endcase
    end

    //-------------------------------------------------------------------------
    // Phase and VOA DAC output assignments
    //-------------------------------------------------------------------------
    always_comb begin
        if (svd_mode_latched) begin
            // SVD mode: route all 10 parameters
            for (int i = 0; i < NUM_PHASES_V; i++) begin
                phi_dac_v[i] = phi_reg_v[i];
            end
            for (int i = 0; i < NUM_PHASES_U; i++) begin
                phi_dac_u[i] = phi_reg_u[i];
            end
            voa_dac[0] = voa_reg[0];
            voa_dac[1] = voa_reg[1];

            // Legacy interface gets V† phases for backward compatibility
            for (int i = 0; i < NUM_PHASES_P; i++) begin
                phi_dac[i] = phi_reg_v[i];
            end
        end else begin
            // Unitary mode: original behavior
            for (int i = 0; i < NUM_PHASES_P; i++) begin
                phi_dac[i] = phi_reg[i];
                phi_dac_v[i] = phi_reg[i];
                phi_dac_u[i] = '0;
            end
            // VOAs at full transmission (no attenuation)
            voa_dac[0] = VOA_FULL_TRANSMISSION;
            voa_dac[1] = VOA_FULL_TRANSMISSION;
        end
    end

    //-------------------------------------------------------------------------
    // Evaluation datapath
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y0_out  <= '0;
            y1_out  <= '0;
            y_valid <= 1'b0;
        end else begin
            y_valid <= 1'b0;

            if (top_state == TOP_EVAL && adc_valid) begin
                // Convert I/Q measurements to output
                // For real-only targets, I channel is the primary output
                // Scale ADC codes to Q1.15 range
                y0_out  <= {{(DATA_WIDTH_P-ADC_WIDTH_P){adc_i0[ADC_WIDTH_P-1]}}, adc_i0};
                y1_out  <= {{(DATA_WIDTH_P-ADC_WIDTH_P){adc_i1[ADC_WIDTH_P-1]}}, adc_i1};
                y_valid <= 1'b1;
                status.eval_done <= 1'b1;
            end
        end
    end

endmodule
