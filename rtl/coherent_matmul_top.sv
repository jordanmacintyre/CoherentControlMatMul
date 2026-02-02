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

    // Sample accumulators
    logic signed [DATA_WIDTH_P+3:0] i0_accum, q0_accum;
    logic signed [DATA_WIDTH_P+3:0] i1_accum, q1_accum;

    // Control flags
    logic weights_valid;
    logic cal_start_pending;
    logic eval_start_pending;

    // Mode latch (captured at calibration start)
    logic svd_mode_latched;

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
                        // Scale accumulator to Q1.15 (divide by samples, scale by gain)
                        m_hat_real[0] <= i0_accum[DATA_WIDTH_P+2:3];  // M00_real
                        m_hat_imag[0] <= q0_accum[DATA_WIDTH_P+2:3];  // M00_imag
                        m_hat_real[2] <= i1_accum[DATA_WIDTH_P+2:3];  // M10_real
                        m_hat_imag[2] <= q1_accum[DATA_WIDTH_P+2:3];  // M10_imag
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
                        m_hat_real[1] <= i0_accum[DATA_WIDTH_P+2:3];  // M01_real
                        m_hat_imag[1] <= q0_accum[DATA_WIDTH_P+2:3];  // M01_imag
                        m_hat_real[3] <= i1_accum[DATA_WIDTH_P+2:3];  // M11_real
                        m_hat_imag[3] <= q1_accum[DATA_WIDTH_P+2:3];  // M11_imag
                    end
                end

                CAL_COMPUTE_ERROR: begin
                    // Compute error = sum of squared differences
                    // Simplified: just compare real parts to target weights
                    // Full implementation would include imaginary error term
                    error_current <= compute_error(m_hat_real, {w0, w1, w2, w3});
                    phase_index <= '0;
                    settle_counter <= '0;
                    sample_counter <= '0;
                end

                CAL_PROBE_PLUS: begin
                    // Apply phi + delta and measure
                    settle_counter <= settle_counter + 1'b1;
                    if (settle_counter == 0) begin
                        phi_reg[phase_index] <= phi_reg[phase_index] + phi_step;
                    end
                    // Store error after settling and sampling
                    if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES) begin
                        error_probe_plus <= error_current;
                        // Restore original phase for minus probe
                        phi_reg[phase_index] <= phi_reg[phase_index] - phi_step;
                        settle_counter <= '0;
                        sample_counter <= '0;
                    end
                end

                CAL_PROBE_MINUS: begin
                    // Apply phi - delta and measure
                    settle_counter <= settle_counter + 1'b1;
                    if (settle_counter == 0) begin
                        phi_reg[phase_index] <= phi_reg[phase_index] - phi_step;
                    end
                    // Store error after settling and sampling
                    if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES) begin
                        error_probe_minus <= error_current;
                    end
                end

                CAL_UPDATE_PHASE: begin
                    // Choose best direction and update phase
                    if (error_probe_plus < error_probe_minus && error_probe_plus < error_best) begin
                        // Plus direction is better
                        phi_reg[phase_index] <= phi_reg[phase_index] + phi_step + phi_step;
                        error_best <= error_probe_plus;
                    end else if (error_probe_minus < error_best) begin
                        // Minus direction is better (already at phi - delta)
                        error_best <= error_probe_minus;
                    end else begin
                        // No improvement, restore original
                        phi_reg[phase_index] <= phi_reg[phase_index] + phi_step;
                    end
                    iteration_counter <= iteration_counter + 1'b1;
                end

                CAL_CHECK_CONVERGE: begin
                    // Check if error is below threshold
                    if (error_best < CAL_LOCK_THRESHOLD) begin
                        lock_counter <= lock_counter + 1'b1;
                    end else begin
                        lock_counter <= '0;
                        // Reduce step size when not converging
                        if (iteration_counter[4:0] == '0 && phi_step > PHASE_STEP_MIN) begin
                            phi_step <= phi_step >> PHASE_STEP_DECAY;
                        end
                    end

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
                            m_hat_real[0] <= i0_accum[DATA_WIDTH_P+2:3];
                            m_hat_imag[0] <= q0_accum[DATA_WIDTH_P+2:3];
                            m_hat_real[2] <= i1_accum[DATA_WIDTH_P+2:3];
                            m_hat_imag[2] <= q1_accum[DATA_WIDTH_P+2:3];
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
                            m_hat_real[1] <= i0_accum[DATA_WIDTH_P+2:3];
                            m_hat_imag[1] <= q0_accum[DATA_WIDTH_P+2:3];
                            m_hat_real[3] <= i1_accum[DATA_WIDTH_P+2:3];
                            m_hat_imag[3] <= q1_accum[DATA_WIDTH_P+2:3];
                        end
                    end

                    SVD_CAL_COMPUTE_ERR_V: begin
                        error_current <= compute_error(m_hat_real, {w0, w1, w2, w3});
                        phase_index <= '0;
                        settle_counter <= '0;
                        sample_counter <= '0;
                    end

                    SVD_CAL_PROBE_PLUS_V: begin
                        settle_counter <= settle_counter + 1'b1;
                        if (settle_counter == 0)
                            phi_reg_v[phase_index] <= phi_reg_v[phase_index] + phi_step;
                        if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES) begin
                            error_probe_plus <= error_current;
                            phi_reg_v[phase_index] <= phi_reg_v[phase_index] - phi_step;
                            settle_counter <= '0;
                            sample_counter <= '0;
                        end
                    end

                    SVD_CAL_PROBE_MINUS_V: begin
                        settle_counter <= settle_counter + 1'b1;
                        if (settle_counter == 0)
                            phi_reg_v[phase_index] <= phi_reg_v[phase_index] - phi_step;
                        if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES)
                            error_probe_minus <= error_current;
                    end

                    SVD_CAL_UPDATE_V: begin
                        if (error_probe_plus < error_probe_minus && error_probe_plus < error_best) begin
                            phi_reg_v[phase_index] <= phi_reg_v[phase_index] + phi_step + phi_step;
                            error_best <= error_probe_plus;
                        end else if (error_probe_minus < error_best) begin
                            error_best <= error_probe_minus;
                        end else begin
                            phi_reg_v[phase_index] <= phi_reg_v[phase_index] + phi_step;
                        end
                        iteration_counter <= iteration_counter + 1'b1;
                    end

                    SVD_CAL_CHECK_V: begin
                        if (error_best < CAL_LOCK_THRESHOLD) begin
                            lock_counter <= lock_counter + 1'b1;
                            if (lock_counter >= CAL_LOCK_COUNT - 1) begin
                                v_locked_reg <= 1'b1;
                                status.v_locked <= 1'b1;
                            end
                        end else begin
                            lock_counter <= '0;
                            if (iteration_counter[4:0] == '0 && phi_step > PHASE_STEP_MIN)
                                phi_step <= phi_step >> PHASE_STEP_DECAY;
                        end
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
                            m_hat_real[0] <= i0_accum[DATA_WIDTH_P+2:3];
                            m_hat_imag[0] <= q0_accum[DATA_WIDTH_P+2:3];
                            m_hat_real[2] <= i1_accum[DATA_WIDTH_P+2:3];
                            m_hat_imag[2] <= q1_accum[DATA_WIDTH_P+2:3];
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
                            m_hat_real[1] <= i0_accum[DATA_WIDTH_P+2:3];
                            m_hat_imag[1] <= q0_accum[DATA_WIDTH_P+2:3];
                            m_hat_real[3] <= i1_accum[DATA_WIDTH_P+2:3];
                            m_hat_imag[3] <= q1_accum[DATA_WIDTH_P+2:3];
                        end
                    end

                    SVD_CAL_COMPUTE_ERR_U: begin
                        error_current <= compute_error(m_hat_real, {w0, w1, w2, w3});
                        phase_index <= '0;
                        settle_counter <= '0;
                        sample_counter <= '0;
                    end

                    SVD_CAL_PROBE_PLUS_U: begin
                        settle_counter <= settle_counter + 1'b1;
                        if (settle_counter == 0)
                            phi_reg_u[phase_index] <= phi_reg_u[phase_index] + phi_step;
                        if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES) begin
                            error_probe_plus <= error_current;
                            phi_reg_u[phase_index] <= phi_reg_u[phase_index] - phi_step;
                            settle_counter <= '0;
                            sample_counter <= '0;
                        end
                    end

                    SVD_CAL_PROBE_MINUS_U: begin
                        settle_counter <= settle_counter + 1'b1;
                        if (settle_counter == 0)
                            phi_reg_u[phase_index] <= phi_reg_u[phase_index] - phi_step;
                        if (settle_counter >= CAL_SETTLE_CYCLES && sample_counter >= CAL_AVG_SAMPLES)
                            error_probe_minus <= error_current;
                    end

                    SVD_CAL_UPDATE_U: begin
                        if (error_probe_plus < error_probe_minus && error_probe_plus < error_best) begin
                            phi_reg_u[phase_index] <= phi_reg_u[phase_index] + phi_step + phi_step;
                            error_best <= error_probe_plus;
                        end else if (error_probe_minus < error_best) begin
                            error_best <= error_probe_minus;
                        end else begin
                            phi_reg_u[phase_index] <= phi_reg_u[phase_index] + phi_step;
                        end
                        iteration_counter <= iteration_counter + 1'b1;
                    end

                    SVD_CAL_CHECK_U: begin
                        if (error_best < CAL_LOCK_THRESHOLD) begin
                            lock_counter <= lock_counter + 1'b1;
                            if (lock_counter >= CAL_LOCK_COUNT - 1) begin
                                u_locked_reg <= 1'b1;
                                status.u_locked <= 1'b1;
                            end
                        end else begin
                            lock_counter <= '0;
                            if (iteration_counter[4:0] == '0 && phi_step > PHASE_STEP_MIN)
                                phi_step <= phi_step >> PHASE_STEP_DECAY;
                        end
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
    //-------------------------------------------------------------------------
    function automatic logic [ACC_WIDTH-1:0] compute_error(
        input logic signed [DATA_WIDTH_P-1:0] m_real [4],
        input logic signed [DATA_WIDTH_P-1:0] w [4]
    );
        logic signed [DATA_WIDTH_P:0] diff [4];
        logic [ACC_WIDTH-1:0] sum;
        sum = '0;
        for (int i = 0; i < 4; i++) begin
            diff[i] = m_real[i] - w[i];
            sum = sum + (diff[i] * diff[i]);
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
                        SVD_CAL_APPLY_BASIS0_U, SVD_CAL_SETTLE0_U, SVD_CAL_SAMPLE_COL0_U: begin
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
                        CAL_APPLY_BASIS0, CAL_SETTLE0, CAL_SAMPLE_COL0: begin
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
