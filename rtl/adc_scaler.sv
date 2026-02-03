//-----------------------------------------------------------------------------
// adc_scaler.sv
// Converts accumulated ADC samples to Q1.15 fixed-point
//
// Interface Contract:
//   ADC full-scale (±2047 LSB) represents ±1.0 optical field amplitude
//   Q1.15 full-scale (±32767) represents ±1.0
//
// Scaling Math:
//   With 8 accumulated samples of max 2047 each:
//     accumulator_max = 8 × 2047 = 16376
//   Target Q1.15 for 1.0:
//     q15_max = 32767
//   Scale factor:
//     32767 / 16376 ≈ 2.0
//   Implementation:
//     q15_out = accumulator << 1 (multiply by 2)
//
// This matches production coherent optical systems where the analog front-end
// (TIA + VGA/AGC) is designed to fill the ADC dynamic range.
//-----------------------------------------------------------------------------

module adc_scaler #(
    parameter int ADC_WIDTH   = 12,
    parameter int ACC_SAMPLES = 8,
    parameter int Q15_WIDTH   = 16,
    parameter int SCALE_SHIFT = 1   // accumulator << 1 = (accum/8)*16 ≈ accum*2
)(
    input  logic signed [ADC_WIDTH+3:0] accumulator,  // 15-bit signed (8 samples of 12-bit)
    output logic signed [Q15_WIDTH-1:0] q15_out
);

    // Extended width for scaling before saturation
    logic signed [ADC_WIDTH+4:0] scaled;

    always_comb begin
        // Apply scaling (left arithmetic shift preserves sign)
        scaled = accumulator <<< SCALE_SHIFT;

        // Saturate to Q1.15 range [-32768, 32767]
        // Use decimal literals to avoid bit-width mismatch with 17-bit scaled
        if (scaled > 32767)
            q15_out = 16'sh7FFF;
        else if (scaled < -32768)
            q15_out = 16'sh8000;
        else
            q15_out = scaled[Q15_WIDTH-1:0];
    end

endmodule
