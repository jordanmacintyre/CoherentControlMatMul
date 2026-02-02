# Makefile for cocotb + Verilator simulation
# Coherent Photonic 2x2 Matrix Multiply

# Simulation configuration
SIM ?= verilator
TOPLEVEL_LANG ?= verilog

# RTL sources (pkg_types.sv must come first for type definitions)
VERILOG_SOURCES = rtl/pkg_types.sv rtl/coherent_matmul_top.sv
TOPLEVEL = coherent_matmul_top

# Default test module
MODULE ?= tests.test_calibration

# Python path for plant module
export PYTHONPATH := $(PWD):$(PYTHONPATH)

# Verilator-specific flags
ifeq ($(SIM),verilator)
    EXTRA_ARGS += --trace --trace-structs
    EXTRA_ARGS += -Wno-fatal
    EXTRA_ARGS += -Wno-WIDTHEXPAND
    EXTRA_ARGS += -Wno-WIDTHTRUNC
    EXTRA_ARGS += -Wno-UNUSEDSIGNAL
    EXTRA_ARGS += -Wno-UNUSEDPARAM
    EXTRA_ARGS += --timing
    # Build directory
    SIM_BUILD = sim/obj_dir
endif

# Include cocotb makefile
include $(shell cocotb-config --makefiles)/Makefile.sim

# Convenience targets
.PHONY: test-cal test-compute test-drift test-all clean-sim lint help

test-cal:
	MODULE=tests.test_calibration $(MAKE) sim

test-compute:
	MODULE=tests.test_compute $(MAKE) sim

test-drift:
	MODULE=tests.test_drift $(MAKE) sim

test-all:
	pytest tests/ -v

clean-sim:
	rm -rf sim/obj_dir sim/results/*.vcd sim/results/*.json
	rm -rf __pycache__ tests/__pycache__ plant/__pycache__

lint:
	ruff check plant/ tests/

help:
	@echo "Available targets:"
	@echo "  make sim          - Run simulation with MODULE=tests.test_calibration"
	@echo "  make test-cal     - Run calibration tests"
	@echo "  make test-compute - Run compute correctness tests"
	@echo "  make test-drift   - Run drift robustness tests"
	@echo "  make test-all     - Run all tests with pytest"
	@echo "  make clean-sim    - Clean simulation artifacts"
	@echo "  make lint         - Run ruff linter"
