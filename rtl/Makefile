# Makefile for cocotb mixed-signal co-simulation
#
# This Makefile supports running RTL + Python plant model co-simulation
# using cocotb and verilator

# Default simulator - use Verilator for better SystemVerilog support
# Can override with SIM=icarus, SIM=questa, SIM=xcelium, etc.
SIM ?= verilator
TOPLEVEL_LANG ?= verilog

# Verilator build options
EXTRA_ARGS += --timing
EXTRA_ARGS += -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL

# Source files
VERILOG_SOURCES = \
    $(PWD)/pkg_types.sv \
    $(PWD)/adc_scaler.sv \
    $(PWD)/coherent_matmul_top.sv

# Top-level module
TOPLEVEL = coherent_matmul_top

# Python test module
MODULE = tests.test_cosim

# cocotb options
export COCOTB_REDUCED_LOG_FMT = 1

# Include cocotb's make rules
include $(shell cocotb-config --makefiles)/Makefile.sim

# Additional targets
.PHONY: clean-all waves

clean-all: clean
	rm -rf __pycache__ tests/__pycache__
	rm -rf sim_build
	rm -f results.xml *.vcd *.fst

waves:
	gtkwave dump.vcd &

# Quick test targets
test-unitary:
	$(MAKE) MODULE=tests.test_cosim TESTCASE=test_unitary_calibration

test-svd:
	$(MAKE) MODULE=tests.test_cosim TESTCASE=test_svd_calibration

test-all:
	$(MAKE) MODULE=tests.test_cosim
