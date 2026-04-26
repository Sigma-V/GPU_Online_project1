NVCC ?= nvcc
TARGET := bin/cuda_signal_batch
SRC := src/main.cu
NVCCFLAGS := -O3 -std=c++17 -Xcompiler -Wall,-Wextra,-Wpedantic

.PHONY: all clean run-small run-large help

all: $(TARGET)

$(TARGET): $(SRC) | bin
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

bin:
	mkdir -p bin

clean:
	rm -rf bin output/* artifacts/*.txt

run-small: all
	./bin/cuda_signal_batch \
		--input_dir data/small_signals \
		--output_dir output/small \
		--window_radius 4 \
		--threads_per_block 256 \
		--artifact_path artifacts/proof_small_signals.txt

run-large: all
	./bin/cuda_signal_batch \
		--input_dir data/large_signals \
		--output_dir output/large \
		--window_radius 8 \
		--threads_per_block 256 \
		--artifact_path artifacts/proof_large_signals.txt

help:
	@echo "Targets:"
	@echo "  make           Build CUDA binary"
	@echo "  make clean     Remove build and artifact outputs"
	@echo "  make run-small Run small-signal experiment"
	@echo "  make run-large Run large-signal experiment"
