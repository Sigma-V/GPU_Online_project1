#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::ostringstream cuda_error_stream;                                  \
      cuda_error_stream << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                        << " -> " << cudaGetErrorString(status);            \
      throw std::runtime_error(cuda_error_stream.str());                     \
    }                                                                        \
  } while (0)

struct Options {
  fs::path input_dir;
  fs::path output_dir;
  fs::path artifact_path = "artifacts/last_run_summary.txt";
  int window_radius = 4;
  int max_files = -1;
  int threads_per_block = 256;
  bool help = false;
};

struct Dataset {
  std::vector<fs::path> files;
  std::vector<std::vector<float>> signals;
  std::vector<int> lengths;
  std::vector<float> mins;
  std::vector<float> ranges;
  int max_length = 0;
  int min_length = 0;
};

struct RunStats {
  float h2d_ms = 0.0f;
  float normalize_ms = 0.0f;
  float smooth_ms = 0.0f;
  float gradient_ms = 0.0f;
  float d2h_ms = 0.0f;
  float total_ms = 0.0f;
  double throughput_million_samples_per_sec = 0.0;
  std::size_t total_samples = 0;
};

void PrintUsage(const char* binary_name) {
  std::cout
      << "Usage:\n"
      << "  " << binary_name
      << " --input_dir <path> --output_dir <path> [options]\n\n"
      << "Required:\n"
      << "  --input_dir <path>           Directory with input CSV signal files\n"
      << "  --output_dir <path>          Directory for processed output files\n\n"
      << "Options:\n"
      << "  --window_radius <int>        Moving-average radius (default: 4)\n"
      << "  --max_files <int>            Limit number of files loaded (default: -1)\n"
      << "  --threads_per_block <int>    CUDA threads per block (default: 256)\n"
      << "  --artifact_path <path>       Summary artifact file path\n"
      << "                               (default: artifacts/last_run_summary.txt)\n"
      << "  --help                       Show this help message\n\n"
      << "Example:\n"
      << "  " << binary_name
      << " --input_dir data/small_signals --output_dir output/small"
      << " --window_radius 4 --artifact_path artifacts/proof_small.txt\n";
}

bool IsFlag(const std::string& token) { return token.rfind("--", 0) == 0; }

int ParseIntOrThrow(const std::string& value, const std::string& name) {
  try {
    return std::stoi(value);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid integer for " + name + ": " + value);
  }
}

Options ParseArgs(int argc, char** argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    std::string token(argv[i]);
    if (!IsFlag(token)) {
      throw std::runtime_error("Unexpected token: " + token);
    }

    std::string key = token;
    std::string value;
    const std::size_t equals_pos = token.find('=');
    if (equals_pos != std::string::npos) {
      key = token.substr(0, equals_pos);
      value = token.substr(equals_pos + 1);
    }

    if (key == "--help") {
      options.help = true;
      continue;
    }

    if (value.empty()) {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for argument: " + key);
      }
      value = argv[++i];
      if (IsFlag(value)) {
        throw std::runtime_error("Missing value for argument: " + key);
      }
    }

    if (key == "--input_dir") {
      options.input_dir = value;
    } else if (key == "--output_dir") {
      options.output_dir = value;
    } else if (key == "--window_radius") {
      options.window_radius = ParseIntOrThrow(value, key);
    } else if (key == "--max_files") {
      options.max_files = ParseIntOrThrow(value, key);
    } else if (key == "--threads_per_block") {
      options.threads_per_block = ParseIntOrThrow(value, key);
    } else if (key == "--artifact_path") {
      options.artifact_path = value;
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }

  if (!options.help) {
    if (options.input_dir.empty()) {
      throw std::runtime_error("--input_dir is required.");
    }
    if (options.output_dir.empty()) {
      throw std::runtime_error("--output_dir is required.");
    }
  }

  if (options.window_radius < 0) {
    throw std::runtime_error("--window_radius must be >= 0.");
  }
  if (options.threads_per_block <= 0) {
    throw std::runtime_error("--threads_per_block must be > 0.");
  }
  if (options.threads_per_block > 1024) {
    throw std::runtime_error("--threads_per_block must be <= 1024.");
  }

  return options;
}

std::string Trim(const std::string& raw) {
  const std::size_t start = raw.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  const std::size_t end = raw.find_last_not_of(" \t\r\n");
  return raw.substr(start, end - start + 1);
}

std::vector<float> ReadSignalCsv(const fs::path& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open input file: " + file_path.string());
  }

  std::vector<float> samples;
  std::string line;
  while (std::getline(file, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }

    std::stringstream stream(trimmed);
    std::string token;
    bool line_had_number = false;

    while (std::getline(stream, token, ',')) {
      const std::string token_trimmed = Trim(token);
      if (token_trimmed.empty()) {
        continue;
      }

      char* end_ptr = nullptr;
      const float parsed = std::strtof(token_trimmed.c_str(), &end_ptr);
      if (end_ptr != token_trimmed.c_str()) {
        samples.push_back(parsed);
        line_had_number = true;
      }
    }

    if (!line_had_number && samples.empty()) {
      continue;
    }
  }

  return samples;
}

void WriteSignalCsv(const fs::path& file_path, const std::vector<float>& values) {
  std::ofstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to write output file: " + file_path.string());
  }

  file << "index,value\n";
  file << std::fixed << std::setprecision(7);
  for (std::size_t i = 0; i < values.size(); ++i) {
    file << i << "," << values[i] << "\n";
  }
}

Dataset LoadDataset(const Options& options) {
  if (!fs::exists(options.input_dir)) {
    throw std::runtime_error("Input directory does not exist: " +
                             options.input_dir.string());
  }

  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(options.input_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path path = entry.path();
    const std::string file_name = path.filename().string();
    if (path.extension() == ".csv" && file_name != "manifest.csv") {
      files.push_back(path);
    }
  }

  std::sort(files.begin(), files.end());

  if (options.max_files > 0 &&
      static_cast<int>(files.size()) > options.max_files) {
    files.resize(options.max_files);
  }

  if (files.empty()) {
    throw std::runtime_error("No CSV files found in input directory: " +
                             options.input_dir.string());
  }

  Dataset dataset;
  dataset.files = files;
  dataset.signals.reserve(files.size());
  dataset.lengths.reserve(files.size());
  dataset.mins.reserve(files.size());
  dataset.ranges.reserve(files.size());

  dataset.max_length = 0;
  dataset.min_length = std::numeric_limits<int>::max();

  for (const fs::path& path : files) {
    std::vector<float> signal = ReadSignalCsv(path);
    if (signal.empty()) {
      throw std::runtime_error("Input file has no numeric samples: " +
                               path.string());
    }

    float min_value = signal[0];
    float max_value = signal[0];
    for (float sample : signal) {
      min_value = std::min(min_value, sample);
      max_value = std::max(max_value, sample);
    }

    const int length = static_cast<int>(signal.size());
    const float range = std::max(max_value - min_value, 1e-6f);

    dataset.max_length = std::max(dataset.max_length, length);
    dataset.min_length = std::min(dataset.min_length, length);
    dataset.lengths.push_back(length);
    dataset.mins.push_back(min_value);
    dataset.ranges.push_back(range);
    dataset.signals.push_back(std::move(signal));
  }

  return dataset;
}

void EnsureDirectory(const fs::path& path) {
  if (!fs::exists(path)) {
    fs::create_directories(path);
  }
}

void EnsureParentDirectory(const fs::path& path) {
  const fs::path parent = path.parent_path();
  if (!parent.empty()) {
    EnsureDirectory(parent);
  }
}

__global__ void NormalizeKernel(const float* input, float* output,
                                const int* lengths, const float* mins,
                                const float* ranges, int signal_length,
                                int num_signals) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_signals * signal_length;
  if (index >= total) {
    return;
  }

  const int signal_id = index / signal_length;
  const int sample_id = index % signal_length;
  const int valid_length = lengths[signal_id];

  if (sample_id >= valid_length) {
    output[index] = 0.0f;
    return;
  }

  const float value = input[index];
  output[index] = (value - mins[signal_id]) / ranges[signal_id];
}

__global__ void MovingAverageKernel(const float* input, float* output,
                                    const int* lengths, int signal_length,
                                    int num_signals, int radius) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_signals * signal_length;
  if (index >= total) {
    return;
  }

  const int signal_id = index / signal_length;
  const int sample_id = index % signal_length;
  const int valid_length = lengths[signal_id];

  if (sample_id >= valid_length) {
    output[index] = 0.0f;
    return;
  }

  const int start = max(0, sample_id - radius);
  const int stop = min(valid_length - 1, sample_id + radius);

  const int base = signal_id * signal_length;
  float sum = 0.0f;
  int count = 0;
  for (int i = start; i <= stop; ++i) {
    sum += input[base + i];
    ++count;
  }

  output[index] = sum / static_cast<float>(count);
}

__global__ void GradientMagnitudeKernel(const float* input, float* output,
                                        const int* lengths, int signal_length,
                                        int num_signals) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_signals * signal_length;
  if (index >= total) {
    return;
  }

  const int signal_id = index / signal_length;
  const int sample_id = index % signal_length;
  const int valid_length = lengths[signal_id];

  if (sample_id >= valid_length) {
    output[index] = 0.0f;
    return;
  }

  const int base = signal_id * signal_length;
  const int left_index = max(0, sample_id - 1);
  const int right_index = min(valid_length - 1, sample_id + 1);

  const float left_value = input[base + left_index];
  const float right_value = input[base + right_index];
  const float gradient = 0.5f * (right_value - left_value);
  output[index] = fabsf(gradient);
}

void WriteOutputs(const Dataset& dataset, const Options& options,
                  const std::vector<float>& smoothed,
                  const std::vector<float>& gradient) {
  EnsureDirectory(options.output_dir);

  const int signal_length = dataset.max_length;
  const int num_signals = static_cast<int>(dataset.files.size());

  fs::path metrics_path = options.output_dir / "signal_metrics.csv";
  std::ofstream metrics_file(metrics_path);
  if (!metrics_file.is_open()) {
    throw std::runtime_error("Failed to write metrics CSV: " +
                             metrics_path.string());
  }

  metrics_file
      << "input_file,num_samples,smooth_min,smooth_max,smooth_mean,"
      << "gradient_mean,gradient_max\n";
  metrics_file << std::fixed << std::setprecision(7);

  for (int signal_id = 0; signal_id < num_signals; ++signal_id) {
    const int valid_length = dataset.lengths[signal_id];
    const int base = signal_id * signal_length;

    std::vector<float> smooth_signal(valid_length);
    std::vector<float> gradient_signal(valid_length);
    for (int i = 0; i < valid_length; ++i) {
      smooth_signal[i] = smoothed[base + i];
      gradient_signal[i] = gradient[base + i];
    }

    float smooth_min = smooth_signal[0];
    float smooth_max = smooth_signal[0];
    double smooth_sum = 0.0;
    double gradient_sum = 0.0;
    float gradient_max = gradient_signal[0];

    for (int i = 0; i < valid_length; ++i) {
      smooth_min = std::min(smooth_min, smooth_signal[i]);
      smooth_max = std::max(smooth_max, smooth_signal[i]);
      gradient_max = std::max(gradient_max, gradient_signal[i]);
      smooth_sum += smooth_signal[i];
      gradient_sum += gradient_signal[i];
    }

    const double smooth_mean = smooth_sum / static_cast<double>(valid_length);
    const double gradient_mean =
        gradient_sum / static_cast<double>(valid_length);

    const std::string stem = dataset.files[signal_id].stem().string();
    WriteSignalCsv(options.output_dir / (stem + "_smooth.csv"), smooth_signal);
    WriteSignalCsv(options.output_dir / (stem + "_gradient.csv"),
                   gradient_signal);

    metrics_file << dataset.files[signal_id].filename().string() << ","
                 << valid_length << "," << smooth_min << "," << smooth_max
                 << "," << smooth_mean << "," << gradient_mean << ","
                 << gradient_max << "\n";
  }
}

void WriteArtifact(const Dataset& dataset, const Options& options,
                   const RunStats& stats, const std::string& gpu_name) {
  EnsureParentDirectory(options.artifact_path);

  std::ofstream artifact_file(options.artifact_path);
  if (!artifact_file.is_open()) {
    throw std::runtime_error("Failed to write artifact file: " +
                             options.artifact_path.string());
  }

  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

  artifact_file << "CUDA Signal Processing Run Summary\n";
  artifact_file << "=================================\n";
  artifact_file << "Timestamp: " << std::put_time(std::localtime(&now_time),
                                                  "%Y-%m-%d %H:%M:%S")
                << "\n";
  artifact_file << "GPU: " << gpu_name << "\n";
  artifact_file << "Input directory: " << options.input_dir << "\n";
  artifact_file << "Output directory: " << options.output_dir << "\n";
  artifact_file << "Input files processed: " << dataset.files.size() << "\n";
  artifact_file << "Min length: " << dataset.min_length << "\n";
  artifact_file << "Max length: " << dataset.max_length << "\n";
  artifact_file << "Total samples: " << stats.total_samples << "\n";
  artifact_file << "Window radius: " << options.window_radius << "\n";
  artifact_file << "Threads per block: " << options.threads_per_block << "\n";
  artifact_file << "\nTiming (ms):\n";
  artifact_file << "  H2D copy: " << stats.h2d_ms << "\n";
  artifact_file << "  Normalize kernel: " << stats.normalize_ms << "\n";
  artifact_file << "  Smooth kernel: " << stats.smooth_ms << "\n";
  artifact_file << "  Gradient kernel: " << stats.gradient_ms << "\n";
  artifact_file << "  D2H copy: " << stats.d2h_ms << "\n";
  artifact_file << "  Total: " << stats.total_ms << "\n";
  artifact_file << "Throughput (million samples/second): "
                << stats.throughput_million_samples_per_sec << "\n";
  artifact_file << "\nGenerated files:\n";
  artifact_file << "  - Per-input *_smooth.csv\n";
  artifact_file << "  - Per-input *_gradient.csv\n";
  artifact_file << "  - signal_metrics.csv\n";
}

RunStats ExecutePipeline(const Dataset& dataset, const Options& options,
                         std::vector<float>* smoothed_out,
                         std::vector<float>* gradient_out) {
  const int num_signals = static_cast<int>(dataset.signals.size());
  const int signal_length = dataset.max_length;
  const std::size_t total_values =
      static_cast<std::size_t>(num_signals) * signal_length;

  std::vector<float> host_input(total_values, 0.0f);
  for (int signal_id = 0; signal_id < num_signals; ++signal_id) {
    const std::vector<float>& signal = dataset.signals[signal_id];
    const std::size_t base = static_cast<std::size_t>(signal_id) * signal_length;
    std::copy(signal.begin(), signal.end(), host_input.begin() + base);
  }

  smoothed_out->assign(total_values, 0.0f);
  gradient_out->assign(total_values, 0.0f);

  float* d_input = nullptr;
  float* d_normalized = nullptr;
  float* d_smoothed = nullptr;
  float* d_gradient = nullptr;
  int* d_lengths = nullptr;
  float* d_mins = nullptr;
  float* d_ranges = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, total_values * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normalized, total_values * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_smoothed, total_values * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gradient, total_values * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lengths, num_signals * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_mins, num_signals * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ranges, num_signals * sizeof(float)));

  cudaEvent_t total_start;
  cudaEvent_t total_stop;
  cudaEvent_t h2d_start;
  cudaEvent_t h2d_stop;
  cudaEvent_t normalize_start;
  cudaEvent_t normalize_stop;
  cudaEvent_t smooth_start;
  cudaEvent_t smooth_stop;
  cudaEvent_t gradient_start;
  cudaEvent_t gradient_stop;
  cudaEvent_t d2h_start;
  cudaEvent_t d2h_stop;

  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&h2d_start));
  CUDA_CHECK(cudaEventCreate(&h2d_stop));
  CUDA_CHECK(cudaEventCreate(&normalize_start));
  CUDA_CHECK(cudaEventCreate(&normalize_stop));
  CUDA_CHECK(cudaEventCreate(&smooth_start));
  CUDA_CHECK(cudaEventCreate(&smooth_stop));
  CUDA_CHECK(cudaEventCreate(&gradient_start));
  CUDA_CHECK(cudaEventCreate(&gradient_stop));
  CUDA_CHECK(cudaEventCreate(&d2h_start));
  CUDA_CHECK(cudaEventCreate(&d2h_stop));

  const int blocks = static_cast<int>(
      (total_values + options.threads_per_block - 1) / options.threads_per_block);

  RunStats stats;
  stats.total_samples = 0;
  for (int length : dataset.lengths) {
    stats.total_samples += static_cast<std::size_t>(length);
  }

  CUDA_CHECK(cudaEventRecord(total_start));

  CUDA_CHECK(cudaEventRecord(h2d_start));
  CUDA_CHECK(cudaMemcpy(d_input, host_input.data(), total_values * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lengths, dataset.lengths.data(),
                        num_signals * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mins, dataset.mins.data(), num_signals * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ranges, dataset.ranges.data(),
                        num_signals * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(h2d_stop));

  CUDA_CHECK(cudaEventRecord(normalize_start));
  NormalizeKernel<<<blocks, options.threads_per_block>>>(
      d_input, d_normalized, d_lengths, d_mins, d_ranges, signal_length,
      num_signals);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(normalize_stop));

  CUDA_CHECK(cudaEventRecord(smooth_start));
  MovingAverageKernel<<<blocks, options.threads_per_block>>>(
      d_normalized, d_smoothed, d_lengths, signal_length, num_signals,
      options.window_radius);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(smooth_stop));

  CUDA_CHECK(cudaEventRecord(gradient_start));
  GradientMagnitudeKernel<<<blocks, options.threads_per_block>>>(
      d_smoothed, d_gradient, d_lengths, signal_length, num_signals);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(gradient_stop));

  CUDA_CHECK(cudaEventRecord(d2h_start));
  CUDA_CHECK(cudaMemcpy(smoothed_out->data(), d_smoothed,
                        total_values * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gradient_out->data(), d_gradient,
                        total_values * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(d2h_stop));

  CUDA_CHECK(cudaEventRecord(total_stop));
  CUDA_CHECK(cudaEventSynchronize(total_stop));

  CUDA_CHECK(cudaEventElapsedTime(&stats.h2d_ms, h2d_start, h2d_stop));
  CUDA_CHECK(cudaEventElapsedTime(&stats.normalize_ms, normalize_start,
                                  normalize_stop));
  CUDA_CHECK(cudaEventElapsedTime(&stats.smooth_ms, smooth_start, smooth_stop));
  CUDA_CHECK(cudaEventElapsedTime(&stats.gradient_ms, gradient_start,
                                  gradient_stop));
  CUDA_CHECK(cudaEventElapsedTime(&stats.d2h_ms, d2h_start, d2h_stop));
  CUDA_CHECK(cudaEventElapsedTime(&stats.total_ms, total_start, total_stop));

  if (stats.total_ms > 0.0f) {
    stats.throughput_million_samples_per_sec =
        static_cast<double>(stats.total_samples) /
        (static_cast<double>(stats.total_ms) * 1000.0);
  }

  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(h2d_start));
  CUDA_CHECK(cudaEventDestroy(h2d_stop));
  CUDA_CHECK(cudaEventDestroy(normalize_start));
  CUDA_CHECK(cudaEventDestroy(normalize_stop));
  CUDA_CHECK(cudaEventDestroy(smooth_start));
  CUDA_CHECK(cudaEventDestroy(smooth_stop));
  CUDA_CHECK(cudaEventDestroy(gradient_start));
  CUDA_CHECK(cudaEventDestroy(gradient_stop));
  CUDA_CHECK(cudaEventDestroy(d2h_start));
  CUDA_CHECK(cudaEventDestroy(d2h_stop));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_normalized));
  CUDA_CHECK(cudaFree(d_smoothed));
  CUDA_CHECK(cudaFree(d_gradient));
  CUDA_CHECK(cudaFree(d_lengths));
  CUDA_CHECK(cudaFree(d_mins));
  CUDA_CHECK(cudaFree(d_ranges));

  return stats;
}

int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    if (options.help) {
      PrintUsage(argv[0]);
      return 0;
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
      throw std::runtime_error("No CUDA-capable GPU detected.");
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp device_properties;
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));

    std::cout << "Using GPU: " << device_properties.name << "\n";
    std::cout << "Loading CSV signals from: " << options.input_dir << "\n";

    const Dataset dataset = LoadDataset(options);
    std::cout << "Loaded " << dataset.files.size() << " files"
              << " (min length=" << dataset.min_length
              << ", max length=" << dataset.max_length << ")\n";

    std::vector<float> smoothed;
    std::vector<float> gradient;
    const RunStats stats =
        ExecutePipeline(dataset, options, &smoothed, &gradient);

    std::cout << "Writing processed outputs to: " << options.output_dir << "\n";
    WriteOutputs(dataset, options, smoothed, gradient);

    WriteArtifact(dataset, options, stats, device_properties.name);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nRun complete.\n";
    std::cout << "  Files processed: " << dataset.files.size() << "\n";
    std::cout << "  Total samples: " << stats.total_samples << "\n";
    std::cout << "  Total time (ms): " << stats.total_ms << "\n";
    std::cout << "  Throughput (million samples/s): "
              << stats.throughput_million_samples_per_sec << "\n";
    std::cout << "  Artifact: " << options.artifact_path << "\n";

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    std::cerr << "Use --help for usage details.\n";
    return 1;
  }
}
