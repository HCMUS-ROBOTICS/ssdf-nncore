#pragma once
#include <NvInfer.h>

#include <array>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ssdf::serve::trt {
enum class SparsityFlag { kDISABLE, kENABLE, kFORCE };
enum class TimingCacheMode { kDISABLE, kLOCAL, kGLOBAL };

using IOFormat = std::pair<nvinfer1::DataType, nvinfer1::TensorFormats>;
using ShapeRange = std::array<std::vector<int>, nvinfer1::EnumMax<nvinfer1::OptProfileSelector>()>;

struct BuildOptions {
  int max_batch{0};
  int workspace_mb{2000};
  int min_timing{-1}, avg_timing{-1};
  bool tf32{true}, fp16{false}, int8{false};
  bool strict_types{false};
  bool safe{false};
  bool consistency{false};
  bool restricted{false};
  bool save{false};
  bool load{false};
  bool refittable{false};
  SparsityFlag sparsity{SparsityFlag::kDISABLE};
  nvinfer1::ProfilingVerbosity profiling_verbosity{nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY};
  std::filesystem::path calibration;
  std::unordered_map<std::string, ShapeRange> shapes;
  std::unordered_map<std::string, ShapeRange> shapes_calib;
  std::vector<IOFormat> input_formats, output_formats;
  nvinfer1::TacticSources enabled_tactics{0};
  nvinfer1::TacticSources disabled_tactics{0};
  TimingCacheMode timing_cache_mode{TimingCacheMode::kLOCAL};
  std::filesystem::path timing_cache_file;
};

struct SystemOptions {
  int device{0};
  int dla_core{-1};
  bool fallback{false};
  std::vector<std::filesystem::path> plugin_paths;
};

}  // namespace ssdf::serve::trt
