#pragma once
#include <NvInfer.h>
#include <fmt/core.h>

#include <mutex>
#include <string_view>
#include <utility>

namespace ssdf::serve {
/**
 * @brief Interface to handle logger
 *
 */
class ILogger {
 public:
  using Level = nvinfer1::ILogger::Severity;

  explicit ILogger(Level level = Level::kINFO) : level_(level) {}

  template <typename... Args>
  void debug(std::string_view pattern, Args &&...args) const {
    log(Level::kVERBOSE, pattern, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void info(std::string_view pattern, Args &&...args) const {
    log(Level::kINFO, pattern, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void warn(std::string_view pattern, Args &&...args) const {
    log(Level::kWARNING, pattern, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void error(std::string_view pattern, Args &&...args) const {
    log(Level::kERROR, pattern, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void fatal(std::string_view pattern, Args &&...args) const {
    log(Level::kINTERNAL_ERROR, pattern, std::forward<Args>(args)...);
  }

  nvinfer1::ILogger &getTRTLogger() { return trt_logger_; }

  Level getCurrentLevel() const { return level_; }

 protected:
  // TensorRT internal logger
  class TRTLogger : public nvinfer1::ILogger {
   public:
    explicit TRTLogger(const serve::ILogger &parent) : parent_(parent) {}
    inline void log(Severity severity, const char *message) noexcept override {
      parent_.log(severity, message);
    }

   private:
    const serve::ILogger &parent_;
  } trt_logger_{*this};

  const Level level_;
  mutable std::mutex mutex_;

  template <typename... Args>
  void log(Level level, std::string_view pattern, Args &&...args) const {
    log_(level, pattern, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void log_(Level level, std::string_view pattern, Args &&...args) const {
    if (level > level_) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    flushLog(level, fmt::vformat(pattern, fmt::make_format_args(args...)));
  }

  virtual void flushLog(Level level, std::string_view message) const = 0;
};
}  // namespace ssdf::serve
