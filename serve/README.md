# SSDF-serve

This is a C++ serving library for modern C++, which provide a simple interface for working with neural network models. Right now it is having those features:

- [Generate TensorRT engine](#convert-onnx-to-tensorrt-engine)
- [Perform inference in multiple backends](#perform-inference)

## Prerequisite

- Nvidia TensorRT
- [fmt](https://fmt.dev/latest/index.html) (until C++20)
- C++17

### Create your custom logger

Some inference backends (i.e. TensorRT) requires a custom logger. To provide a robust logging system, this library request user to provide their definition for **flushLog** function in **ILogger** inferface.

```cpp
void Logger::flushLog(Level level, std::string_view message) const {
  switch (level) {
    case Level::kINTERNAL_ERROR:
      ROS_FATAL_STREAM_NAMED(name_, message);
      break;
    ...
    case Level::kVERBOSE:
      ROS_DEBUG_STREAM_NAMED(name_, message);
      break;
  }
}
```

## Usage

### Implemented backend

- TensorRT: .engine, .trt

### Convert ONNX to TensorRT engine

1. Populate `BuildOptions` and `SystemOptions` in [option.hpp](include/trt/option.hpp)
2. Create a `Generator` instance with those options, then use `generator.getSerializedEngine` to allocate memory for the network. Right now it could only take ONNX model path; using TensorRT's layers is under development. This function return raw pointer, **remember to delete it after saving the model** (or use smart pointer)
3. Use `saveEngine` function in [utils.hpp](include/trt/utils.hpp) to save allocated model to file.

## Perform inference

1. Create a `Session` instance by using `InferenceOptions`. This class will automatically choose the right backend using model's file extension
2. Call `session.doInference` to perform synchronous inference. This function receives a map, which its key is input layer's name and value are host's pointer + size in bytes.
The output results in also a map of (layer's name, buffer in host)

## Add custom backend

1. Provide implementation for `IBackend` interface
2. Register the new backend and its file extensions **in .cpp file**, see `bool registered` in [backend.cpp](src/trt/backend.cpp) for example.
