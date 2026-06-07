add_executable(gpu-runner
  gpu-runner.cpp
)
target_link_libraries(gpu-runner PRIVATE
  -lcuda
)
