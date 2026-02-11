cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# 设置项目名称和版本
project(test.bundle VERSION 1.0)

# 指定 C++ 标准
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加一个 macOS bundle 可执行文件
add_executable(test.bundle bundle.c)
target_link_options(test.bundle PUBLIC -bundle)
# 如果有需要，可以链接库
# target_link_libraries(test.bundle SomeLibrary)

# 包含 CMake 的 macOS 应用程序 bundle 模块
# include(CMakeOSXBundleInfo)
