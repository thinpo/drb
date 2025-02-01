cmake_minimum_required(VERSION 3.14)
project(drb VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find pthread package
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Enable AVX2 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

# Options
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library target (header-only, but useful for installation and dependency management)
add_library(dynamic_ring_buffer INTERFACE)
target_include_directories(dynamic_ring_buffer INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(dynamic_ring_buffer INTERFACE Threads::Threads)

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Add exchange engine
add_subdirectory(exchange)

# Tests
if(BUILD_TESTS)
    # Add GoogleTest subdirectory
    add_subdirectory(third_party/googletest)
    enable_testing()

    # Add test executable
    add_executable(ring_buffer_test tests/ring_buffer_test.cpp)
    target_link_libraries(ring_buffer_test 
        GTest::gtest 
        GTest::gtest_main 
        Threads::Threads
    )

    # Include directories
    target_include_directories(ring_buffer_test PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include
    )

    # Add test
    add_test(NAME ring_buffer_test COMMAND ring_buffer_test)
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS dynamic_ring_buffer
    EXPORT dynamic_ring_buffer-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT dynamic_ring_buffer-targets
    FILE dynamic_ring_buffer-targets.cmake
    NAMESPACE dynamic_ring_buffer::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dynamic_ring_buffer
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dynamic_ring_buffer-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dynamic_ring_buffer-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/dynamic_ring_buffer-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dynamic_ring_buffer
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dynamic_ring_buffer-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/dynamic_ring_buffer-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dynamic_ring_buffer
)

# Enable optimizations and debug info
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

# Enable thread sanitizer for testing
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
endif() 