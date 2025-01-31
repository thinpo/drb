cmake_minimum_required(VERSION 3.14)
project(exchange VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(cppzmq REQUIRED)
find_package(nlohmann_json REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Add the exchange executable
add_executable(exchange_engine src/main.cpp)

# Link libraries
target_link_libraries(exchange_engine PRIVATE
    dynamic_ring_buffer
    cppzmq
    nlohmann_json::nlohmann_json
    pthread
)

# Install configuration
install(TARGETS exchange_engine
    RUNTIME DESTINATION bin
)

install(DIRECTORY config/
    DESTINATION etc/exchange
    FILES_MATCHING PATTERN "*.json"
)

# Testing
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif() 