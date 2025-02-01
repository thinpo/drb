cmake_minimum_required(VERSION 3.14)
project(exchange VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(NATS REQUIRED IMPORTED_TARGET libnats)
find_package(nlohmann_json REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    /usr/local/include  # Add this to find system-installed headers
)

# Add link directories
link_directories(
    /usr/local/lib  # Add this to find system-installed libraries
)

# Add the exchange executable
add_executable(exchange_engine src/main.cpp)

# Link libraries
target_link_libraries(exchange_engine PRIVATE
    dynamic_ring_buffer
    PkgConfig::NATS
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