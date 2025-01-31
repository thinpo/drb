# Dynamic Ring Buffer

A modern C++ implementation of a dynamic ring buffer (circular buffer) with advanced features and statistical operations.

## Features

- Dynamic resizing (grows and shrinks automatically)
- Template-based implementation (works with any type)
- STL-compatible iterators
- Statistical operations (mean, median, standard deviation)
- Functional programming features (map, reduce)
- Bulk operations (add, multiply)
- Slicing support
- Custom type support

## Requirements

- C++17 compatible compiler
- CMake 3.10 or higher
- Docker (optional, for development container)

## Building the Project

### Using DevContainer (Recommended)

1. Install Visual Studio Code and the "Remote - Containers" extension
2. Clone the repository
3. Open the project in VS Code
4. Click "Reopen in Container" when prompted
5. The project will be automatically built

### Manual Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Demo Application

The demo application showcases various features of the DynamicRingBuffer:

1. Basic Operations
   - Push/pop operations
   - Automatic resizing
   - Iteration

2. Statistical Operations
   - Mean calculation
   - Standard deviation
   - Median

3. Sensor Reading Simulation
   - Custom type support
   - Timestamped data handling

4. Data Transformations
   - Bulk addition
   - Bulk multiplication
   - Custom mapping functions

## Usage Example

```cpp
#include "dynamic_ring_buffer.hpp"

// Create a buffer of integers
drb::DynamicRingBuffer<int> buffer;

// Add some values
buffer.push(1);
buffer.push(2);
buffer.push(3);

// Perform operations
buffer.add(10);      // Add 10 to each element
buffer.multiply(2);  // Multiply each element by 2

// Calculate statistics
double mean = buffer.mean();
double stddev = buffer.stddev();

// Use with STL algorithms
std::vector<int> vec(buffer.begin(), buffer.end());
```

## License

MIT License 