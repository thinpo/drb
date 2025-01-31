find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(ring_buffer_tests ring_buffer_tests.cpp)
target_link_libraries(ring_buffer_tests PRIVATE 
    dynamic_ring_buffer
    GTest::GTest 
    GTest::Main
    pthread
)

add_executable(market_data_tests market_data_tests.cpp)
target_link_libraries(market_data_tests PRIVATE 
    dynamic_ring_buffer
    GTest::GTest 
    GTest::Main
    pthread
)

gtest_discover_tests(ring_buffer_tests)
gtest_discover_tests(market_data_tests) 