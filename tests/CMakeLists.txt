# No need to find GTest package as we're using FetchContent
# find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(ring_buffer_tests ring_buffer_tests.cpp)
target_link_libraries(ring_buffer_tests PRIVATE 
    dynamic_ring_buffer
    gtest 
    gtest_main
    pthread
)

add_executable(market_data_tests market_data_tests.cpp)
target_link_libraries(market_data_tests PRIVATE 
    dynamic_ring_buffer
    gtest 
    gtest_main
    pthread
)

add_executable(mapreduce_tests mapreduce_tests.cpp)
target_link_libraries(mapreduce_tests PRIVATE 
    dynamic_ring_buffer
    gtest 
    gtest_main
    pthread
)

gtest_discover_tests(ring_buffer_tests)
gtest_discover_tests(market_data_tests)
gtest_discover_tests(mapreduce_tests) 