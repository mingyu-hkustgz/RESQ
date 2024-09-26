set(CMAKE_CXX_STANDARD 17)

add_executable(rabit_index RaBitQ/index.cpp)

add_executable(rabit_fast_scan_search RaBitQ/search.cpp)
target_compile_definitions(rabit_fast_scan_search PRIVATE FAST_SCAN)
add_executable(rabit_scan_search RaBitQ/search.cpp)
target_compile_definitions(rabit_scan_search PRIVATE SCAN)

add_executable(res_index ResBitQ/res_index.cpp)
add_executable(res_split_index ResBitQ/res_index.cpp)
target_compile_definitions(res_split_index PRIVATE RESIDUAL_SPLIT)

add_executable(res_search ResBitQ/res_search.cpp)
add_executable(res_split_search ResBitQ/res_search.cpp)
target_compile_definitions(res_split_search PRIVATE RESIDUAL_SPLIT)

add_executable(graph_rabit_search GraphBitQ/graph_rabitq_search.cpp)
add_executable(graph_rabit_index GraphBitQ/graph_rabitq_index.cpp)
add_executable(graph_res_search GraphBitQ/graph_res_search.cpp)
add_executable(graph_res_index GraphBitQ/graph_res_index.cpp)
add_executable(graph_naive_search GraphBitQ/graph_naive_search.cpp)

#please comment below which is used for local test
#add_executable(test_res_bit ResBitQ/local_test_res_bit.cpp)