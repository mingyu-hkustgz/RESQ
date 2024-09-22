set(CMAKE_CXX_STANDARD 17)

add_executable(rabit_index RaBitQ/index.cpp)

add_executable(rabit_search RaBitQ/search.cpp)

add_executable(res_index ResBitQ/res_index.cpp)
add_executable(res_split_index ResBitQ/res_index.cpp)
target_compile_definitions(res_split_index PRIVATE RESIDUAL_SPLIT)

add_executable(res_search ResBitQ/res_search.cpp)
add_executable(res_split_search ResBitQ/res_search.cpp)
target_compile_definitions(res_split_search PRIVATE RESIDUAL_SPLIT)

add_executable(test_res_bit ResBitQ/local_test_res_bit.cpp)