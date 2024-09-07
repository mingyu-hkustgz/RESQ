set(CMAKE_CXX_STANDARD 17)

add_executable(rabit_index RaBitQ/index.cpp)

add_executable(rabit_search RaBitQ/search.cpp)

add_executable(res_index ResBitQ/res_index.cpp)

add_executable(res_search ResBitQ/res_search.cpp)

add_executable(test_res_bit ResBitQ/test_res_bit.cpp)