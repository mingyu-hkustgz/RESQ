set(CMAKE_CXX_STANDARD 17)

add_executable(index_hnsw HNSW/hnsw_index.cpp)
add_executable(search_hnsw HNSW/hnsw_search.cpp)
add_executable(rabit_index_hnsw HNSW/rabit_index.cpp)
add_executable(res_index_hnsw HNSW/res_index.cpp)
add_executable(rabit_search_hnsw HNSW/rabit_search.cpp)
add_executable(res_search_hnsw HNSW/res_search.cpp)

add_executable(rabit_index RaBitQ/index.cpp)

add_executable(rabit_disk_index RaBitQ/index.cpp)
target_compile_definitions(rabit_disk_index PRIVATE DISK_SCAN FAST_SCAN)

add_executable(rabit_fast_scan_search RaBitQ/search.cpp)
target_compile_definitions(rabit_fast_scan_search PRIVATE FAST_SCAN)

add_executable(rabit_scan_search RaBitQ/search.cpp)
target_compile_definitions(rabit_scan_search PRIVATE SCAN)

add_executable(rabit_fast_disk_search RaBitQ/disk_search.cpp)
target_compile_definitions(rabit_fast_disk_search PRIVATE FAST_SCAN DISK_SCAN)

add_executable(res_index ResBitQ/res_index.cpp)
add_executable(res_split_index ResBitQ/res_index.cpp)
add_executable(res_large_index ResBitQ/res_index.cpp)
target_compile_definitions(res_split_index PRIVATE RESIDUAL_SPLIT)
target_compile_definitions(res_large_index PRIVATE RESIDUAL_SPLIT LARGE_DATA)

add_executable(res_disk_index ResBitQ/res_index.cpp)
target_compile_definitions(res_disk_index PRIVATE DISK_SCAN)

add_executable(res_search ResBitQ/res_search.cpp)
add_executable(res_split_search ResBitQ/res_search.cpp)
add_executable(res_large_search ResBitQ/res_search.cpp)
target_compile_definitions(res_split_search PRIVATE RESIDUAL_SPLIT)
target_compile_definitions(res_large_search PRIVATE RESIDUAL_SPLIT LARGE_DATA)

add_executable(res_disk_search ResBitQ/disk_res_search.cpp)
target_compile_definitions(res_disk_search PRIVATE DISK_SCAN)

