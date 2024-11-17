set(CMAKE_CXX_STANDARD 17)

add_executable(rabit_index RaBitQ/index.cpp)

add_executable(rabit_disk_index RaBitQ/index.cpp)

add_executable(rabit_fast_scan_search RaBitQ/search.cpp)
target_compile_definitions(rabit_fast_scan_search PRIVATE FAST_SCAN)

add_executable(rabit_scan_search RaBitQ/search.cpp)
target_compile_definitions(rabit_scan_search PRIVATE SCAN)

add_executable(rabit_disk_scan_search RaBitQ/disk_search.cpp)
target_compile_definitions(rabit_disk_scan_search PRIVATE FAST_SCAN)
target_link_libraries(rabit_disk_scan_search ${DISKANN_ASYNC_LIB})

add_executable(res_index ResBitQ/res_index.cpp)
add_executable(res_split_index ResBitQ/res_index.cpp)
target_compile_definitions(res_split_index PRIVATE RESIDUAL_SPLIT)

add_executable(res_disk_index ResBitQ/res_index.cpp)

add_executable(res_search ResBitQ/res_search.cpp)

add_executable(res_split_search ResBitQ/res_search.cpp)
target_compile_definitions(res_split_search PRIVATE RESIDUAL_SPLIT)

add_executable(res_disk_search ResBitQ/res_search.cpp)

add_executable(convert-512-aligned App/convert-to-512-aligned.cpp)

add_executable(test_disk_io ResBitQ/local_test_disk_io.cpp)
target_link_libraries(test_disk_io ${DISKANN_ASYNC_LIB})