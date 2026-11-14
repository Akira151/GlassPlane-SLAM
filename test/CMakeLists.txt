set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/test)

#为test整体添加常用的库和include
link_libraries(${OpenCV_LIBRARIES} Eigen3::Eigen ${PROJECT_NAME})
include_directories(${PROJECT_SOURCE_DIR}/include/)

#test_keyplane
add_executable(test_keyplane test_keyplane.cc)

add_executable(test_maskratio test_maskratio.cc)


# message("111111111111111111111111111111111{}",${PROJECT_SOURCE_DIR})

#test_inline 注释：test_inline是个反例，会编译报错。
# add_executable(test_inline test_inline/test_inline1.cc test_inline/test_inline2.cc test_inline/main.cc)
# target_include_directories(test_inline PRIVATE
#     ${PROJECT_SOURCE_DIR}/test/test_inline/
# )

