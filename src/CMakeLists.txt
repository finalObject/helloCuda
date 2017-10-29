#cuda_add_executable(main main.cu)

cuda_add_executable(hey hey.cu)

cuda_add_executable(image dispImage.cu)
target_link_libraries(image ${OpenCV_LIBS})

cuda_add_executable(cvl cvl.cu)
target_link_libraries(cvl ${OpenCV_LIBS})

