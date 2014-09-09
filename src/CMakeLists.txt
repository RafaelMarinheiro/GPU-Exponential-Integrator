if (BUILD_TESTS)
	include_directories(${MAGMA_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
	add_executable(magma_test magma_test/example.c
				   magma_test/zfill.c magma_test/zfill.h)
	target_link_libraries(magma_test
						  ${MAGMA_LIBRARIES}
						  ${CUDA_LIBRARIES}
						  ${CUDA_CUBLAS_LIBRARIES}
						  ${MKL_LIBRARY}
						  ${MKL_SCALAPACK_LIBRARY}
						  )
endif (BUILD_TESTS)