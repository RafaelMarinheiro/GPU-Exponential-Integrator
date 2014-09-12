if (BUILD_LIB)
	include_directories(${MAGMA_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
	add_library(neti
				gpu_sparse_neti.h
				neti/gpu_sparse_neti_s.c neti/gpu_sparse_neti_d.c)
	target_link_libraries(neti
						  ${MAGMA_LIBRARIES}
						  ${CUDA_LIBRARIES}
						  ${CUDA_CUBLAS_LIBRARIES}
						  ${MKL_LIBRARY}
						  ${MKL_SCALAPACK_LIBRARY}
						  )
endif (BUILD_LIB)

if (BUILD_TESTS)
	include_directories(${MAGMA_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
	add_executable(magma_test test/magma_test/example.c
				   test/magma_test/zfill.c test/magma_test/zfill.h)
	target_link_libraries(magma_test
						  ${MAGMA_LIBRARIES}
						  ${CUDA_LIBRARIES}
						  ${CUDA_CUBLAS_LIBRARIES}
						  ${MKL_LIBRARY}
						  ${MKL_SCALAPACK_LIBRARY}
						  )
endif (BUILD_TESTS)