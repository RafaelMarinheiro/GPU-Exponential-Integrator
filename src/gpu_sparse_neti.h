// The MIT License (MIT)

// Copyright (c) 2014 Rafael Farias Marinheiro

// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "neti_types.h"

/**
 * @brief A function that return the required buffer size for the gpu_crs_symmetric_sparse_neti_s algorithm. \see{gpu_crs_symmetric_sparse_neti_s}
 * @details [long description]
 * 
 * @param[in] ndof Number of degrees of freedom
 * @param[in] krylov_length The dimension of the krylov space
 * @param[out] gpu_int_work_space_size Number of elements of a gpu int buffer
 * @param[out] gpu_float_work_space_size Number of elements of a gpu float buffer
 * @param[out] cpu_float_work_space_size Number of elements of a cpu float buffer
 * @return Returns the buffer size for each buffer type
 */
int gpu_crs_symmetric_sparse_neti_s_get_size(int ndof, int krylov_length,
											 int * gpu_int_work_space_size,
											 int * gpu_float_work_space_size,
											 int * cpu_float_work_space_size);

/**
 * @brief A GPU implementation of the NETI algorithm [Michels et Al 2014]
 * @details [long description]
 * 
 * @param[in] ndof Number of degrees of freedom
 * @param[in] massMatrix CRS Sparse Representation of the Mass Matrix (a ndof.ndof matrix)
 * @param[in] dampingMatrix CRS Sparse Representation of the Damping Matrix (a ndof.ndof matrix)
 * @param[in] stiffnessMatrix CRS Sparse Representation of the Stiffness Matrix  (a ndof.ndof matrix)
 * @param[in] forceVector Force Vector (a ndof vector)
 * @param[in,out] oldPosition The old position (a ndof vector)
 * @param[in,out] currentPosition The current position (a ndof vector)
 * @param[in] gpu_int_work_space Integer array allocated in gpu \see{gpu_crs_symmetric_sparse_neti_s_get_size}
 * @param[in] gpu_float_work_space Float array allocated in gpu
 * @param[in] cpu_float_work_space Float array allocated in cpu
 * @param[in] work_space Pre-allocated work space (@)
 * @param[in] deltaTime The timestep
 * @param[in] krylov_length The dimension of the krylov space
 * @return oldPosition will contain the old currentPosition, while currentPosition will contain the new position
 */
int gpu_crs_symmetric_sparse_neti_s(int ndof,
									const sparse_s_matrix_t * invMassMatrix,
									const sparse_s_matrix_t * dampingMatrix,
									const sparse_s_matrix_t * stiffnessMatrix,
									const float * forceVector,
									float * oldPosition,
									float * currentPosition,
									magma_int_t * gpu_int_work_space,
									float * gpu_float_work_space,
									float * cpu_float_work_space,
									float deltaTime,
									int krylov_length
									);


/**
 * @brief A function that return the required buffer size for the gpu_crs_symmetric_sparse_neti_d algorithm. \see{gpu_crs_symmetric_sparse_neti_d}
 * @details [long description]
 * 
 * @param[in] ndof Number of degrees of freedom
 * @param[in] krylov_length The dimension of the krylov space
 * @param[out] gpu_int_work_space_size Number of elements of a gpu int buffer
 * @param[out] gpu_double_work_space_size Number of elements of a gpu double buffer
 * @param[out] cpu_double_work_space_size Number of elements of a cpu double buffer
 * @return Returns the buffer size for each buffer type
 */
int gpu_crs_symmetric_sparse_neti_s_get_size(int ndof, int krylov_length,
											 int * gpu_int_work_space_size,
											 int * gpu_float_work_space_size,
											 int * cpu_float_work_space_size);

/**
 * @brief A GPU implementation of the NETI algorithm [Michels et Al 2014]
 * @details [long description]
 * 
 * @param[in] ndof Number of degrees of freedom
 * @param[in] massMatrix CRS Sparse Representation of the Mass Matrix (a ndof.ndof matrix)
 * @param[in] dampingMatrix CRS Sparse Representation of the Damping Matrix (a ndof.ndof matrix)
 * @param[in] stiffnessMatrix CRS Sparse Representation of the Stiffness Matrix  (a ndof.ndof matrix)
 * @param[in] forceVector Force Vector (a ndof vector)
 * @param[in,out] oldPosition The old position (a ndof vector)
 * @param[in,out] currentPosition The current position (a ndof vector)
 * @param[in] gpu_int_work_space Integer array allocated in gpu \see{gpu_crs_symmetric_sparse_neti_s_get_size}
 * @param[in] gpu_double_work_space Double array allocated in gpu
 * @param[in] cpu_double_work_space Double array allocated in cpu
 * @param[in] work_space Pre-allocated work space (@)
 * @param[in] deltaTime The timestep
 * @param[in] krylov_length The dimension of the krylov space
 * @return oldPosition will contain the old currentPosition, while currentPosition will contain the new position
 */
int gpu_crs_symmetric_sparse_neti_d(int ndof,
									const sparse_d_matrix_t * invMassMatrix,
									const sparse_d_matrix_t * dampingMatrix,
									const sparse_d_matrix_t * stiffnessMatrix,
									const double * forceVector,
									double * oldPosition,
									double * currentPosition,
									magma_int_t * gpu_int_work_space,
									double * gpu_double_work_space,
									double * cpu_double_work_space,
									double deltaTime,
									int krylov_length
									);