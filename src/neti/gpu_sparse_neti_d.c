/* 
* @Author: Rafael Farias Marinheiro
* @Date:   2014-09-12 14:49:29
* @Last Modified by:   Rafael Farias Marinheiro
* @Last Modified time: 2014-09-12 15:01:59
*/

#include "../gpu_sparse_neti.h"

int gpu_crs_symmetric_sparse_neti_d_get_size(int ndof, int krylov_length,
											 int * gpu_int_work_space_size,
											 int * gpu_double_work_space_size,
											 int * cpu_double_work_space_size){
	int d = ndof;
	int m = krylov_length;
	int nb = magma_get_dgehrd_nb(m);

	*gpu_int_work_space_size = m;
	*gpu_double_work_space_size = d*m + m*m + m + d + d + d;
	*cpu_double_work_space_size = m*m + m*m + 2*m + (2 + 2*nb)*m;

	return 0;

}

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
									){
	int d = ndof;
	int m = krylov_length;
	//GPU double Data [Total: d*m + m*m + m + d + d + d]
	double * V_gpu 		= gpu_double_work_space;
	double * T_gpu 		= gpu_double_work_space + d*m;
	double * z_gpu 		= gpu_double_work_space + d*m + m*m;
	double * xc_gpu 		= gpu_double_work_space + d*m + m*m + m;
	double * xphi_gpu 	= gpu_double_work_space + d*m + m*m + m + d;
	double * v_gpu 		= gpu_double_work_space + d*m + m*m + m + d + d;

	magma_int_t * ipiv = gpu_int_work_space;

	//CPU Data [Total: m*m + m*m + 2*m + ? ]
	double * H_cpu = cpu_double_work_space;
	double * T_cpu = cpu_double_work_space + m*m;
	double * eigen_cpu = cpu_double_work_space + m*m + m*m;
	double * sgeev_work = cpu_double_work_space + m*m + m*m + 2*m;
	
	//Compute matrix function
	{
		double normv = magma_dnrm2(d, currentPosition, 1);
		if(normv == 0.0){
			//Filling V matrix with [b Av A^2b ... A^(m-1)b] where A = M^-1K
			{
				magma_dcopy(d, currentPosition, 1, V_gpu, 1);
				double * V_pointer = V_gpu;
				int i;
				for(i = 1; i < m; i++){
					magma_dgecsrmv(MagmaNoTrans, ndof, ndof, 1,
								   stiffnessMatrix->d_val, stiffnessMatrix->d_rowptr, stiffnessMatrix->d_colind,
								   V_pointer,
								   0,
								   V_pointer+d);
					magma_dgecsrmv(MagmaNoTrans, ndof, ndof, 1,
								   invMassMatrix->d_val, invMassMatrix->d_rowptr, invMassMatrix->d_colind,
								   V_pointer+d,
								   0,
								   V_pointer+d);
					V_pointer = V_pointer+d;
				}
			}

			//Performing Arnoldi Iteration
			{
				//Normalize q_1
				magma_dscal(d, 1/normv, V_gpu, 1);
				
				//Arnoldi
				int k;
				for(k = 1; k < m; k++){
					int j;
					for(j = 0; j < k; j++){
						H_cpu[j + (k-1)*m] = magma_ddot(d, V_gpu+d*j, 1, V_gpu+d*k, 1);
						magma_daxpy(d, -H_cpu[j + (k-1)*m], V_gpu+d*j, 1, V_gpu+d*k, 1);
					}

					H_cpu[k + (k-1)*m] = magma_dnrm2(d, V_gpu+d*k, 1);
					magma_dscal(d, 1/H_cpu[k + (k-1)*m], V_gpu+d*k, 1);
					
					for(j = k+1; j < m; j++){
						H_cpu[j + (k-1)*m] = 0;
					}
				}
			}

			{
				//Eigen decomposition
				int nb = magma_get_dgehrd_nb(m);

				int info;
				magma_dgeev	(MagmaNoVec, MagmaVec,
							 m, H_cpu, m,
							 eigen_cpu,
							 eigen_cpu + m,
							 NULL,
							 0,
							 T_cpu,
							 m,
							 sgeev_work,
							 (2 + 2*nb)*m,
							 &info);

				magma_dsetmatrix(m, m, T_cpu, m, T_gpu, m);

				int i;
				for(i = 0; i < m; i++){
					eigen_cpu[m + i] = cos(deltaTime*sqrt(eigen_cpu[i])) * T_cpu[i];
				}

				magma_dsetvector(m, eigen_cpu+m, 1, z_gpu, 1);
				magma_dgesv_gpu(m, 1, T_gpu, m, ipiv, z_gpu, 1);
			}

			{
				//Evaluate z
				magma_dgemv(MagmaNoTrans, d, m, normv, V_gpu, d, z_gpu, 1, 0, xc_gpu, 1);
			}
		} else{
			magma_dscal(d, 0, xc_gpu, 1);
		}
	}

	{
		//Evaluate equivalent force
		magma_dcopy(d, currentPosition, 1, v_gpu, 1);
		magma_daxpy(d, -1, oldPosition, 1, v_gpu, 1);
		magma_dscal(d, 1/deltaTime, v_gpu, 1);
		magma_dgecsrmv(MagmaNoTrans, ndof, ndof, 1,
					   dampingMatrix->d_val, dampingMatrix->d_rowptr, dampingMatrix->d_colind,
					   v_gpu,
					   0,
					   v_gpu);
		magma_daxpy(d,  1, forceVector, 1, v_gpu, 1);
	}

	//Compute matrix function
	{
		double normv = magma_dnrm2(d, v_gpu, 1);
		if(normv == 0){
			//Filling V matrix with [b Av A^2b ... A^(m-1)b] where A = M^-1K
			{
				magma_dcopy(d, v_gpu, 1, V_gpu, 1);
				double * V_pointer = V_gpu;
				int i;
				for(i = 1; i < m; i++){
					magma_dgecsrmv(MagmaNoTrans, ndof, ndof, 1,
								   stiffnessMatrix->d_val, stiffnessMatrix->d_rowptr, stiffnessMatrix->d_colind,
								   V_pointer,
								   0,
								   V_pointer+d);
					magma_dgecsrmv(MagmaNoTrans, ndof, ndof, 1,
								   invMassMatrix->d_val, invMassMatrix->d_rowptr, invMassMatrix->d_colind,
								   V_pointer+d,
								   0,
								   V_pointer+d);
					V_pointer = V_pointer+d;
				}
			}

			//Performing Arnoldi Iteration
			{
				//Normalize q_1
				magma_dscal(d, 1/normv, V_gpu, 1);
				
				//Arnoldi
				int k;
				for(k = 1; k < m; k++){
					int j;
					for(j = 0; j < k; j++){
						H_cpu[j + (k-1)*m] = magma_ddot(d, V_gpu+d*j, 1, V_gpu+d*k, 1);
						magma_daxpy(d, -H_cpu[j + (k-1)*m], V_gpu+d*j, 1, V_gpu+d*k, 1);
					}

					H_cpu[k + (k-1)*m] = magma_dnrm2(d, V_gpu+d*k, 1);
					magma_dscal(d, 1/H_cpu[k + (k-1)*m], V_gpu+d*k, 1);
					
					for(j = k+1; j < m; j++){
						H_cpu[j + (k-1)*m] = 0;
					}
				}
			}

			{
				//Eigen decomposition
				int nb = magma_get_dgehrd_nb(m);

				int info;
				magma_dgeev	(MagmaNoVec, MagmaVec,
							 m, H_cpu, m,
							 eigen_cpu,
							 eigen_cpu + m,
							 NULL,
							 0,
							 T_cpu,
							 m,
							 sgeev_work,
							 (2 + 2*nb)*m,
							 &info);

				magma_dsetmatrix(m, m, T_cpu, m, T_gpu, m);

				int i;
				for(i = 0; i < m; i++){
					eigen_cpu[m + i] = cos(deltaTime*sqrt(eigen_cpu[i])) * T_cpu[i];
				}

				magma_dsetvector(m, eigen_cpu+m, 1, z_gpu, 1);
				magma_dgesv_gpu(m, 1, T_gpu, m, ipiv, z_gpu, 1);
			}

			{
				//Evaluate z
				magma_dgemv(MagmaNoTrans, d, m, normv, V_gpu, d, z_gpu, 1, 0, xphi_gpu, 1);
			}
		} else{
			magma_dscal(d, 0, xphi_gpu, 1);
		}

		{
			//Compute the next position
			magma_dcopy(d, currentPosition, 1, v_gpu, 1);
			magma_dcopy(d, xc_gpu, 1, currentPosition,1);
			magma_dgecsrmv(MagmaNoTrans, ndof, ndof, -deltaTime*deltaTime,
						   invMassMatrix->d_val, invMassMatrix->d_rowptr, invMassMatrix->d_colind,
						   xphi_gpu,
						   2,
						   currentPosition);
			magma_daxpy(d, -1, oldPosition, 1, currentPosition, 1);
			magma_dcopy(d, v_gpu, 1, oldPosition, 1);
		}
	}
	return 0;
}