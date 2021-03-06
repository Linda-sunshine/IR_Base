package mains;

import Jama.Matrix;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.random.tdouble.Normal;


public class MatrixEfficiencyTest {

	public static void main(String[] args) {
		//to test the efficiency of matrix operations in JAMA and colt
		
		int n = 14, size = 64, m = 50; //size of the input matrix 2^6 to 2^n, repeat the test m times
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra(1e-10);
//		LUDecompositionQuick luSolver = new LUDecompositionQuick(1e-10);
		
		for(int t=6; t<=n; t++) {
			double[][] matrix = new double[size][size];
			
			double timeInv = 0, qualityInv = 0;
			double timeLU = 0, qualityLU = 0, diff;
			double timeJAMA = 0, qualityJAMA = 0;
			
			for(int s=0; s<m; s++) {
				//construct the testing matrix
				for(int i=0; i<size; i++) {
					matrix[i][i] += 1;//diagonal dominate
					for(int j=0; j<size; j++) {
						matrix[i][j] += Normal.staticNextDouble(0, 1);
					}
				}
				
				//colt matrix inverse
				long start = System.currentTimeMillis();
				DoubleMatrix2D mat = new DenseDoubleMatrix2D(matrix);
				DoubleMatrix2D ins = alg.inverse(mat);
				double[][] res = alg.mult(mat, ins).toArray();
				timeInv += (System.currentTimeMillis() - start)/1000.0;
				
				//quality test				
				double norm = 0;
				for(int i=0; i<size; i++) {
					for(int j=0; j<size; j++) {
						if (i==j)
							norm += (res[i][i]-1) * (res[i][i]-1);
						else
							norm += res[i][j] * res[i][j];
					}
				}
				qualityInv += norm/size/size;
				
				
				//inverse by colt LU decomposition
//				SparseDoubleMatrix2D identity = new SparseDoubleMatrix2D(size,size);				
//				for(int k=0; k<size; k++)
//					identity.setQuick(k, k, 1);
//				
//				start = System.currentTimeMillis();
//				luSolver.decompose(mat);
//				luSolver.solve(identity);
//				timeLU += (System.currentTimeMillis() - start)/1000.0;
//				
//				//quality test
//				norm = 0;
//				for(int i=0; i<size; i++) {
//					for(int j=0; j<size; j++) {
//						diff = ins.getQuick(i, j) - identity.getQuick(i, j);
//						norm += diff * diff;
//					}
//				}
//				qualityLU += norm/size/size;
				
				
				//inverse by JAMA matrix inverse
				start = System.currentTimeMillis();
				Matrix jMat = new Matrix(matrix);				
				Matrix invsMtx = jMat.inverse();
				Matrix resMtx = jMat.times(invsMtx);
				timeJAMA += (System.currentTimeMillis() - start)/1000.0;
				
				//quality test				
				norm = 0;
				for(int i=0; i<size; i++) {
					for(int j=0; j<size; j++) {
						diff = resMtx.get(i, j);
						if (i==j)
							diff -= 1;
						
						norm += diff * diff;
					}
				}
				qualityJAMA += norm/size/size;
			}
			
			System.out.format("%dX%d: Inv-residual=%.10f, Inv-time=%.3fs, LU-residual=%.10f, LU-time=%.3fs, JAMA-residual=%.10f, JAMA-time=%.3fs\n", size, size, 
					qualityInv, timeInv/m, 
					qualityLU, timeLU/m,
					qualityJAMA, timeJAMA/m);
			size *= 2;//double the size
		}

	}

}
