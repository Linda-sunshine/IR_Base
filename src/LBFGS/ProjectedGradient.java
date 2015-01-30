package LBFGS;

import LBFGS.linesearch.LineSearch;
import LBFGS.linesearch.LineSearchMoreThuente;

abstract public class ProjectedGradient {
	
	//interface to the objective function to be minimized
	Optimizable m_func; 
	
	//line search algorithm
	LineSearch m_linesearch;
	
	//maximum iteration of gradient descent
	int m_maxIter; 
	
	//convergence criterion
	double m_fdelta, m_gdelta;
	
	//cache for the gradient
	double[] m_g;
	
	//cache for the parameters
	double[] m_x, m_x_old;//current point and next point
	
	public ProjectedGradient(Optimizable objFunc, int maxIter, double fdelta, double gdelta, double istp, double ftol, double gtol) {
		m_func = objFunc;
		m_maxIter = maxIter;
		m_fdelta = fdelta;
		m_gdelta = gdelta;
		
		//m_linesearch = new LineSearchBacktracking(objFunc, istp, ftol, gtol, maxIter);
		m_linesearch = new LineSearchMoreThuente(objFunc, istp, ftol, gtol, 1e-3, maxIter);
	}
	
	void init() {
		// the function is supposed to perform necessary initialization before calling optimization
		m_x = m_func.getParameters();//we will pass the reference so that the parameters will be overwritten
		m_x_old = new double[m_x.length];
		System.arraycopy(m_x, 0, m_x_old, 0, m_x.length);
		
		m_g = new double[m_x.length];
	}
	
	abstract public boolean optimize();
}
