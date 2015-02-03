package LBFGS;

import utils.Utils;
import LBFGS.linesearch.LineSearch;
import LBFGS.linesearch.LineSearchMoreThuente;
import LBFGS.optimzationTest.Problem1;
import LBFGS.optimzationTest.Problem2;
import LBFGS.optimzationTest.Problem3;
import LBFGS.optimzationTest.Problem5;
import LBFGS.optimzationTest.Problem6;
import LBFGS.optimzationTest.QuadraticTest;

public class ProjectedGradient {
	
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
	
	// search direction
	double[] m_sd; 
	
	//cache for the parameters
	double[] m_x, m_x_old;//current point and next point
	
	public ProjectedGradient(Optimizable objFunc, int maxIter, double fdelta, double gdelta, double istp, double ftol, double gtol) {
		m_func = objFunc;
		m_maxIter = maxIter;
		m_fdelta = fdelta;
		m_gdelta = gdelta;
		
		//m_linesearch = new LineSearchBacktracking(objFunc, istp, ftol, gtol, maxIter);
		m_linesearch = new LineSearchMoreThuente(objFunc, istp, ftol, gtol, 1e-5, maxIter);
	}
	
	void init() {
		// the function is supposed to perform necessary initialization before calling optimization
		m_x = m_func.getParameters();//we will pass the reference so that the parameters will be overwritten
		m_x_old = new double[m_x.length];
		System.arraycopy(m_x, 0, m_x_old, 0, m_x.length);
		
		m_g = new double[m_x.length];
		m_sd = new double[m_x.length];
	}
	
	void getSearchDirection() {
		//gradient as the search direction
		Utils.setArray(m_sd, m_g, -1);
	}
	
	public boolean optimize() {
		init();

		int k = 1; // the first step has been explore in init()
		double gNorm, xNorm, fx_old, converge;//get the initial function value and gradient
		
		//initial step for L-BFGS
		double fx = m_func.calcFuncGradient(m_g);		
		getSearchDirection();
		fx = m_linesearch.linesearch(fx, m_x, m_x_old, m_g, m_sd);
		
		do {
			fx_old = fx;
			
			getSearchDirection();
			fx = m_linesearch.linesearch(fx, m_x, m_x_old, m_g, m_sd);
			
			gNorm = Utils.L2Norm(m_g);
			xNorm = Utils.L2Norm(m_x);
			if (fx_old==0)
				converge = 1.0;//no way to compute
			else
				converge = (fx_old-fx)/fx_old;
			
			//System.out.format("%d. f(x)=%.10f |g(x)|=%.10f converge=%.5f\n", k, fx, gNorm, converge);
		} while (++k<m_maxIter && gNorm>xNorm*m_gdelta && Math.abs(converge)>m_fdelta);
		
		return k<m_maxIter;//also need other convergence condition checking
	}
	
	
	static public void main(String[] args) {
		QuadraticTest[] testcases = new QuadraticTest[]{new Problem1(), new Problem2(), new Problem3(), new Problem5(), new Problem6()};
		for(QuadraticTest testcase:testcases) {
			
			
			System.out.format("In %s:\n", testcase.toString());
			
			double value = testcase.byLBFGS();
			double[] x = testcase.getParameters();
			System.out.format("By L-BFGS\n%s\t%.10f\t%d\n", Utils.formatArray(x), value, testcase.getNumEval());
			testcase.reset();
			
			ProjectedGradient opt = new ProjectedGradient(testcase, 200, 1e-7, 1e-3, 1.0, 1e-5, 0.2);
			x = testcase.getParameters();
			opt.optimize();
			System.out.format("By Projected Gradient\n%s\t%.10f\t%d\n", Utils.formatArray(x), testcase.calcFunc(), testcase.getNumEval());
			testcase.reset();
			
			opt = new ProjectedLBFGS(testcase, 5, false, 1000, 1e-5, 1e-3, 1.0, 1e-8, 0.4);
			x = testcase.getParameters();
			opt.optimize();
			System.out.format("By Projected L-BFGS\n%s\t%.10f\t%d\n\n", Utils.formatArray(x), testcase.calcFunc(), testcase.getNumEval());
		}
	}
}
