package LBFGS;

import utils.Utils;
import LBFGS.optimzationTest.Problem6;
import LBFGS.optimzationTest.QuadraticTest;

public class ProjectedGradient {
	
	//interface to the objective function to be minimized
	Optimizable m_func; 
	
	//maximum iteration of gradient descent
	int m_maxIter; 
	
	//convergence criterion
	double m_converge;
	
	//cache for the gradient
	double[] m_g;
	
	//cache for the parameters
	double[] m_x, m_x_old;//current point and next point
	
	//parameters for line search
	double m_alpha, m_beta, m_delta; 
	
	public ProjectedGradient(Optimizable objFunc, int maxIter, double converge, double beta, double delta) {
		m_func = objFunc;
		m_maxIter = maxIter;
		m_converge = converge;
		
		m_alpha = 1.0;
		m_beta = beta;
		m_delta = delta;
	}
	
	void init() {
		// the function is supposed to perform necessary initialization before calling optimization
		m_x = m_func.getParameters();//we will pass the reference so that the parameters will be overwritten
		m_x_old = new double[m_x.length];
		System.arraycopy(m_x, 0, m_x_old, 0, m_x.length);
		
		m_g = new double[m_x.length];
	}
	
	public boolean optimize() {
		init();

		int iter = 0;
		double gNorm, xNorm;//get the initial function value and gradient
		
		do {
			gradientDescent(m_func.calcFuncGradient(m_g));//get function value and gradient at m_x1
			
			gNorm = Utils.L2Norm(m_g);
			xNorm = Utils.L2Norm(m_x);
		} while (++iter<m_maxIter && gNorm>xNorm*m_converge);
		
		return gNorm<=xNorm*m_converge;//also need other convergence condition checking
	}
	
	double update() {
		for(int i=0; i<m_x.length; i++)
			m_x[i] = m_x_old[i] - m_alpha * m_g[i];
		m_func.projection(m_x);
		return m_func.calcFunc(m_x);
	}
	
	double gradientDescent(double initV) {
		m_alpha = 1.0 / m_beta;
		
		double value;
		do {
			m_alpha *= m_beta;
			value = update();
		} while (value-initV > suffDescent());//until alpha satisfies the condition
				
		System.arraycopy(m_x, 0, m_x_old, 0, m_x.length);//will directly affect the parameters in m_func
		return value;
	}
	
//	private double gradientDescent(double initV) {
//		double value = update();//check the current alpha
//		
//		if (value-initV <= suffDescent()) {
//			do {
//				initV = value;
//				m_alpha /= m_beta;//increase alpha
//				value = update();
//			} while (difference()>0 && value-initV <= suffDescent());//until alpha does not satisfy the condition
//		} else {
//			do {
//				initV = value;
//				m_alpha *= m_beta;//decrease alpha
//				value = update();
//			} while (value-initV > suffDescent());//until alpha satisfies the condition
//		}
//		
//		System.arraycopy(m_x2, 0, m_x1, 0, m_x1.length);//will directly affect the parameters in m_func
//		return value;
//	}
	
	double suffDescent() {
		double diff = 0;
		for(int i=0; i<m_x.length; i++)
			diff += m_g[i] * (m_x[i] - m_x_old[i]);
		return diff * m_delta;
	}
	
	private double difference() {
		double diff = 0;
		for(int i=0; i<m_x.length; i++)
			diff += (m_x_old[i] - m_x[i]) * (m_x_old[i] - m_x[i]);
		return diff;
	}
	
	static public void main(String[] args) {
		QuadraticTest testcase = new Problem6();
		ProjectedGradient opt = new ProjectedGradient(testcase, 100000, 1e-4, 0.35, 0.0001);
		
		double value = testcase.byLBFGS();
		double[] x = testcase.getParameters();
		System.out.format("By L-BFGS\n%.10f\t%.10f\t%.10f\n", x[0], x[1], value);
		
		testcase.reset();
		x = testcase.getParameters();
		System.out.println(opt.optimize()?"success!":"failed!");
		System.out.format("By Projected Gradient\n%.10f\t%.10f\t%.10f\n", x[0], x[1], testcase.calcFunc());
	}
}
