package LBFGS.optimzationTest;

import java.util.Arrays;

import LBFGS.Optimizable;

public class Problem5 extends QuadraticTest implements Optimizable {
	double[] m_Y = new double[]{1.5, 2.25, 2.625};
	
	public Problem5() {
		m_x = new double[2];
	}
	
	@Override
	public double calcFuncGradient(double[] g) {
		m_neval ++;
		
		double f;
		Arrays.fill(m_g, 0);
		for(int i=0; i<m_Y.length; i++) {
			f = m_Y[i] - m_x[0] * (1-Math.pow(m_x[1], i+1));
			
			m_g[0] += 2 * f * (Math.pow(m_x[1], i+1) - 1);
			m_g[1] += 2 * f * m_x[0] * (i+1) * Math.pow(m_x[1], i);
		}
		return calcFunc(m_x);
	}

	@Override
	public double calcFunc(double[] x) {
		m_neval ++;
		
		double f, sum = 0;
		for(int i=0; i<m_Y.length; i++){
			f = m_Y[i] - x[0] * (1-Math.pow(x[1], i+1));
			sum += f*f;
		}
		return sum;
	}
	
	@Override
	public void projection(double[] x) {
		
	}
	
	@Override
	public void reset() {
		m_x[0] = 1;
		m_x[1] = 1;
		m_neval = 0;
	}
}
