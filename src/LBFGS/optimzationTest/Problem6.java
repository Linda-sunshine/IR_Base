package LBFGS.optimzationTest;

import java.util.Arrays;

import LBFGS.Optimizable;

public class Problem6 extends QuadraticTest implements Optimizable {

	public Problem6() {
		m_x = new double[2];
	}
	
	@Override
	public double calcFuncGradient(double[] g) {
		m_neval ++;
		
		double f;
		Arrays.fill(m_g, 0);
		for(int i=1; i<=10; i++){
			f = 2*(i+1) - Math.exp(i*m_x[0]) - Math.exp(i*m_x[1]);
			m_g[0] -= 2*f * i*Math.exp(i*m_x[0]);
			m_g[1] -= 2*f * i*Math.exp(i*m_x[1]);
		}
		return calcFunc(m_x);
	}

	@Override
	public double calcFunc(double[] x) {
		m_neval ++;
		
		double f, sum = 0;
		for(int i=1; i<=10; i++){
			f = 2*(i+1) - Math.exp(i*x[0]) - Math.exp(i*x[1]);
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
