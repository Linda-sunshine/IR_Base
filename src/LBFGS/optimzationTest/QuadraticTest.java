package LBFGS.optimzationTest;

import LBFGS.LBFGS;
import LBFGS.Optimizable;
import LBFGS.LBFGS.ExceptionWithIflag;

public abstract class QuadraticTest implements Optimizable {

	double[] m_x;
	double[] m_g, m_diag;
	int m_neval;
	
	@Override
	public double[] getParameters() {
		return m_x;
	}

	@Override
	public void setParameters(double[] x) {
		if (x.length != m_x.length)
			return;
		System.arraycopy(x, 0, m_x, 0, m_x.length);
	}
	
	@Override
	public int getNumParameters() {
		return m_x.length;
	}
	
	public double calcFunc() {
		return calcFunc(m_x);
	}

	abstract public void reset();
	
	void init() {
		reset();
		
		m_g = new double[m_x.length];
		m_diag = new double[m_x.length];
	}

	public double byLBFGS() {
		int[] iflag = {0}, iprint = { -1, 3 };
		double fValue = 0;
		int fSize = m_x.length;
		
		init();
		
		try{
			do {
				fValue = calcFuncGradient(m_g);
				LBFGS.lbfgs(fSize, 6, m_x, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-20, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e){
			e.printStackTrace();
		}
		return fValue;
	}

	@Override
	public void calcDiagnoal(double[] x) {
		m_neval ++;
	}

	@Override
	public int getNumEval() {
		return m_neval;
	}
}
