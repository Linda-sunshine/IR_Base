package Classifier.supervised.modelAdaptation.DirichletProcess;

import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;

public class NormalPrior {
	protected Normal m_normal; // Normal distribution.
	double m_meanA, m_sdA;
	
	public NormalPrior (double mean, double sd) {
		m_meanA = mean;
		m_sdA = sd;
		m_normal = new Normal(mean, sd, new DoubleMersenneTwister());
	}
	
	public void sampling(double[] target) {
		for(int i=0; i<target.length; i++)
			target[i] = m_normal.nextDouble();
	}
	
	public double likelihood(double[] target) {
		double L = 0;
		for(int i=0; i<target.length; i++)
			L += (target[i]-m_meanA)*(target[i]-m_meanA)/m_sdA/m_sdA;
		return L/2;
	}
}
