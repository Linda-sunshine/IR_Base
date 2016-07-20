package Classifier.supervised.modelAdaptation.DirichletProcess;

import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;

public class NormalPrior {
	protected Normal m_normal; // Normal distribution.
	
	public NormalPrior (double mean, double sd) {
		m_normal = new Normal(mean, sd, new DoubleMersenneTwister());
	}
	
	public void sampling(double[] target) {
		for(int i=0; i<target.length; i++)
			target[i] = m_normal.nextDouble();
	}
}
