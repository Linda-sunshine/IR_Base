package Classifier.supervised.modelAdaptation.DirichletProcess;

import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;

public class DoubleNormalPrior extends NormalPrior {
	protected Normal m_2ndNormal; // Normal distribution for second half of target random variable
	
	public DoubleNormalPrior(double mean1, double sd1, double mean2, double sd2) {
		super(mean1, sd2);

		m_2ndNormal = new Normal(mean2, sd2, new DoubleMersenneTwister());
	}

	@Override
	public void sampling(double[] target) {
		int i = 0;
		for(; i<target.length/2; i++)
			target[i] = m_normal.nextDouble();//first half for scaling
		
		for(; i<target.length; i++)
			target[i] = m_2ndNormal.nextDouble();//second half for shifting
	}
}
