package Classifier.supervised.modelAdaptation.HDP;

import cc.mallet.types.Dirichlet;

// Dirichlet distribution.
public class DirichletPrior {
	Dirichlet m_dirichlet;// dirichlet distribution.
	public DirichletPrior(){
		m_dirichlet = new Dirichlet();
	}
	
	//Given concentraciton parameter alpha and target vector, sample the probs from dirichlet distribution.
	//Shall we use stick-breaking or sample directly from dirichlet distribution?
	public void sampling(double[] target, double alpha){
		
	}
}
