package markovmodel;

import structures._Doc;
import utils.Utils;

public class LRFastRestrictedHMM extends FastRestrictedHMM {
	
	double[] m_omega; // feature weight for topic transition
	double[] m_epsilons; // topic transition for each sentence
	
	public LRFastRestrictedHMM (double[] omega) {
		super(-1);//no global epsilon
		m_omega = omega;
	}

	public double ForwardBackward(_Doc d, double[][] emission) {
		initEpsilons(d);
		
		return super.ForwardBackward(d, emission);
	}
	
	//all epsilons in real space!!
	void initEpsilons(_Doc d) {
		int stnSize = d.getSenetenceSize();
		m_epsilons = new double[stnSize];
		for(int t=1; t<stnSize; t++)
			m_epsilons[t] = Utils.logistic(d.m_sentence_features[t-1], m_omega);//first sentence does not have features
	}
	
	@Override
	double getEpsilon(int t) {
		return m_epsilons[t];
	}
	
	public void BackTrackBestPath(_Doc d, double[][] emission, int[] path) {
		initEpsilons(d);		
		super.BackTrackBestPath(d, emission, path);
	}
	
}
