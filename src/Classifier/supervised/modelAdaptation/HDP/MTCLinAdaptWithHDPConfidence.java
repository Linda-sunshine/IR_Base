package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import structures._Review;
import structures._WeightedCount;

public class MTCLinAdaptWithHDPConfidence extends MTCLinAdaptWithHDP{

	public MTCLinAdaptWithHDPConfidence(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
		// TODO Auto-generated constructor stub
	}
	
	// Write this as an independent function for overriding purpose.
	public void updateUserHDPThetaStarMemSize(_HDPAdaptStruct user, _Review r, int k){
		double confidence = calcLogLikelihoodY(r);
		confidence = Math.exp(confidence);
		confidence = 1 - confidence * (1 - confidence);
		user.incHDPThetaStarMemSize(m_hdpThetaStars[k], new _WeightedCount(confidence, 1));//-->3	
	}	
}
