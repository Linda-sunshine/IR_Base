package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import structures._HDPThetaStar;
import structures._Review;
import structures._WeightedCount;
import utils.Utils;

public class MTCLinAdaptWithHDPConfidence extends MTCLinAdaptWithHDP{

	public MTCLinAdaptWithHDPConfidence(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
		// TODO Auto-generated constructor stub
	}
	
	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithHDPConfidence[dim:%d,supDim:%d,lmDim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),supScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]",
											m_dim,m_dimSup,m_lmDim,m_M,m_alpha,m_eta,m_beta,m_eta1,m_eta2,m_eta3,m_eta4,m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	// Write this as an independent function for overriding purpose.
	public void incUserHDPThetaStarMemSize(_HDPAdaptStruct user, _Review r, int k){
		double confidence = calcLogLikelihoodY(r);
		confidence = Math.exp(confidence);
		confidence = 1 - confidence * (1 - confidence);
		_WeightedCount wc = new _WeightedCount(confidence, 1);
		r.setWeightedCount(wc);
		user.incHDPThetaStarMemSize(m_hdpThetaStars[k], wc);//-->3	
	}	

	public void decUserHDPThetaStarMemSize(_HDPAdaptStruct user, _Review r){
		if(r.getWeightedCount() != null)
			user.decHDPThetaStarMemSize(r.getHDPThetaStar(), r.getWeightedCount());				
	}	
}
