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
	
	// Write this as an independent function for overriding purpose.
	public void updateUserHDPThetaStarMemSize(_HDPAdaptStruct user, _Review r, int k){
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
	
	//Sample how many local groups inside user reviews.
	protected int sampleH(_HDPAdaptStruct user, _HDPThetaStar s){
		int n = user.getHDPThetaMemSize(s);
		if(n==1)
			return 1;//s(1,1)=1		

		double etaGammak = Math.log(m_eta) + Math.log(s.getGamma());
		//the number of local groups lies in the range [1, n];
		for(int h=1; h<=n; h++){
			double stir = stirling(n, h);
			m_cache[h-1] = h*etaGammak + Math.log(stir);
		}
			
		//h starts from 0, we want the number of tables here.	
		return Utils.sampleInLogArray(m_cache, n) + 1;
	}
		
}
