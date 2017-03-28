package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import cern.jet.random.tdouble.Gamma;

import structures._HDPThetaStar;
import structures._Review;
import structures._WeightedCount;
import utils.Utils;

public class MTCLinAdaptWithHDPConfidenceE extends MTCLinAdaptWithHDP{

	public MTCLinAdaptWithHDPConfidenceE(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
		// TODO Auto-generated constructor stub
	}
	
	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithHDPConfidenceE[dim:%d,supDim:%d,lmDim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),supScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]",
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
	
	//Sample how many local groups inside user reviews.
	protected int sampleH(_HDPAdaptStruct user, _HDPThetaStar s){
		int n = user.getHDPThetaSumMemSize(s);
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
	
	//Sample the global mixture proportion, \gamma~Dir(m1, m2,..,\alpha)
	protected void sampleGamma(){

		for(int k=0; k<m_kBar; k++)
			m_hdpThetaStars[k].m_hSize = 0;
		
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);	
			// we use another map inside the user to record the weighted count for each theta.
			for(_HDPThetaStar s:user.getWeightedHDPThetas())
				s.m_hSize += sampleH(user, s);
		}		
		
		m_cache[m_kBar] = Gamma.staticNextDouble(m_alpha, 1);//for gamma_e
		
		double sum = m_cache[m_kBar];
		for(int k=0; k<m_kBar; k++){
			m_cache[k] = Gamma.staticNextDouble(m_hdpThetaStars[k].m_hSize+m_alpha, 1);
			sum += m_cache[k];
		}
		
		for(int k=0; k<m_kBar; k++) 
			m_hdpThetaStars[k].setGamma(m_cache[k]/sum);
		
		m_gamma_e = m_cache[m_kBar]/sum;//\gamma_e.
	}
	
	// For later overwritten methods.
	public double calcGroupPopularity(_HDPAdaptStruct user, int k, double gamma_k){
		return user.getHDPThetaWeightedSumMemSize(m_hdpThetaStars[k]) + m_eta*gamma_k;
	}
}
