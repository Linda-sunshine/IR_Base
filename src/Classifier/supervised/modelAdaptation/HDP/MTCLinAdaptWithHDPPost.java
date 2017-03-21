package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import cern.jet.random.tdouble.Beta;

import structures._Review;
import structures._SparseFeature;
import utils.Utils;

public class MTCLinAdaptWithHDPPost extends MTCLinAdaptWithHDP {
	// Calculate the likelihood based on the posterior predictive distribution
	double m_betaSum = 0;
	public MTCLinAdaptWithHDPPost(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
	}

	public void setBetas(double[] lm){
		super.setBetas(lm);
		m_betaSum = Utils.sumOfArray(m_betas);
	}
	//Assign cluster to each review.
	protected void sampleOneInstance(_HDPAdaptStruct user, _Review r){
		double likelihood, logSum = 0, gamma_k;
		int k;
		
		//Step 1: reset thetaStars for the auxiliary thetaStars.
		sampleThetaStars();
		
		//Step 2: sample thetaStar based on the loglikelihood of p(z=k|\gamma,\eta)p(y|x,\phi)p(x|\psi)
		for(k=0; k<m_kBar+m_M; k++){

			r.setHDPThetaStar(m_hdpThetaStars[k]);
			
			//log likelihood of y, i.e., p(y|x,\phi)
			likelihood = calcLogLikelihoodY(r);
			
			//log likelihood of x, i.e., p(x|\psi)
			likelihood += calcLogLikelihoodX(r);
			
			//p(z=k|\gamma,\eta)
			gamma_k = m_hdpThetaStars[k].getGamma();
			likelihood += Math.log(calcGroupPopularity(user, k, gamma_k));
			
			m_hdpThetaStars[k].setProportion(likelihood);//this is in log space!
			
			if(k==0) 
				logSum = likelihood;
			else 
				logSum = Utils.logSum(logSum, likelihood);
//			System.out.print(String.format("gammak: %.5f\tlikehood: %.5f\tlogsum:%.5f\n", gamma_k, likelihood, logSum));
		}
		//Sample group k with likelihood.
		k = sampleInLogSpace(logSum);
//		System.out.print(String.format("------kBar:%d, k:%d-----\n", m_kBar, k));
		
		//Step 3: update the setting after sampling z_ij.
		m_hdpThetaStars[k].updateMemCount(1);//-->1
		r.setHDPThetaStar(m_hdpThetaStars[k]);//-->2
//		m_hdpThetaStars[k].addLMStat(r.getLMSparse());
		
		//Step 4: Update the user info with the newly sampled hdpThetaStar.
		incUserHDPThetaStarMemSize(user, r, k);
		
		if(k >= m_kBar)
			sampleNewCluster(k, r.getLMSparse());
		
		m_hdpThetaStars[k].addLMStat(r.getLMSparse());

	}
	
	// Current implementation, sample psi based on posterior.
	public void sampleNewCluster(int k, _SparseFeature[] fvs){
		m_hdpThetaStars[k].enable();
		m_hdpThetaStars[k].initPsiModel(m_lmDim);
		m_D0.sampling(m_hdpThetaStars[k].getPsiModel(), m_betas, fvs, true);//we should sample from Dir(\beta)
		m_hdpThetaStars[k].initLMStat(m_lmDim);
		
		double rnd = Beta.staticNextDouble(1, m_alpha);
		m_hdpThetaStars[k].setGamma(rnd*m_gamma_e);
		m_gamma_e = (1-rnd)*m_gamma_e;
			
		swapTheta(m_kBar, k);
		m_kBar++;
	}
	
	protected double calcLogLikelihoodX(_Review r){
		if(r.getHDPThetaStar().getLMStat() == null){
			return r.getL4NewCluster();
		}else {		
			double[] Ns = r.getHDPThetaStar().getLMStat();
			double N = Utils.sumOfArray(Ns);
			double n = r.getLMSum();
			_SparseFeature[] fvs = r.getLMSparse();
			double L = Utils.lgamma(m_betaSum+N) - Utils.lgamma(m_betaSum+N+n);
			for(_SparseFeature fv: fvs){
				L += Utils.lgamma(m_betas[fv.getIndex()]+Ns[fv.getIndex()]+fv.getValue());
				L -= Utils.lgamma(m_betas[fv.getIndex()]+Ns[fv.getIndex()]);
			}
			return L;
		}
	}
	
	// We need to erase the stat for the words count in one cluster.
	protected void calculate_E_step(){
		for(int k=0; k<m_hdpThetaStars.length; k++){
			m_hdpThetaStars[k].clearLMStat();
		}
		super.calculate_E_step();
	}
	// Sample the weights given the cluster assignment.
	@Override
	protected double calculate_M_step(){
		assignClusterIndex();		
		
		//Step 1: sample gamma based on the current assignment.
		sampleGamma(); // why for loop BETA_K times?
		
		//Step 2: Optimize language model parameters with MLE.
		//Step 3: Optimize logistic regression parameters with lbfgs.
		return estPhi();
	}
}
