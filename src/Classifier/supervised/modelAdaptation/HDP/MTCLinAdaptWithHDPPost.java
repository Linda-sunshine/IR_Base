package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import cern.jet.random.tdouble.Beta;

import structures._HDPThetaStar;
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

	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithHDPPost[dim:%d,supDim:%d,lmDim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),supScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]",
											m_dim,m_dimSup,m_lmDim,m_M,m_alpha,m_eta,m_beta,m_eta1,m_eta2,m_eta3,m_eta4,m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	public void setBetas(double[] lm){
		super.setBetas(lm);
		m_betaSum = Utils.sumOfArray(m_betas);
	}
	//Assign cluster to each review.
	protected void sampleOneInstance(_HDPAdaptStruct user, _Review r){
		super.sampleOneInstance(user, r);
		// We need to add the lm stat to the thetastar.
		r.getHDPThetaStar().addLMStat(r.getLMSparse());
	}
	
	// Current implementation, sample psi based on posterior.
	public void sampleNewCluster(int k, _SparseFeature[] fvs){
		
		m_hdpThetaStars[k].enable();
		m_hdpThetaStars[k].initLMStat(m_lmDim);
		
//		m_hdpThetaStars[k].initPsiModel(m_lmDim);
//		m_D0.sampling(m_hdpThetaStars[k].getPsiModel(), m_betas, fvs, true);//we should sample from Dir(\beta)
		
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
				L += logGammaDivision((int)fv.getValue(), m_betas[fv.getIndex()], Ns[fv.getIndex()]);
			}
			return L;
		}
	}
	
	// \Gamma(n_v+beta_v+N_v)/\Gamma(beta_v+N_v) = \prod_{i=1}^{n_v}(i+beta_v+N_v)
	// In log space, it is addition.
	protected double logGammaDivision(int n, double beta_v, double N_v){
		double res = 0;
		for(int i=1; i<=n; i++){
			res += Math.log(i+beta_v+N_v);
		}
		return res;
	}
	
	public void updateDocMembership(_HDPAdaptStruct user, _Review r){
		int index = -1;
		_HDPThetaStar curThetaStar = r.getHDPThetaStar();
		//Step 1: remove the current review from the thetaStar and user side.
		decUserHDPThetaStarMemSize(user, r);
		curThetaStar.updateMemCount(-1);
		curThetaStar.rmLMStat(r.getLMSparse());
		
		if(curThetaStar.getMemSize() == 0) {// No data associated with the cluster.
//			curThetaStar.resetPsiModel();
			// just for checkig purpose, to see if every dim gets 0 count.
			LMStatSanityCheck(curThetaStar);
			m_gamma_e += curThetaStar.getGamma();
			index = findHDPThetaStar(curThetaStar);
			swapTheta(m_kBar-1, index); // move it back to \theta*
			m_kBar --;
		}
	}
	public void LMStatSanityCheck(_HDPThetaStar theta){
		for(double c: theta.getLMStat()){
			if(c != 0){
				System.err.println("Non-zero count in lm stat!");
				return;
			}
		}
	}
	// Sample the weights given the cluster assignment.
	@Override
	protected double calculate_M_step(){
		assignClusterIndex();		
		
		//Step 1: sample gamma based on the current assignment.
		sampleGamma(); // why for loop BETA_K times?
		
		//Step 2: Optimize logistic regression parameters with lbfgs.
		return estPhi();
	}
}
