package Classifier.supervised.modelAdaptation.DirichletProcess;

import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;

import cern.jet.random.tfloat.FloatUniform;

import structures._Review;
import structures._thetaStar;
import structures._Review.rType;
import utils.Utils;

public class IsoMTCLinAdaptWithDP extends MTCLinAdaptWithDP {
	int m_testSize;
	public IsoMTCLinAdaptWithDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap, featureGroup4Sup);
	}

//	@Override
//	protected void init(){
//		m_userSize = 0;//need to get the total number of valid users to construct feature vector for MT-SVM
//		m_testSize = 0;
//		for(_AdaptStruct user:m_userList){			
//			if (user.getAdaptationSize()>0) 				
//				m_userSize ++;	
//			else{
//				m_testSize ++;
//			}
//			user.getPerfStat().clear(); // clear accumulate performance statistics
//		}
//	}
	
	// The main MCMC algorithm, assign each user to clusters.
	protected void calculate_E_step() {
		_thetaStar curThetaStar;
		_DPAdaptStruct user;

		for (int i = 0; i < m_userList.size(); i++) {
			user = (_DPAdaptStruct) m_userList.get(i);
			if(user.getAdaptationSize() == 0)
				continue;
			curThetaStar = user.getThetaStar();
			curThetaStar.updateMemCount(-1);

			if (curThetaStar.getMemSize() == 0) {// No data associated with the
													// cluster.
				swapTheta(m_kBar - 1, findThetaStar(curThetaStar)); 
				m_kBar--;
			}
			sampleOneInstance(user);
		}
	}
	
	protected void assignCluster(){
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			if(user.getAdaptationSize() == 0)
				assignOneCluster(user);
		}
	}
	protected void assignOneCluster(_DPAdaptStruct u){
		_DPAdaptStruct user = u;
		double likelihood, logSum = 0;
		int k;
		for (k = 0; k < m_kBar; k++) {
			user.setThetaStar(m_thetaStars[k]);
			likelihood = calcLogLikelihood(user, true);
			
			likelihood += Math.log(m_thetaStars[k].getMemSize());
			m_thetaStars[k].setProportion(likelihood);// this is in log space!
		
			if (k == 0)
				logSum = likelihood;
			else
				logSum = Utils.logSum(logSum, likelihood);
		}

		logSum += Math.log(FloatUniform.staticNextFloat());

		k = 0;
		double newLogSum = m_thetaStars[0].getProportion();
		do {
			if (newLogSum >= logSum)
				break;
			k++;
			newLogSum = Utils.logSum(newLogSum, m_thetaStars[k].getProportion());
		} while (k < m_kBar);

		user.setThetaStar(m_thetaStars[k]);
	}
	
	protected double calcLogLikelihood(_AdaptStruct user, boolean test){
		double L = 0; //log likelihood.
		double Pi = 0;
		
		_Review review = user.getReviews().get(0);
		
		Pi = logit(review.getSparse(), user);
		if(review.getYLabel() == 1) {
			if (Pi>0.0)
				L += Math.log(Pi);					
			else
				L -= Utils.MAX_VALUE;
		} else {
			if (Pi<1.0)
				L += Math.log(1 - Pi);					
			else
				L -= Utils.MAX_VALUE;
		}
		return L;
	}
	// After we finish estimating the clusters, we calculate the probability of each user belongs to each cluster.
	@Override
	protected void calculateClusterProbPerUser(){
		double prob;
		_DPAdaptStruct user;
		double[] probs = new double[m_kBar];
		_thetaStar oldTheta;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			if(user.getTestSize() == 0)
				continue;
			oldTheta = user.getThetaStar();
			for(int k=0; k<m_kBar; k++){
				user.setThetaStar(m_thetaStars[k]);
				prob = calcLogLikelihood(user, true);
				prob += Math.log(m_thetaStars[k].getMemSize());//this proportion includes the user's current cluster assignment
				probs[k] = Math.exp(prob);//this will be in real space!
			}
			Utils.L1Normalization(probs);
			user.setClusterPosterior(probs);
			user.setThetaStar(oldTheta);//restore the cluster assignment during EM iterations
		}
	}
	// Sample one instance's cluster assignment.
	protected void sampleOneInstance(_DPAdaptStruct user){
		if(user.getAdaptationSize() != 0){
			super.sampleOneInstance(user);
		}	
	}
}
