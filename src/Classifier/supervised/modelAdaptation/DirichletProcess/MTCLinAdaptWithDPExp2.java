package Classifier.supervised.modelAdaptation.DirichletProcess;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import structures._PerformanceStat;
import structures._Review;
import structures._thetaStar;
import structures._PerformanceStat.TestMode;
import structures._Review.rType;
import utils.Utils;

public class MTCLinAdaptWithDPExp2 extends MTCLinAdaptWithDP {
	public MTCLinAdaptWithDPExp2(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup);
	}
	PrintWriter m_writer;
	int m_count = 0;
	protected void calculateClusterProbPerUser(){
		try{
			m_writer = new PrintWriter(new File(String.format("compare_%d.txt", m_count++)));
			double prob, dist;
			_DPAdaptStruct user;
			double[] probs = new double[m_kBar];
			double[] probs_orc = new double[m_kBar];
			_thetaStar oldTheta;
		
			for(int i=0; i<m_userList.size(); i++){
				user = (_DPAdaptStruct) m_userList.get(i);
				
				oldTheta = user.getThetaStar();
				for(int k=0; k<m_kBar; k++){
					user.setThetaStar(m_thetaStars[k]);
				
					prob = calcLogLikelihood(user) + Math.log(m_thetaStars[k].getMemSize());
					probs[k] = Math.exp(prob);//this will be in real space!
				
					prob = calcLogLikelihood4Posterior(user) + Math.log(m_thetaStars[k].getMemSize());//this proportion includes the user's current cluster assignment
					probs_orc[k] = Math.exp(prob);
				}
				Utils.L1Normalization(probs);
				Utils.L1Normalization(probs_orc);
				dist = Utils.EuclideanDistance(probs, probs_orc);				
				System.out.print(String.format("%d\t%d\t%.4f\t[", user.getUser().getReviewSize(), user.getUser().getCtgSize(), dist));
				
				double adaPosRatio = user.getUser().getAdaptationPos()/user.getAdaptationSize();
				double testPosRatio = user.getUser().getTestPos()/user.getTestSize();
				System.out.print(String.format("adaPosRation: %.4f, testPosRatio: %.4f\n", adaPosRatio, testPosRatio));
				
				user.setClusterPosterior(probs);
				user.setClusterPosterior_orc(probs_orc);
				user.setThetaStar(oldTheta);//restore the cluster assignment during EM iterations
			}
			m_writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}
