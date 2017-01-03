package Classifier.supervised.modelAdaptation.DirichletProcess;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;

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
		// TODO Auto-generated constructor stub
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
//				m_writer.write(String.format("%d\t%d\t%.4f\t[", user.getUser().getReviewSize(), user.getUser().getCtgSize(), dist));
				
				System.out.print(String.format("%d\t%d\t%.4f\t[", user.getUser().getReviewSize(), user.getUser().getCtgSize(), dist));
//				
				double adaPosRatio = user.getUser().getAdaptationPos()/user.getAdaptationSize();
				double testPosRatio = user.getUser().getTestPos()/user.getTestSize();
				System.out.print(String.format("adaPosRation: %.4f, testPosRatio: %.4f\n", adaPosRatio, testPosRatio));
//				for(int in: user.getUser().getCategory()){
//					m_writer.write(in+" ");
//					System.out.print(in+" ");
//				}
//				m_writer.write("]\n");
//				System.out.print("]\n");
				
				user.setClusterPosterior(probs);
				user.setClusterPosterior_orc(probs_orc);
				user.setThetaStar(oldTheta);//restore the cluster assignment during EM iterations
			}
			m_writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	@Override
	public double test(){
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		try{
		final PrintWriter writer = new PrintWriter(new File("perf_compare.txt"));
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				public void run() {
					_DPAdaptStruct user;
					_PerformanceStat userPerfStat;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							user = (_DPAdaptStruct) m_userList.get(i+core);
							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
								continue;
								
							userPerfStat = user.getPerfStat();
							_PerformanceStat orc = new _PerformanceStat(m_classNo);
							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
								//record prediction results
								for(_Review r:user.getReviews()) {
									if (r.getType() != rType.TEST)
										continue;
									int trueL = r.getYLabel();
									int predL = user.predict(r); // evoke user's own model
									int predL_orc = user.predict_orc(r);
									userPerfStat.addOnePredResult(predL, trueL);
									orc.addOnePredResult(predL_orc, trueL);
								}
							}							
							userPerfStat.calculatePRF();
							orc.calculatePRF();
							writer.write(String.format("%d\t%d\t%.4f\t%.4f\n", 
							user.getUser().getReviewSize(),user.getUser().getCtgSize(),
							(userPerfStat.getF1(0)+userPerfStat.getF1(1))/2,(orc.getF1(0)+orc.getF1(1))/2));
						}
					} catch(Exception ex) {
						ex.printStackTrace(); 
					}
				}
				
				private Thread initialize(int core, int numOfCores) {
					this.core = core;
					this.numOfCores = numOfCores;
					return this;
				}
			}).initialize(k, numberOfCores));
			
			threads.get(k).start();
		}
		
		for(int k=0;k<numberOfCores;++k){
			try {
				threads.get(k).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		}
		writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
		return 0;
	}
}
