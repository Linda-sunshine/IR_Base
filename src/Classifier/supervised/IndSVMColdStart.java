package Classifier.supervised;

import java.util.ArrayList;

import structures._PerformanceStat;
import structures._Review;
import structures._PerformanceStat.TestMode;
import structures._Review.rType;
import utils.Utils;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.modelAdaptation._AdaptStruct;

public class IndSVMColdStart extends IndividualSVM {
	int m_threshold = 1;
	
	public IndSVMColdStart(int classNo, int featureSize, int t) {
		super(classNo, featureSize);
		m_threshold = t;
	}

	@Override
	public double train() {
		init();
		
		//Transfer all user reviews to instances recognized by SVM, indexed by users.
		int trainSize = 0, validUserIndex = 0;
		ArrayList<Feature []> fvs = new ArrayList<Feature []>();
		ArrayList<Double> ys = new ArrayList<Double>();		
		_Review r;
		//Two for loop to access the reviews, indexed by users.
		ArrayList<_Review> reviews;
		for(_AdaptStruct user: m_userList){
			int count = 0;
			if(user.getTestSize() == 0)
				continue;// ignore the adaptation users.
			reviews = user.getReviews();		
			trainSize = 0;
			boolean validUser = false;
			while(count < m_threshold){
				r = reviews.get(count);
				fvs.add(createLibLinearFV(r, validUserIndex));
				ys.add(new Double(r.getYLabel()));
				trainSize ++;
				validUser = true;
				count++;
			}	
			
			if (validUser)
				validUserIndex ++;
			
			// Train individual model for each user.
			Problem libProblem = new Problem();
			libProblem.l = trainSize;		
			libProblem.x = new Feature[trainSize][];
			libProblem.y = new double[trainSize];
			for(int i=0; i<trainSize; i++) {
				libProblem.x[i] = fvs.get(i);
				libProblem.y[i] = ys.get(i);
			}
			if (m_bias) {
				libProblem.n = m_featureSize + 1; // including bias term; global model + user models
				libProblem.bias = 1;// bias term in liblinear.
			} else {
				libProblem.n = m_featureSize;
				libProblem.bias = -1;// no bias term in liblinear.
			}
			m_libModel = Linear.train(libProblem, new Parameter(m_solverType, m_C, SVM.EPS));
			setPersonalizedModel(user);
		}
		return 0;
	}
	@Override
	public double test(){
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		m_perf = new double[m_classNo * 2];
		m_microStat = new _PerformanceStat(m_classNo);

		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				public void run() {
					_AdaptStruct user;
					_PerformanceStat userPerfStat;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							int count = 0;
							user = m_userList.get(i+core);
							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
								continue;
							userPerfStat = user.getPerfStat();								
							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
								//record prediction results
								for(_Review r:user.getReviews()) {
									if (r.getType() != rType.TEST)
										continue;
									if(count++ < m_threshold)// For the review used for assignment
										continue;
									int trueL = r.getYLabel();
									int predL = user.predict(r); // evoke user's own model
									userPerfStat.addOnePredResult(predL, trueL);
								}
							}							
							userPerfStat.calculatePRF();	
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
		
		int count = 0;
		double[] macroF1 = new double[m_classNo];
		_PerformanceStat userPerfStat;
		
		for(_AdaptStruct user:m_userList) {
			if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
				|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
				|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
				continue;
			
			userPerfStat = user.getPerfStat();
			for(int i=0; i<m_classNo; i++)
				macroF1[i] += userPerfStat.getF1(i);
			m_microStat.accumulateConfusionMat(userPerfStat);
			count ++;
		}
		
		System.out.println(toString());
		calcMicroPerfStat();
		
		// macro average
		System.out.println("\nMacro F1:");
		for(int i=0; i<m_classNo; i++){
			System.out.format("Class %d\t%.4f\t", i, macroF1[i]/count);
			m_perf[i+m_classNo] = macroF1[i]/count;
		}
		System.out.println();
		return Utils.sumOfArray(macroF1);
	}
}
