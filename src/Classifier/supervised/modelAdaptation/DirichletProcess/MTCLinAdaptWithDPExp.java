package Classifier.supervised.modelAdaptation.DirichletProcess;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;

import structures.MyPriorityQueue;
import structures._Doc;
import structures._HDPThetaStar;
import structures._PerformanceStat;
import structures._RankItem;
import structures._Review;
import structures._thetaStar;
import structures._PerformanceStat.TestMode;
import structures._Review.rType;
import utils.Utils;
import Classifier.supervised.SVM;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.HDP._HDPAdaptStruct;

public class MTCLinAdaptWithDPExp extends MTCLinAdaptWithDP {
	double[] m_trainPerf;
	ArrayList<double[]> m_perfs = new ArrayList<double[]>();
	ArrayList<double[]> m_trainPerfs = new ArrayList<double[]>();
	
	public MTCLinAdaptWithDPExp(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup);
		// TODO Auto-generated constructor stub
	}
	
	public ArrayList<ArrayList<_Review>> collectClusterRvws(){
		HashMap<Integer, ArrayList<_Review>> clusters = new HashMap<Integer, ArrayList<_Review>>();
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct)m_userList.get(i);
			int index = user.getThetaStar().getIndex();
			if(!clusters.containsKey(index))
				clusters.put(index, new ArrayList<_Review>());
			
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.ADAPTATION)
					continue; // only touch the adaptation data
				clusters.get(index).add(r);
			}
		}
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(clusters.size());
		for(int index: clusters.keySet()){
			// sort the clusters of reviews in descending order.
			queue.add(new _RankItem(index, clusters.get(index).size()));
		}
		ArrayList<ArrayList<_Review>> sortedClusters = new ArrayList<ArrayList<_Review>>();
		for(_RankItem it: queue){
			sortedClusters.add(clusters.get(it.m_index));
		}
		
		System.out.println(String.format("Collect %d clusters of reviews for sanity check.\n", clusters.size()));
		return sortedClusters;
	}
	
	SVM m_svm;// added by Lin, svm for cross validation.
	public void CrossValidation(int kfold, int threshold){
		ArrayList<ArrayList<_Review>> sortedClusters = collectClusterRvws();
		ArrayList<double[]> prfs = new ArrayList<double[]>();
		ArrayList<Integer> sizes = new ArrayList<Integer>();
		
		// Initialize the svm for training purpose.
		m_svm = new SVM(m_classNo, m_featureSize, 1);
		for(ArrayList<_Review> cluster: sortedClusters){
			if(cluster.size() > threshold){
				sizes.add(cluster.size());
				prfs.add(CV4OneCluster(cluster, kfold));
			}
		}
		System.out.print("Size\tNeg:Precision\tRecall\t\tF1\t\tPos:Precision\tRecall\t\tF1\n");
		for(int i=0; i<prfs.size(); i++){
			double[] prf = prfs.get(i);
			System.out.print(String.format("%d\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\n", 
					sizes.get(i), prf[0], prf[6], prf[1], prf[7], prf[2], prf[8],
					prf[3], prf[9], prf[4], prf[10], prf[5], prf[11]));
		}
		System.out.println(sortedClusters.size() + " clusters in total!");

	} 
	
	public double[] CV4OneCluster(ArrayList<_Review> reviews, int kfold){
		Random r = new Random();
		int[] masks = new int[reviews.size()];
		// Assign the review fold index first.
		for(int i=0; i<reviews.size(); i++){
			masks[i] = r.nextInt(kfold);
		}
		ArrayList<_Doc> trainSet = new ArrayList<_Doc>();
		ArrayList<_Doc> testSet = new ArrayList<_Doc>();
		double[][] prfs = new double[kfold][6];
		double[] AvgVar = new double[12];
		for(int k=0; k<kfold; k++){
			trainSet.clear();
			testSet.clear();
			for(int j=0; j<reviews.size(); j++){
				if(masks[j] == k)
					testSet.add(reviews.get(j));
				else
					trainSet.add(reviews.get(j));
			}
			m_svm.train(trainSet);
			prfs[k] = test(testSet);
			// sum over all the folds 
			for(int j=0; j<6; j++)
				AvgVar[j] += prfs[k][j];
		}
		// prfs[k]: avg. calculate the average performance among different folds.
		for(int j=0; j<6; j++)
			AvgVar[j] /= kfold;
		// prfs[k+1]: var. calculate the variance among different folds.
		for(int j=0; j<6; j++){
			for(int k=0; k<kfold; k++){
				AvgVar[j+6] += (prfs[k][j] - AvgVar[j]) * (prfs[k][j] - AvgVar[j]);
			}
			AvgVar[j+6] = Math.sqrt(AvgVar[j+6]/kfold);
		}
		return AvgVar;
	}
	
	public double[] test(ArrayList<_Doc> testSet){
		double[][] TPTable = new double[m_classNo][m_classNo];
		for(_Doc doc: testSet){
			int pred = m_svm.predict(doc), ans = doc.getYLabel();
			TPTable[pred][ans] += 1; //Compare the predicted label and original label, construct the TPTable.
		}
		
		double[] prf = new double[6];
		for (int i = 0; i < m_classNo; i++) {
			prf[3*i] = (double) TPTable[i][i] / (Utils.sumOfRow(TPTable, i) + 0.00001);// Precision of the class.
			prf[3*i + 1] = (double) TPTable[i][i] / (Utils.sumOfColumn(TPTable, i) + 0.00001);// Recall of the class.
			prf[3*i + 2] = 2 * prf[3 * i] * prf[3 * i + 1] / (prf[3 * i] + prf[3 * i + 1] + 0.00001);
		}
		return prf;
	}
	
	@Override
	public double train(){
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		int count = 0;
		
		init(); // clear user performance and init cluster assignment	
//		calculate_M_step();
		
		// Burn in period.
		while(count++ < m_burnIn){
			calculate_E_step();
			lastLikelihood = calculate_M_step();
		}
		
		// EM iteration.
		for(int i=0; i<m_numberOfIterations; i++){
			// Cluster assignment, thinning to reduce auto-correlation.
			calculate_E_step();
			
			// Optimize the parameters
			curLikelihood = calculate_M_step();

			delta = (lastLikelihood - curLikelihood)/curLikelihood;
			
			if (i%m_thinning==0) {
				evaluateModel();
				test();
				testTrain();
			}
			
//			printInfo(i%5==0);//no need to print out the details very often
			System.out.print(String.format("\n[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
			if(Math.abs(delta) < m_converge)
				break;
			lastLikelihood = curLikelihood;
		}

		evaluateModel(); // we do not want to miss the last sample?!
//		setPersonalizedModel();
		return curLikelihood;
	}
	// In this part, we will also check each adaptation review's prediction.
	protected void evaluateModel() {//this should be only used in batch testing!
		for(int i=0; i<m_featureSize+1; i++)
			m_supWeights[i] = getSupWeights(i);
		
		System.out.println("[Info]Accumulating evaluation results during sampling...");

		//calculate cluster posterior p(c|u)
		calculateClusterProbPerUser();
		
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();		
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				public void run() {
					_DPAdaptStruct user;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							user = (_DPAdaptStruct)m_userList.get(i+core);
							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
								continue;
								
							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
								//record prediction results
								for(_Review r:user.getReviews()) {
									if (r.getType() != rType.TEST)
										user.evaluateTrain(r);
									else
										user.evaluate(r); // evoke user's own model
								}
							}							
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
	}
	
	@Override
	public double test(){
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		// Init all users in user list.
		for(int i=0; i<m_userList.size(); i++){
			m_userList.get(i).getPerfStat().clear();
		}
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				public void run() {
					_AdaptStruct user;
					_PerformanceStat userPerfStat;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
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
		m_microStat.clear();
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
		m_perf = new double[2];
		System.out.println("\nMacro F1:");
		for(int i=0; i<m_classNo; i++){
			System.out.format("Class %d: %.4f\t", i, macroF1[i]/count);
			m_perf[i] = macroF1[i]/count;
		}
		m_perfs.add(m_perf);
		System.out.println("\n");
		return Utils.sumOfArray(macroF1);
	}
	// test the performance of the traing documents.
	public void testTrain(){
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		// Init all users in user list.
		for(int i=0; i<m_userList.size(); i++){
			m_userList.get(i).getPerfStat().clear();
		}
		
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
							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
								//record prediction results
								for(_Review r:user.getReviews()) {
									if (r.getType() == rType.ADAPTATION){
										int trueL = r.getYLabel();
										int predL = user.predictTrain(r);
										userPerfStat.addOnePredResult(predL, trueL);
									}
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
		m_microStat.clear();
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
		m_trainPerf = new double[2];
		System.out.println("\nMacro F1:");
		for(int i=0; i<m_classNo; i++){
			System.out.format("Class %d: %.4f\t", i, macroF1[i]/count);
			m_trainPerf[i] = macroF1[i]/count;
		}
		m_trainPerfs.add(m_trainPerf);
		System.out.println("\n");
	}
	
	public void printPerfs(){
		System.out.println("Test documents performance:");
		for(double[] perf: m_perfs){
			System.out.print(String.format("%.4f\t%.4f\n", perf[0], perf[1]));
		}
		System.out.println("Train documents performance:");
		for(double[] perf: m_trainPerfs){
			System.out.print(String.format("%.4f\t%.4f\n", perf[0], perf[1]));
		}
	}
	// After we finish estimating the clusters, we calculate the probability of each user belongs to each cluster.
	protected void calculateClusterProbPerUser(){
		double prob;
		_DPAdaptStruct user;
		double[] probs = new double[m_kBar];
		_thetaStar oldTheta;

		constructClusterMap();
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
				
			oldTheta = user.getThetaStar();
			for(int k=0; k<m_kBar; k++){
				user.setThetaStar(m_thetaStars[k]);

				prob = calcDistance(user) + calcLogLikelihood(user) + Math.log(m_thetaStars[k].getMemSize());//this proportion includes the user's current cluster assignment
//				prob = calcLogLikelihood4Posterior(user) + Math.log(m_thetaStars[k].getMemSize());//this proportion includes the user's current cluster assignment

				probs[k] = Math.exp(prob);//this will be in real space!
			}
			Utils.L1Normalization(probs);
			user.setClusterPosterior(probs);

			user.setThetaStar(oldTheta);//restore the cluster assignment during EM iterations
		}
	}
	HashMap<_thetaStar, ArrayList<_DPAdaptStruct>> m_thetaUserMap = new HashMap<_thetaStar, ArrayList<_DPAdaptStruct>>();
	public void constructClusterMap(){
		for(int i=0; i<m_kBar; i++){
			m_thetaUserMap.put(m_thetaStars[i], new ArrayList<_DPAdaptStruct>());
		}
		for(_AdaptStruct u: m_userList){
			_DPAdaptStruct user = (_DPAdaptStruct) u;
			m_thetaUserMap.get(user.getThetaStar()).add(user);
		}
	}
	public double calcDistance(_AdaptStruct user){
		return 0;
	}
}
