package Classifier.supervised.modelAdaptation.HDP;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Random;

import structures.MyPriorityQueue;
import structures._Doc;
import structures._HDPThetaStar;
import structures._PerformanceStat;
import structures._SparseFeature;
import structures._PerformanceStat.TestMode;
import structures._RankItem;
import structures._Review;
import structures._Review.rType;
import utils.Utils;
import Classifier.supervised.SVM;
import Classifier.supervised.modelAdaptation._AdaptStruct;

public class MTCLinAdaptWithHDPExp extends MTCLinAdaptWithHDP {
	double[] m_trainPerf;
	ArrayList<double[]> m_perfs = new ArrayList<double[]>();
	ArrayList<double[]> m_trainPerfs = new ArrayList<double[]>();

	boolean m_postCheck = false;
	public MTCLinAdaptWithHDPExp(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
	}
	public MTCLinAdaptWithHDPExp(int classNo, int featureSize, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
	}
	
	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithHDPExp[dim:%d,supDim:%d,lmDim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),supScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]",
											m_dim,m_dimSup,m_lmDim,m_M,m_alpha,m_eta,m_beta,m_eta1,m_eta2,m_eta3,m_eta4,m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	public void setPosteriorSanityCheck(boolean b){
		m_postCheck = b;
	}
	
	public ArrayList<ArrayList<_Review>> collectClusterRvws(){
		HashMap<Integer, ArrayList<_Review>> clusters = new HashMap<Integer, ArrayList<_Review>>();
		_HDPThetaStar theta = null;
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct)m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.ADAPTATION)
					continue; // only touch the adaptation data
				else{
					theta = r.getHDPThetaStar();
					if(clusters.containsKey(theta.getIndex()))
						clusters.get(theta.getIndex()).add(r);
					else 
						clusters.put(theta.getIndex(), new ArrayList<_Review>(Arrays.asList(r)));
				}
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
		for(int j=0; j<6; j++){
			AvgVar[j] /= (kfold*1.0);
			if(Double.isNaN(AvgVar[j]))
				System.out.println("NaN here!!");
		}

		// prfs[k+1]: var. calculate the variance among different folds.
		for(int j=0; j<6; j++){
			for(int k=0; k<kfold; k++){
				AvgVar[j+6] += (prfs[k][j] - AvgVar[j]) * (prfs[k][j] - AvgVar[j]);
			}
			AvgVar[j+6] = Math.sqrt(AvgVar[j+6]/(kfold*1.0));
			if(Double.isNaN(AvgVar[j+6]))
				System.out.println("NaN here!!");
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
	
//	@Override
//	// After we finish estimating the clusters, we calculate the probability of each testing review belongs to each cluster.
//	// Indeed, it is for per review, for inheritance we don't change the function name.
//	protected void calculateClusterProbPerUser(){
//		double prob, logSum;
//		double[] probs;
//		if(m_newCluster) 
//			probs = new double[m_kBar+1];
//		else 
//			probs = new double[m_kBar];
//		
//		_HDPAdaptStruct user;
//		_HDPThetaStar curTheta;
//		
//		//sample a new cluster parameter first.
//		if(m_newCluster) {
//			m_hdpThetaStars[m_kBar].setGamma(m_gamma_e);//to make it consistent since we will only use one auxiliary variable
//			m_G0.sampling(m_hdpThetaStars[m_kBar].getModel());
//		}
//
//		for(int i=0; i<m_userList.size(); i++){
//			user = (_HDPAdaptStruct) m_userList.get(i);
//			for(_Review r: user.getReviews()){
//				if (r.getType() != rType.TEST)
//					continue;				
//				
//				for(int k=0; k<probs.length; k++){
//					curTheta = m_hdpThetaStars[k];
//					r.setHDPThetaStar(curTheta);
//					if(m_postCheck)
//						prob = calcLogLikelihoodX(r) + calcLogLikelihoodY(r) + Math.log(user.getHDPThetaMemSize(curTheta) + m_eta*curTheta.getGamma());//this proportion includes the user's current cluster assignment
//					else
//						prob = calcLogLikelihoodX(r) + Math.log(curTheta.getMemSize() + m_eta*curTheta.getGamma());//this proportion includes the user's current cluster assignment
//					probs[k] = prob;
//				}
////				r.setHDPThetaStar(m_hdpThetaStars[Utils.maxOfArrayIndex(probs)]);
//				logSum = Utils.logSumOfExponentials(probs);
//				for(int k=0; k<probs.length; k++)
//					probs[k] -= logSum;
//				r.setClusterPosterior(probs);//posterior in log space
//			}
//		}
//	}
	
	@Override
	// After we finish estimating the clusters, we calculate the probability of each testing review belongs to each cluster.
	// Indeed, it is for per review, for inheritance we don't change the function name.
	protected void calculateClusterProbPerUser(){
		double prob, logSum;
		double[] probs;
		if(m_newCluster) 
			probs = new double[m_kBar+1];
		else 
			probs = new double[m_kBar];
		
		_HDPAdaptStruct user;
		_HDPThetaStar curTheta;
		
		//sample a new cluster parameter first.
		if(m_newCluster) {
			m_hdpThetaStars[m_kBar].setGamma(m_gamma_e);//to make it consistent since we will only use one auxiliary variable
			m_G0.sampling(m_hdpThetaStars[m_kBar].getModel());
		}
		int rvwSize = 0;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			rvwSize = user.getReviews().size();
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.TEST)
				continue;				
				
				for(int k=0; k<probs.length; k++){
					curTheta = m_hdpThetaStars[k];
					r.setHDPThetaStar(curTheta);
					prob = rvwSize <= 15 ? calcLogLikelihoodX(r) + Math.log(curTheta.getMemSize() + m_eta*curTheta.getGamma()):
						calcLogLikelihoodX(r) + Math.log(user.getHDPThetaMemSize(curTheta) + m_eta*curTheta.getGamma());
					probs[k] = prob;
				}
				logSum = Utils.logSumOfExponentials(probs);
				for(int k=0; k<probs.length; k++)
					probs[k] -= logSum;
				r.setClusterPosterior(probs);//posterior in log space
			}
		}
	}
//	
//	@Override
//	public double train(){
//		System.out.println(toString());
//		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
//		int count = 0;
//		
//		init(); // clear user performance and init cluster assignment	
////		calculate_M_step();
//		
//		// Burn in period.
//		while(count++ < m_burnIn){
//			calculate_E_step();
//			lastLikelihood = calculate_M_step();
//		}
//		
//		// EM iteration.
//		for(int i=0; i<m_numberOfIterations; i++){
//			// Cluster assignment, thinning to reduce auto-correlation.
//			calculate_E_step();
//			
//			// Optimize the parameters
//			curLikelihood = calculate_M_step();
//
//			delta = (lastLikelihood - curLikelihood)/curLikelihood;
//			
//			if (i%m_thinning==0) {
//				evaluateModel();
//				test();
//				testTrain();
//			}
//			
////			printInfo(i%5==0);//no need to print out the details very often
//			System.out.print(String.format("\n[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
//			if(Math.abs(delta) < m_converge)
//				break;
//			lastLikelihood = curLikelihood;
//		}
//
//		evaluateModel(); // we do not want to miss the last sample?!
////		setPersonalizedModel();
//		return curLikelihood;
//	}
//	
//	// In this part, we will also check each adaptation review's prediction.
//	protected void evaluateModel() {//this should be only used in batch testing!
//		for(int i=0; i<m_featureSize+1; i++)
//			m_supWeights[i] = getSupWeights(i);
//		
//		System.out.println("[Info]Accumulating evaluation results during sampling...");
//
//		//calculate cluster posterior p(c|u)
//		calculateClusterProbPerUser();
//		
//		int numberOfCores = Runtime.getRuntime().availableProcessors();
//		ArrayList<Thread> threads = new ArrayList<Thread>();		
//		
//		for(int k=0; k<numberOfCores; ++k){
//			threads.add((new Thread() {
//				int core, numOfCores;
//				public void run() {
//					_HDPAdaptStruct user;
//					try {
//						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
//							user = (_HDPAdaptStruct)m_userList.get(i+core);
//							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
//								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
//								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
//								continue;
//								
//							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
//								//record prediction results
//								for(_Review r:user.getReviews()) {
//									if (r.getType() != rType.TEST)
//										user.evaluateTrain(r);
//									else
//										user.evaluate(r); // evoke user's own model
//								}
//							}							
//						}
//					} catch(Exception ex) {
//						ex.printStackTrace(); 
//					}
//				}
//				
//				private Thread initialize(int core, int numOfCores) {
//					this.core = core;
//					this.numOfCores = numOfCores;
//					return this;
//				}
//			}).initialize(k, numberOfCores));
//			
//			threads.get(k).start();
//		}
//		
//		for(int k=0;k<numberOfCores;++k){
//			try {
//				threads.get(k).join();
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			} 
//		}
//	}
	
//	@Override
//	public double test(){
//		int numberOfCores = Runtime.getRuntime().availableProcessors();
//		ArrayList<Thread> threads = new ArrayList<Thread>();
//		
//		// Init all users in user list.
//		for(int i=0; i<m_userList.size(); i++){
//			m_userList.get(i).getPerfStat().clear();
//		}
//		for(int k=0; k<numberOfCores; ++k){
//			threads.add((new Thread() {
//				int core, numOfCores;
//				public void run() {
//					_AdaptStruct user;
//					_PerformanceStat userPerfStat;
//					try {
//						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
//							user = m_userList.get(i+core);
//							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
//								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
//								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
//								continue;
//								
//							userPerfStat = user.getPerfStat();								
//							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
//								//record prediction results
//								for(_Review r:user.getReviews()) {
//									if (r.getType() != rType.TEST)
//										continue;
//									int trueL = r.getYLabel();
//									int predL = user.predict(r); // evoke user's own model
//									userPerfStat.addOnePredResult(predL, trueL);
//								}
//							}							
//							userPerfStat.calculatePRF();	
//						}
//					} catch(Exception ex) {
//						ex.printStackTrace(); 
//					}
//				}
//				
//				private Thread initialize(int core, int numOfCores) {
//					this.core = core;
//					this.numOfCores = numOfCores;
//					return this;
//				}
//			}).initialize(k, numberOfCores));
//			
//			threads.get(k).start();
//		}
//		
//		for(int k=0;k<numberOfCores;++k){
//			try {
//				threads.get(k).join();
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			} 
//		}
//		
//		int count = 0;
//		double[] macroF1 = new double[m_classNo];
//		_PerformanceStat userPerfStat;
//		m_microStat.clear();
//		for(_AdaptStruct user:m_userList) {
//			if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
//				|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
//				|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
//				continue;
//			
//			userPerfStat = user.getPerfStat();
//			for(int i=0; i<m_classNo; i++)
//				macroF1[i] += userPerfStat.getF1(i);
//			m_microStat.accumulateConfusionMat(userPerfStat);
//			count ++;
//		}
//		
//		System.out.println(toString());
//		calcMicroPerfStat();
//		
//		// macro average
//		m_perf = new double[2];
//		System.out.println("\nMacro F1:");
//		for(int i=0; i<m_classNo; i++){
//			System.out.format("Class %d: %.4f\t", i, macroF1[i]/count);
//			m_perf[i] = macroF1[i]/count;
//		}
//		m_perfs.add(m_perf);
//		System.out.println("\n");
//		return Utils.sumOfArray(macroF1);
//	}
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
					_HDPAdaptStruct user;
					_PerformanceStat userPerfStat;

					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							user = (_HDPAdaptStruct) m_userList.get(i+core);
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
	
	// we want to test if one cluster's model works better than others.
	public void sanityCheck(int k){
		setSupModel();
		// we first collect the test review size for each hdpthetastar.
		_HDPAdaptStruct user;
		HashMap<Integer, ArrayList<_Review>> indexRvwMap = new HashMap<Integer, ArrayList<_Review>>();
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.TEST)
					continue;
				int index = r.getHDPThetaStar().getIndex();
				if(!indexRvwMap.containsKey(index))
					indexRvwMap.put(index, new ArrayList<_Review>());
				indexRvwMap.get(index).add(r);
			}
		}
		MyPriorityQueue<_RankItem> q = new MyPriorityQueue<_RankItem>(k);
		for(int in: indexRvwMap.keySet()){
			q.add(new _RankItem(in, indexRvwMap.get(in).size()));
		}
		ArrayList<_RankItem> rq = new ArrayList<_RankItem>();
		for(_RankItem it: q)
			rq.add(it);
		Collections.sort(rq, new Comparator<_RankItem>(){
			@Override
			public int compare(_RankItem r1, _RankItem r2){
				return (int) (r2.m_value - r1.m_value);
			}
		});
		int[] indexes = new int[rq.size()];
		for(int i=0; i<rq.size(); i++)
			indexes[i] = rq.get(i).m_index;
		double[][][] perf = new double[k][k][2];
		int i = 0;// thetastar[k]
		for(int in: indexes){
			int j = 0;
			_HDPThetaStar theta = m_hdpThetaStars[in];
			System.out.print(indexRvwMap.get(in).size() + "\t");
			for(int subin: indexes){
				perf[i][j] = calcPerf(theta, indexRvwMap.get(subin));
				System.out.print(String.format("%.4f/%.4f\t", perf[i][j][0], perf[i][j][1]));
				j++;
			}
			System.out.println();
			i++;
		}
	}
	
	public double[] calcPerf(_HDPThetaStar theta, ArrayList<_Review> rs){
		int[][] TPTable = new int[m_classNo][m_classNo];
		for(_Review r: rs){
			int predL = predict(theta, r);
			int trueL = r.getYLabel();
			TPTable[predL][trueL]++;
		}
		double[] prf = new double[6];
		for (int i = 0; i < m_classNo; i++) {
			prf[3*i] = (double) TPTable[i][i] / (Utils.sumOfRow(TPTable, i) + 0.00001);// Precision of the class.
			prf[3*i + 1] = (double) TPTable[i][i] / (Utils.sumOfColumn(TPTable, i) + 0.00001);// Recall of the class.
			prf[3*i + 2] = 2 * prf[3 * i] * prf[3 * i + 1] / (prf[3 * i] + prf[3 * i + 1] + 0.00001);
		}
		return new double[]{prf[2], prf[5]};
	}
	
	public void setSupModel(){
		for(int i=0; i<m_featureSize+1; i++)
			m_supWeights[i] = getSupWeights(i);
	}
	public int predict(_HDPThetaStar theta, _Review r){
		
		double[] As = theta.getModel();
		double prob, sum = As[0]*m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
		int m, n;
		for(_SparseFeature fv: r.getSparse()){
			n = fv.getIndex() + 1;
			m = m_featureGroupMap[n];
			sum += (As[m]*m_supWeights[n] + As[m_dim+m]) * fv.getValue();
		}
		prob = Utils.logistic(sum);
		return prob > 0.5 ? 1 : 0;
	}
}
