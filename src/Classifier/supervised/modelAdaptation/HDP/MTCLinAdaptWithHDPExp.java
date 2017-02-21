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
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.SVM;
import Classifier.supervised.modelAdaptation._AdaptStruct;

public class MTCLinAdaptWithHDPExp extends MTCLinAdaptWithHDP {
	double[] m_trainPerf;
	ArrayList<double[]> m_perfs = new ArrayList<double[]>();
	ArrayList<double[]> m_trainPerfs = new ArrayList<double[]>();

	boolean m_postCheck = false;
	
	GlobalSVM m_gsvm;
	public void trainGlobalSVM(){
		m_gsvm = new GlobalSVM(m_classNo, m_featureSize);
		m_gsvm.train(m_userList);
	}	
	
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
		
		//Step 4: Update the user info with the newly sampled hdpThetaStar.
		double conf = calcLogLikelihoodY(r);
		conf = Math.exp(conf);
		conf = 1 - conf*(1-conf);
		user.incHDPThetaStarMemSize(m_hdpThetaStars[k], conf);//-->3		
		
		if(k >= m_kBar)
			sampleNewCluster(k, r.getLMSparse());
	}
	
	public double calcConfidence(_Review r){
		double conf = logit(r.getSparse(), r);
		return Math.log(conf) + Math.log(1 - conf);
	}
	int m_threshold = 15;
	public void setThreshold(int t){
		m_threshold = t;
		System.out.println("[Info]Mix model threshold: "+ m_threshold);
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
//		int rvwSize = 0;
//		for(int i=0; i<m_userList.size(); i++){
//			user = (_HDPAdaptStruct) m_userList.get(i);
//			rvwSize = user.getReviews().size();
//			for(_Review r: user.getReviews()){
//				if (r.getType() != rType.TEST)
//				continue;				
//				
//				for(int k=0; k<probs.length; k++){
//					curTheta = m_hdpThetaStars[k];
//					r.setHDPThetaStar(curTheta);
//					prob = rvwSize <= m_threshold ? calcLogLikelihoodX(r) + Math.log(curTheta.getMemSize() + m_eta*curTheta.getGamma()):
//						calcLogLikelihoodX(r) + Math.log(user.getHDPThetaMemSize(curTheta) + m_eta*curTheta.getGamma());
//					probs[k] = prob;
//				}
//				logSum = Utils.logSumOfExponentials(probs);
//				for(int k=0; k<probs.length; k++)
//					probs[k] -= logSum;
//				r.setClusterPosterior(probs);//posterior in log space
//			}
//		}
//	}
	
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
	
	public void setSupModel(){
		for(int i=0; i<m_featureSize+1; i++)
			m_supWeights[i] = getSupWeights(i);
	}
//	public int predict(_HDPThetaStar theta, _Review r){
//		
//		double[] As = theta.getModel();
//		double prob, sum = As[0]*m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
//		int m, n;
//		for(_SparseFeature fv: r.getSparse()){
//			n = fv.getIndex() + 1;
//			m = m_featureGroupMap[n];
//			sum += (As[m]*m_supWeights[n] + As[m_dim+m]) * fv.getValue();
//		}
//		prob = Utils.logistic(sum);
//		return prob > 0.5 ? 1 : 0;
//	}
}
