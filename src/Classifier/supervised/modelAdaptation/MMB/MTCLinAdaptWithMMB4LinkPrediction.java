package Classifier.supervised.modelAdaptation.MMB;

/***
 * The class inherits from MTCLinAdaptWithMMB to achieve link prediction.
 * In link prediction, the train users only have train reviews and test users only have test reivews.
 * We need to calculate the mixture of each user based on review assignment and edge assignment.
 */
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures._HDPThetaStar;
import structures._RankItem;
import structures._Review;
import structures._Review.rType;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.HDP._HDPAdaptStruct;

public class MTCLinAdaptWithMMB4LinkPrediction extends MTCLinAdaptWithMMB{

	// define a friend matrix for evaluating link prediction
	private int[][] m_frdTrainMtx, m_frdTestMtx;
	private int m_trainSize = 0, m_testSize = 0;
	private double[][] m_simMtx; 
	
	// we use MAP for parameter estimation of B
	// In order to calculate the similarity, we need to use MLE to calculate the value of B
	private double[][] m_B;
	
	public MTCLinAdaptWithMMB4LinkPrediction(int classNo, int featureSize, HashMap<String, Integer> featureMap, 
			String globalModel, String featureGroupMap, String featureGroup4Sup, double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap, featureGroup4Sup, betas);
		
	}
	
	// we calculat the mixture of each user based on their review assignment and edge assignment
	// this function is used in link prediction: 
	// train users only have training reviews; test users only have testing reviews.
	protected void calculateMixturePerUser(){
		double prob, logSum;
		double[] probs = new double[m_kBar];
		
		_HDPAdaptStruct user;
		_HDPThetaStar oldTheta, curTheta;
		
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
//			if(user.getTestSize() == 0)
//				calculateTrainUserMixture();
//			if(user.getTrainSize() == 0)
//				calculateTestUserMixture();
			
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.TEST)
					continue;				
				oldTheta = r.getHDPThetaStar();
				for(int k=0; k<probs.length; k++){
					curTheta = m_hdpThetaStars[k];
					r.setHDPThetaStar(curTheta);
					prob = calcLogLikelihoodX(r) + Math.log(calcGroupPopularity(user, k, curTheta.getGamma()));
					probs[k] = prob;
				}
			
				logSum = Utils.logSumOfExponentials(probs);
				for(int k=0; k<probs.length; k++)
					probs[k] -= logSum;
				r.setClusterPosterior(probs);//posterior in log space
				r.setHDPThetaStar(oldTheta);
			}
		}
	}
	
	// In hdp or mmb, we don't assign a review to a specific cluster as we user prob vector for prediction
	// While in link prediction, we need to assign each review of testing user to one cluster 
	// as we need to compute the mixture of each user based on review assignment.
	protected void calculateGlobalMixturePerUser(){	
		
		_MMBAdaptStruct user;

		for(int i=0; i<m_userList.size(); i++){
			user = (_MMBAdaptStruct) m_userList.get(i);
			// if it is train user
			if(user.getTestSize() == 0){
				calculateMixture4TrainUser(user);
			// if it is test user
			} else{
				calculateMixture4TestUser(user);
			}
		}
	}
	
	// calculate the mixture for train user based on review assignment and edge assignment
	public void calculateMixture4TrainUser(_MMBAdaptStruct user){
		double sum = 0;
		double[] probs = new double[m_kBar];
		_HDPThetaStar theta;
		// The set of clusters for review and edge could be different, just iterate over kBar
		for(int k=0; k<m_kBar; k++){
			theta = m_hdpThetaStars[k];
			probs[k] = user.getHDPThetaMemSize(theta) + user.getHDPThetaMemSize(theta);
			sum += probs[k];
		}
		for(int k=0; k<m_kBar; k++){
			probs[k] /= sum;
		}
		user.setMixture(probs);
	}
	
	// calculate the mixture for test user based on review assignment
	public void calculateMixture4TestUser(_MMBAdaptStruct user){
		int cIndex = 0;
		double prob, logSum, sum = 0;
		double[] probs = new double[m_kBar];
		_HDPThetaStar curTheta;
		
		// calculate the cluster assignment for each review first
		for(_Review r: user.getReviews()){
			// suppose all reviews are test review in this setting
			if (r.getType() != rType.TEST)
				continue;
			
			for(int k=0; k<probs.length; k++){
				curTheta = m_hdpThetaStars[k];
				r.setHDPThetaStar(curTheta);
				prob = calcLogLikelihoodX(r) + Math.log(calcGroupPopularity(user, k, curTheta.getGamma()));
				probs[k] = prob;
			}
			// normalize the prob 
			logSum = Utils.logSumOfExponentials(probs);
			for(int k=0; k<probs.length; k++)
				probs[k] -= logSum;
			
			// take the cluster that has maximum prob as the review's cluster assignment
			curTheta = m_hdpThetaStars[Utils.argmax(probs)];
			r.setHDPThetaStar(curTheta);
			// update the cluster assignment for the user
			user.incHDPThetaStarMemSize(r.getHDPThetaStar(), 1);
		}
		// calculate the mixture: get the review assignment and normalize it
		Arrays.fill(probs, 0);
		// calculate the sum first
		for(_HDPThetaStar theta: user.getHDPTheta4Rvw()){
			sum += user.getHDPThetaMemSize(theta);
		}
		// calculate the prob for each dim
		for(_HDPThetaStar theta: user.getHDPTheta4Rvw()){
			cIndex = theta.getIndex();
			probs[cIndex] = user.getHDPThetaMemSize(theta)/sum;
		}
		user.setMixture(probs);
	}
	
	private void calcTrainSize(){
		for(_AdaptStruct user: m_userList){
			if(user.getTestSize() == 0)
				m_testSize++;
			else
				m_trainSize++;
				
		}
	}
	public void linkPrediction(){
		// calculate the global mixture for each user
		calculateGlobalMixturePerUser();
		calcTrainSize();
		
		if(m_trainSize + m_testSize != m_userList.size())
			System.err.println("Bug in calculating train and test size!");
			
		_MMBAdaptStruct ui;
		// The train user and test user may not exist in order, thus we still set the friend 
		// matrix's size as the total number of users for convenient indexing. As their dim 
		// is different, we cannot put them in one array
		m_frdTrainMtx = new int[m_userList.size()][m_trainSize-1];
		m_frdTestMtx = new int[m_userList.size()][m_userList.size()-1];
		m_simMtx = new double[m_userList.size()][m_userList.size()];
		
		// for each testing user, rank their neighbors.
		for(int i=0; i<m_userList.size(); i++){
			ui = (_MMBAdaptStruct) m_userList.get(i);
			// if it is train users
			if(ui.getTestSize() == 0)
				linkPrediction4TrainUsers(i, ui);
			else
				linkPrediction4TestUsers(i, ui);
		}
	}
	
	// for train users, we only consider train users as their friends.
	protected void linkPrediction4TrainUsers(int i, _MMBAdaptStruct ui){
		double sim = 0;
		_MMBAdaptStruct uj;
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(m_trainSize-1);
		for(int j=0; j<m_userList.size(); j++){
			uj = (_MMBAdaptStruct) m_userList.get(j);
			if(uj.getTestSize() != 0 || j == i)
				continue;
			// calculate sim
			if(j > i){
				sim = calcSimilarity(ui, uj);
				m_simMtx[i][j] = sim;
				m_simMtx[j][i] = sim;
			}
			// rank sim
			neighbors.add(new _RankItem(j, m_simMtx[i][j]));
		}
		m_frdTrainMtx[i] = rankFriends(ui, neighbors);
	}
		
	protected void linkPrediction4TestUsers(int i, _MMBAdaptStruct ui){
		double sim = 0;
		_MMBAdaptStruct uj;
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(m_userList.size()-1);
		for(int j=0; j<m_userList.size(); j++){
			uj = (_MMBAdaptStruct) m_userList.get(j);
			if(j == i)
				continue;
			// calculate sim for the pair we have not computed yet
			if(j > i && m_simMtx[j][i]  == 0) {
				sim = calcSimilarity(ui, uj);
				m_simMtx[i][j] = sim;
				m_simMtx[j][i] = sim;
			}
			// rank sim
			neighbors.add(new _RankItem(j, m_simMtx[i][j]));
		}
		m_frdTrainMtx[i] = rankFriends(ui, neighbors);
	}
	
//	protected void MLEB(){
//		m_B = new double[m_kBar][m_kBar];
//		for(int k=0; k<m_kBar; k++){
//			for(int l=0; l<m_kBar; l++){
//				m_B[k][l] = 
//			}
//		}
//	}

	// calculate the similarity between two users based on mixture
	// sim(i,j)=\sum_{k,l}\pi_{i,k}\pi_{j,l}B_{kl}
	protected double calcSimilarity(_MMBAdaptStruct ui, _MMBAdaptStruct uj){
		double sim = 0; 
		double[] mixI = ui.getMixture(), mixJ = uj.getMixture();
		
 		for(int k=0; k<m_kBar; k++){
 			for(int l=0; l<m_kBar; l++){
 				if(mixI[k] == 0 || mixJ[l] == 0)
 					continue;
 				else {
					sim += mixI[k] * mixJ[l] * m_B[k][l];
				}
 			}
 		}
 		return sim;
	}
	
	// decide if the neighbors based on similarity are real friends
	protected int[] rankFriends(_MMBAdaptStruct ui, MyPriorityQueue<_RankItem> neighbors){
		int[] frds = new int[neighbors.size()];
		_RankItem item;
		_MMBAdaptStruct uj;
		for(int i=0; i<neighbors.size(); i++){
			item = neighbors.get(i);
			uj = (_MMBAdaptStruct) m_userList.get(item.m_index);
			if(hasFriend(ui.getUser().getFriends(), uj.getUserID()))
				frds[i] = 1;	
		}
		return frds;
	}
	
	// print out the results of link prediction
	public void printLinkPrediction(String dir, int testSize){
		_MMBAdaptStruct user;
		int[] frd;
		try{
			PrintWriter trainWriter = new PrintWriter(String.format("%s/train_%d_link.txt", dir, 10000-testSize));
			PrintWriter testWriter = new PrintWriter(String.format("%s/test_%d_link.txt", dir, testSize));
			for(int i=0; i<m_userList.size(); i++){
				user = (_MMBAdaptStruct) m_userList.get(i);
				if(user.getTestSize() == 0){
					frd = m_frdTrainMtx[i];
					for(int f:frd)
						trainWriter.write(f+"\t");
					trainWriter.write("\n");
				} else{
					frd = m_frdTestMtx[i];
					for(int f: frd)
						testWriter.write(f+"\t");
					testWriter.write("\n");
				}
			}
			trainWriter.close();
			testWriter.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}
