package Classifier.supervised.modelAdaptation.MMB;

import java.io.File;
/***
 * The class inherits from MTCLinAdaptWithMMB to achieve link prediction.
 * In link prediction, the train users only have train reviews and test users only have test reivews.
 * We need to calculate the mixture of each user based on review assignment and edge assignment.
 */
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.tartarus.snowball.ext.frenchStemmer;

import structures.MyPriorityQueue;
import structures._HDPThetaStar;
import structures._RankItem;
import structures._Review;
import structures._Review.rType;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;

public class MTCLinAdaptWithMMB4LinkPrediction extends MTCLinAdaptWithMMB{

	// define a friend matrix for evaluating link prediction
	protected int[][] m_frdTrainMtx, m_frdTestMtx;
	protected int m_trainSize = 0, m_testSize = 0;
	protected double[][] m_simMtx; 
	protected ArrayList<_MMBAdaptStruct> m_trainSet, m_testSet;
	
	// we use MAP for parameter estimation of B
	// In order to calculate the similarity, we need to use MLE to calculate the value of B
	private double[][] m_B;
	
	public MTCLinAdaptWithMMB4LinkPrediction(int classNo, int featureSize, HashMap<String, Integer> featureMap, 
			String globalModel, String featureGroupMap, String featureGroup4Sup, double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap, featureGroup4Sup, betas);
	}
	
	// calculate training/testing size, construct training set/testing set
	public void initLinkPred(){
		m_trainSet = new ArrayList<_MMBAdaptStruct>();
		m_testSet = new ArrayList<_MMBAdaptStruct>();
		
		for(_AdaptStruct user: m_userList){
			if(user.getTestSize() != 0){
				m_testSize++;
				m_testSet.add((_MMBAdaptStruct) user);
			}
			else{
				m_trainSize++;
				m_trainSet.add((_MMBAdaptStruct) user);
			}
		}
		
		if(m_trainSize + m_testSize != m_userList.size())
			System.out.println("The user size does not match!!");
		
		// The train user and test user may not exist in order, thus we still set the friend 
		// matrix's size as the total number of users for convenient indexing. As their dim 
		// is different, we cannot put them in one array
		m_frdTrainMtx = new int[m_trainSize][m_trainSize-1];
		m_frdTestMtx = new int[m_testSize][m_userList.size()-1];
		m_simMtx = new double[m_userList.size()][m_userList.size()];
	}
	
	// we calculate the mixture of each user based on their review assignment and edge assignment
	// this function is used in link prediction: 
	// train users only have training reviews; test users only have testing reviews.
	protected void calculateMixturePerUser(){	
		
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
	
	public void linkPrediction(){
		initLinkPred();
		// calculate the global mixture for each user
		MLEB();
		calculateMixturePerUser();
		
		if(m_trainSize + m_testSize != m_userList.size())
			System.err.println("Bug in calculating train and test size!");
			
		_MMBAdaptStruct ui;
	
		// for each training user, rank their neighbors.
		for(int i=0; i<m_trainSize; i++){
			ui = m_trainSet.get(i);
			linkPrediction4TrainUsers(i, ui);
		}
		// for each testing user, rank their neighbors.
		for(int i=0; i<m_testSize; i++){
			ui = m_testSet.get(i);
			linkPrediction4TestUsers(i, ui);
		}
	}
	
	// for train users, we only consider train users as their friends.
	protected void linkPrediction4TrainUsers(int i, _MMBAdaptStruct ui){
		double sim = 0;
		_MMBAdaptStruct uj;
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(m_trainSize-1);
		for(int j=0; j<m_trainSize; j++){
			uj = m_trainSet.get(j);
			if(j == i) continue;
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
		
	// for testing user, construct user pair among all the users
	protected void linkPrediction4TestUsers(int i, _MMBAdaptStruct ui){
		double sim = 0;
		_MMBAdaptStruct uj;
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(m_userList.size()-1);
		// go through all the train users first
		for(int j=0; j<m_trainSize; j++){
			uj = m_trainSet.get(j);
			// calculate sim for the pair we have not computed yet
			sim = calcSimilarity(ui, uj);
			m_simMtx[m_trainSize+i][j] = sim;
			// rank sim
			neighbors.add(new _RankItem(j, sim));
		}
		for(int j=0; j<m_testSize; j++){
			uj = m_testSet.get(j);
			if(j == i) continue;
			if(j > i){
				sim = calcSimilarity(ui, uj);
				m_simMtx[m_trainSize+i][m_trainSize+j] = sim;
				m_simMtx[m_trainSize+j][m_trainSize+i] = sim;
			}
			neighbors.add(new _RankItem(m_trainSize+j, m_simMtx[m_trainSize+i][m_trainSize+j]));
		}
		m_frdTestMtx[i] = rankFriends(ui, neighbors);
	}
	
	// MLE of B matrix
	protected void MLEB(){
		int e_0 = 0, e_1 = 0;
		double b = 0;
		m_B = new double[m_kBar][m_kBar];
		_HDPThetaStar theta_g, theta_h;
		for(int g=0; g<m_kBar; g++){
			theta_g = m_hdpThetaStars[g];
			for(int h=0; h<m_kBar; h++){
				theta_h = m_hdpThetaStars[h];
				e_0 = theta_g.getConnectionSize(theta_h, 0);
				e_1 = theta_g.getConnectionSize(theta_h, 1);
				b = (e_1 + m_abcd[0] -1)/(e_0 + e_1 + m_abcd[0] + m_abcd[1] -2);
				m_B[g][h] = b;
				m_B[h][g] = b;
			}
		}
	}

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
			uj = item.m_index >= m_trainSize ? m_testSet.get(item.m_index-m_trainSize) : m_trainSet.get(item.m_index);
			if(hasFriend(ui.getUser().getFriends(), uj.getUserID()))
				frds[i] = 1;	
		}
		return frds;
	}
	
	// print out the results of link prediction
	public void printLinkPrediction(String dir){
		int[] frd;
		File dirFile = new File(dir);
		if(!dirFile.exists())
			dirFile.mkdirs();
		try{
			PrintWriter trainWriter = new PrintWriter(String.format("%s/train_%d_link.txt", dir, m_trainSize));
			PrintWriter testWriter = new PrintWriter(String.format("%s/test_%d_link.txt", dir, m_testSize));
			// print friends for train users
			for(int i=0; i<m_trainSize; i++){
				frd = m_frdTrainMtx[i];
				for(int f:frd)
					trainWriter.write(f+"\t");
				trainWriter.write("\n");	
			} 
			// print friends for test users
			for(int i=0; i<m_testSize; i++){
				frd = m_frdTestMtx[i];
//				System.out.println(frd.length);
				for(int f: frd)
					testWriter.write(f+"\t");
				testWriter.write("\n");
			}
			trainWriter.close();
			testWriter.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	

}
