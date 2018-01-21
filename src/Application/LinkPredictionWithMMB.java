package Application;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures._RankItem;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.MMB.MTCLinAdaptWithMMB;
import Classifier.supervised.modelAdaptation.MMB._MMBAdaptStruct;

/***
 * The class inherits from MTCLinAdaptWithMMB to achieve link prediction.
 * In link prediction, the train users only have train reviews and test users only have test reivews.
 * We need to calculate the mixture of each user based on review assignment and edge assignment.
 */

public class LinkPredictionWithMMB {

	// define a friend matrix for evaluating link prediction
	protected double[][] m_simMtx; 
	protected int[][] m_frdTrainMtx, m_frdTestMtx;
	protected int m_trainSize = 0, m_testSize = 0, m_allUserSize = 0, m_kBar = 0;
	protected ArrayList<_MMBAdaptStruct> m_trainSet, m_testSet;
	
	// we use MAP for parameter estimation of B
	// In order to calculate the similarity, we need to use MLE to calculate the value of B
	protected double[][] m_B;
	protected int m_numberOfCores;
	protected Object m_simMtxLock = null;
	protected Object m_frdMtxLock = null;
	protected Object m_NDCGMAPLock = null;
	
	MTCLinAdaptWithMMB m_mmbModel = null;
	
	double[] m_NDCGs, m_MAPs;
	
	public LinkPredictionWithMMB(){
		m_simMtxLock = new Object();
		m_frdMtxLock = new Object();
		m_NDCGMAPLock = new Object();
	}
	
	public void calcTrainTestSize(){
		m_kBar = m_mmbModel.getKBar();
		m_allUserSize = m_mmbModel.getUserSize();
		m_trainSet = new ArrayList<_MMBAdaptStruct>();
		m_testSet = new ArrayList<_MMBAdaptStruct>();
		for(_AdaptStruct user: m_mmbModel.getUsers()){
			if(user.getTestSize() != 0){
				m_testSize++;
				m_testSet.add((_MMBAdaptStruct) user);
			}
			else{
				m_trainSize++;
				m_trainSet.add((_MMBAdaptStruct) user);
			}
		}
		
		if(m_trainSize + m_testSize != m_allUserSize)
			System.out.println("The user size does not match!!");
		
		calculateFrdStat();
	}

	// calculate the average friend number of training users, testing users.
	// it only applies to the two set of training and testing users.
	public void calculateFrdStat(){

		double trainSum = 0, testSum = 0, trainMiss = 0, testMiss = 0;
		for(_AdaptStruct u: m_mmbModel.getUsers()){
			// training users
			if(u.getUser().getFriendSize() == 0)
				trainMiss++;
			else
				trainSum += u.getUser().getFriendSize();
			if(u.getUser().getTestFriendSize() == 0)
				testMiss++;
			else
				testSum += u.getUser().getTestFriendSize();
		}
		System.out.println(String.format("[Stat]%d training users don't have friends, %d testing users don't have friends.", 
				(m_allUserSize-trainMiss), (m_allUserSize-testMiss)));	
		System.out.println(String.format("[Stat]Avg training friend size is %.2f; avg testing friend size is %.2f.\n",
				trainSum/(m_allUserSize-trainMiss), testSum/(m_allUserSize-testMiss)));	
	}
	
	public MTCLinAdaptWithMMB getMMB(){
		return m_mmbModel;
	}
	
	public void initMMB(int classNo, int featureSize, HashMap<String, Integer> featureMap, 
			String globalModel, String featureGroupMap, String featureGroup4Sup, double[] betas){
		m_mmbModel = new MTCLinAdaptWithMMB(classNo, featureSize, featureMap, globalModel, featureGroupMap, featureGroup4Sup, betas);
		
	}
	// calculate training/testing size, construct training set/testing set
	public void initLinkPred(){
		
		calcTrainTestSize();
		// The train user and test user may not exist in order, thus we still set the friend 
		// matrix's size as the total number of users for convenient indexing. As their dim 
		// is different, we cannot put them in one array
		m_frdTrainMtx = new int[m_trainSize][m_trainSize-1];
		m_frdTestMtx = new int[m_testSize][m_allUserSize-1];
		m_simMtx = new double[m_allUserSize][m_allUserSize];
	}
	
	// calculate the global mixture of each user:
	// we calculate the mixture of each user based on their review assignment and edge assignment
	// train users only have training reviews; test users only have testing reviews.
	public void calculateMixturePerUser(){
		ArrayList<_AdaptStruct> userList = m_mmbModel.getUsers();
		for(int i=0; i<userList.size(); i++){
			_MMBAdaptStruct user = (_MMBAdaptStruct) userList.get(i);
			// if it is train user
			if(user.getTestSize() == 0){
				m_mmbModel.calcMix4UsersWithAdaptReviews(user);
			// if it is test user
			} else{
				m_mmbModel.calcMix4UsersNoAdaptReviews(user);
			}
		}
	}
	public void linkPrediction(){
		initLinkPred();
		
		m_B = m_mmbModel.MLEB();
		calculateMixturePerUser();
			
		_MMBAdaptStruct ui;
		// for each training user, rank their neighbors.
		for(int i=0; i<m_trainSize; i++){
			ui = m_trainSet.get(i);
			linkPrediction4TrainUsers(i, ui);
		}
		System.out.format("[Info]Finish link prediction on %d training users.\n", m_trainSize);

		// for each testing user, rank their neighbors.
		for(int i=0; i<m_testSize; i++){
			ui = m_testSet.get(i);
			linkPrediction4TestUsers(i, ui);
		}
		System.out.format("[Info]Finish link prediction on %d testing users.\n", m_testSize);

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
		if(ui.getUser().getTestFriendSize() == 0)
			return;
		m_frdTrainMtx[i] = rankFriends(ui, neighbors);
	}
		
	// for train users, we only consider train users as their friends.
	protected void linkPrediction4TrainUsers_MultiThread(int i, _MMBAdaptStruct ui){
		double sim = 0;
		_MMBAdaptStruct uj;
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(m_trainSize-1);
		for(int j=0; j<m_trainSize; j++){
			uj = m_trainSet.get(j);
			if(j == i) continue;
			// calculate sim
			if(j > i){
				sim = calcSimilarity(ui, uj);
				synchronized (m_simMtxLock) {
					m_simMtx[i][j] = sim;
					m_simMtx[j][i] = sim;
				}
			}
			// rank sim
			neighbors.add(new _RankItem(j, m_simMtx[i][j]));
		}
		int[] frds = rankFriends(ui, neighbors);
		synchronized (m_frdMtxLock) {
			m_frdTrainMtx[i] = frds;
		}
	}
	
	// for testing user, construct user pair among all the users
	protected void linkPrediction4TestUsers(int i, _MMBAdaptStruct ui){
		double sim = 0;
		_MMBAdaptStruct uj;
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(m_allUserSize-1);
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
	
	// for testing user, construct user pair among all the users
	protected void linkPrediction4TestUsers_MultiThread(int i, _MMBAdaptStruct ui){
		double sim = 0;
		_MMBAdaptStruct uj;
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(m_allUserSize-1);
		// go through all the train users first
		for(int j=0; j<m_trainSize; j++){
			uj = m_trainSet.get(j);
			// calculate sim for the pair we have not computed yet
			sim = calcSimilarity(ui, uj);
			synchronized (m_simMtxLock) {
				m_simMtx[m_trainSize+i][j] = sim;
			}
			// rank sim
			neighbors.add(new _RankItem(j, sim));
		}
		for(int j=0; j<m_testSize; j++){
			uj = m_testSet.get(j);
			if(j == i) continue;
			if(j > i){
				sim = calcSimilarity(ui, uj);
				synchronized (m_simMtxLock) {
					m_simMtx[m_trainSize+i][m_trainSize+j] = sim;
					m_simMtx[m_trainSize+j][m_trainSize+i] = sim;
				}
			}
			neighbors.add(new _RankItem(m_trainSize+j, m_simMtx[m_trainSize+i][m_trainSize+j]));
		}
		
		int[] frds = rankFriends(ui, neighbors);
		synchronized (m_frdMtxLock) {
			m_frdTestMtx[i] = frds;
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
			if(ui.getUser().hasFriend(uj.getUserID()))
				frds[i] = 1;	
		}
		return frds;
	}
	
	// The function for calculating all NDCGs and MAPs.
	public void calculateAllNDCGMAP(){
		m_NDCGs = new double[m_testSize];
		m_MAPs = new double[m_testSize];
		Arrays.fill(m_NDCGs, -1);
		Arrays.fill(m_MAPs, -1);
			
		System.out.print("[Info]Start calculating NDCG and MAP...\n");
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
	
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				@Override
				public void run() {
					try {
						for (int i = 0; i + core <m_frdTestMtx.length; i += numOfCores) {
							if(i%500==0) System.out.print(".");
							int[] frds = m_frdTestMtx[i+core];
							if(frds == null) continue;
							double[] vals = calculateNDCGMAP(frds);
							// put the calculated nDCG into the array for average calculation
							synchronized(m_NDCGMAPLock){
								m_NDCGs[i+core] = vals[0];
								m_MAPs[i+core] = vals[1];
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
		
	public void calculateAvgNDCGMAP(){
		double avgNDCG = 0, avgMAP = 0;
		int valid = 0;
		for(int i=0; i<m_NDCGs.length; i++){
			if(m_NDCGs[i] == -1 || m_MAPs[i] == -1 || Double.isNaN(m_NDCGs[i]) || Double.isNaN(m_MAPs[i]))
				continue;
			valid++;
			avgNDCG += m_NDCGs[i];
			avgMAP += m_MAPs[i];
		}
		avgNDCG /= valid;
		avgMAP /= valid;
		System.out.format("\n[Info]Valid user size: %d, Avg NDCG, MAP -- %.5f\t%.5f\n\n", valid, avgNDCG, avgMAP);
	}
		
	// calculate the nDCG and MAP for each user
	public double[] calculateNDCGMAP(int[] rank){
		double iDCG = 0, DCG = 0, PatK = 0, AP = 0, count = 0;
				
		// sorted friend array based on the similarity
		int[] realRank = Arrays.copyOf(rank, rank.length);
		sortPrimitivesDescending(rank);
		
		//Calculate DCG and iDCG, nDCG = DCG/iDCG.
		for(int i=0; i<rank.length; i++){
			iDCG += (Math.pow(2, rank[i])-1)/(Math.log(i+2));//log(i+1), since i starts from 0, add 1 more.
			DCG += (Math.pow(2, realRank[i])-1)/(Math.log(i+2));
			if(realRank[i] >= 1){
				PatK = (count+1)/((double)i+1);
				AP += PatK;
				count++;
			}
		}
		if(Double.isNaN(DCG/iDCG))
			System.out.println("debug here!!");
		return new double[]{DCG/iDCG, AP/count};
	}
	
	public void sortPrimitivesDescending(int[] rank){
		Arrays.sort(rank);
		// then reverse the array
		for(int i=0; i<rank.length/2; i++){
			int tmp = rank[rank.length-1-i];
			rank[rank.length-i-1] = rank[i];
			rank[i] = tmp;
		}
	}
}
