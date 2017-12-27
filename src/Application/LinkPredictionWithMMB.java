package Application;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
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
	private double[][] m_B;
	protected int m_numberOfCores;
	protected Object m_simMtxLock = null;
	protected Object m_frdMtxLock = null;

	MTCLinAdaptWithMMB m_mmbModel = null;
	
	public LinkPredictionWithMMB(){
		m_simMtxLock = new Object();
		m_frdMtxLock = new Object();
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
	
	public void linkPrediction(){
		initLinkPred();
		
		// calculate the global mixture for each user
		m_B = m_mmbModel.MLEB();
		m_mmbModel.calculateMixturePerUser();
		
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
	
	// perform link prediction in multi-threading
	public void linkPrediction_MultiThread(){
		initLinkPred();
		// calculate the global mixture for each user
		m_B = m_mmbModel.MLEB();
		m_mmbModel.calculateMixturePerUser();
		
		// use a boolean flag to decide whether it is training set or testing set
		linkPrediction_MultiThread_Split(m_trainSet, true);
		System.out.format("[Info]Finish link prediction on %d training users.\n", m_trainSize);

		linkPrediction_MultiThread_Split(m_testSet, false);
		System.out.format("[Info]Finish link prediction on %d testing users.\n", m_testSize);
	}		
	
	// perform multi-thread on training users/testing users
	protected void linkPrediction_MultiThread_Split(ArrayList<_MMBAdaptStruct> userSet, boolean train){
		
		final boolean trainFlag = train;
		final ArrayList<_MMBAdaptStruct> users = userSet;
		final int userSize = users.size();

		m_numberOfCores = Runtime.getRuntime().availableProcessors();
		// for each training user, rank their neighbors.
		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int i=0;i<m_numberOfCores;++i){
			threads.add((new Thread() {
				int core;
				@Override
				public void run() {
					try {
						for(int j=0; j+core <userSize; j+= m_numberOfCores){
							_MMBAdaptStruct uj = users.get(j+core);
							if(trainFlag)
								linkPrediction4TrainUsers_MultiThread(j+core, uj);
							else {
								linkPrediction4TestUsers_MultiThread(j+core, uj);
							}
						}
					} catch(Exception ex){
						ex.printStackTrace();
					}
				}
				private Thread initialize(int core) {
					this.core = core;
					return this;
				}
			}).initialize(i));
			threads.get(i).start();
		}
		for(int i=0;i<m_numberOfCores;++i){
			try {
				threads.get(i).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
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
	
	// print out the results of link prediction
	public void printLinkPrediction(String dir, String model, int trainSize, int testSize){
		int[] frd;
		File dirFile = new File(dir);
		if(!dirFile.exists())
			dirFile.mkdirs();
		try{
			PrintWriter trainWriter = new PrintWriter(String.format("%s/train_%s_%d_link.txt", dir, model, trainSize));
			PrintWriter testWriter = new PrintWriter(String.format("%s/test_%s_%d_link.txt", dir, model, testSize));
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
