package Application;

import java.util.ArrayList;
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures._RankItem;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.MMB._MMBAdaptStruct;

public class LinkPredictionWithMMBPerEdge extends LinkPredictionWithMMB{

	HashMap<String, String[]> m_trainMap = new HashMap<>();
	HashMap<String, String[]> m_testMap = new HashMap<>();
	// calculate training/testing size, construct training set/testing set
	public LinkPredictionWithMMBPerEdge(HashMap<String, String[]> trainMap, HashMap<String, String[]> testMap){
		m_trainMap = trainMap;
		m_testMap = testMap;
	}
	@Override
	public void calcTrainTestSize(){
		
		m_kBar = m_mmbModel.getKBar();
		m_allUserSize = m_mmbModel.getUserSize();
		m_testSet = new ArrayList<_MMBAdaptStruct>();
		for(_AdaptStruct user: m_mmbModel.getUsers()){
			m_testSize++;
			m_testSet.add((_MMBAdaptStruct) user);
		}
		calculateFrdStat();
	}
	
	@Override
	// calculate the user mixture based on their training review and edge assignment
	// since all users have training reviews, 
	public void calculateMixturePerUser(){
		ArrayList<_AdaptStruct> userList = m_mmbModel.getUsers();
		for(int i=0; i<userList.size(); i++){
			_MMBAdaptStruct user = (_MMBAdaptStruct) userList.get(i);
			m_mmbModel.calcMix4UsersNoAdaptReviews(user);
		}
	}
	
	// calculate the average friend number of training users, testing users.
	// it only applies to the two set of training and testing users.
	@Override
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
		System.out.println(String.format("[Stat]%.1f training users have friends, %.1f testing users have friends.", 
				(m_allUserSize-trainMiss), (m_allUserSize-testMiss)));	
		System.out.println(String.format("[Stat]Avg training friend size is %.2f; avg testing friend size is %.2f.\n",
				trainSum/(m_allUserSize-trainMiss), testSum/(m_allUserSize-testMiss)));	
	}
	
	@Override
	public void initLinkPred(){
		calcTrainTestSize();
		
		// The train user and test user may not exist in order, thus we still set the friend 
		// matrix's size as the total number of users for convenient indexing. As their dim 
		// is different, we cannot put them in one array
		m_frdTestMtx = new int[m_testSize][];
		m_simMtx = new double[m_allUserSize][m_allUserSize];
	}
	
	@Override
	public void linkPrediction(){
		initLinkPred();
		
		// calculate the global mixture for each user
		m_B = m_mmbModel.MLEB();
		calculateMixturePerUser();
		
		// calculate the symmetric similarity between user pairs
		for(int i=0; i<m_testSize; i++){
			_MMBAdaptStruct ui = m_testSet.get(i);
			for(int j=i+1; j<m_testSize; j++){
				_MMBAdaptStruct uj = m_testSet.get(j);
				double sim = calcSimilarity(ui, uj);
				m_simMtx[i][j] = sim;
				m_simMtx[j][i] = sim;
			}
		}
	
		// for each user, rank their neighbors.
		int testUser = 0;
		for(int i=0; i<m_testSize; i++){
			_MMBAdaptStruct ui = m_testSet.get(i);
			if(m_testMap.containsKey(ui.getUserID()) && ui.getUser().getTestFriendSize() != 0){
				linkPrediction4TestUsers(i, ui);
				testUser++;
			}
		}
		System.out.format("[Info]Finish link prediction on (%d,%d) testing users/pairs.\n", testUser, m_testPair);
	}
	
	public ArrayList<Integer> calcRankSize(int i, _MMBAdaptStruct ui){
		ArrayList<Integer> neiIndexes = new ArrayList<>();
		for(int j=0; j<m_testSize; j++){
			_MMBAdaptStruct uj = m_testSet.get(j);
			if(!m_trainMap.containsKey(uj.getUserID()))
				continue;
			if(j == i) 
				continue;
			if(ui.getUser().hasFriend(uj.getUserID()))
				continue;
			neiIndexes.add(j);
		}
		m_testPair += neiIndexes.size();
		return neiIndexes;
	}
	int m_testPair = 0;
	// for testing user, construct user pair among all the users
	@Override
	protected void linkPrediction4TestUsers(int i, _MMBAdaptStruct ui){
		ArrayList<Integer> neiIndexes = calcRankSize(i, ui);
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(neiIndexes.size());
		for(int neiIndex: neiIndexes){
			neighbors.add(new _RankItem(neiIndex, m_simMtx[i][neiIndex]));
		}
		m_frdTestMtx[i] = rankFriends(ui, neighbors);
	}
	
	// decide if the neighbors based on similarity are real friends
	@Override
	protected int[] rankFriends(_MMBAdaptStruct ui, MyPriorityQueue<_RankItem> neighbors){
		int[] frds = new int[neighbors.size()];
		for(int i=0; i<neighbors.size(); i++){
			_RankItem it = neighbors.get(i);
			_MMBAdaptStruct uj = m_testSet.get(it.m_index);
			if(ui.getUser().hasTestFriend(uj.getUserID()))
				frds[i] = 1;	
		}
		return frds;
	}
}
