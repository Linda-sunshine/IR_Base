package Application;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

import structures._RankItem;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.MMB._MMBAdaptStruct;

public class LinkPredictionWithMMBPerEdge extends LinkPredictionWithMMB{

	HashMap<String, String[]> m_trainMap = new HashMap<>();
	HashMap<String, String[]> m_testMap = new HashMap<>();
	HashMap<String, Integer> m_userIDIndexMap = new HashMap<>();
	
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
		ArrayList<_AdaptStruct> users = m_mmbModel.getUsers();
		
		for(int i= 0; i<users.size(); i++){
			_MMBAdaptStruct user = (_MMBAdaptStruct) users.get(i);
			m_testSet.add(user);
			m_userIDIndexMap.put(user.getUserID(), i);
			m_testSize++;
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
			m_mmbModel.calcMix4UsersWithAdaptReviews(user);
		}
	}
	
	// calculate the average friend number of training users, testing users.
	// it only applies to the two set of training and testing users.
	@Override
	public void calculateFrdStat(){

		double trainSum = 0, testSum = 0, testNonSum = 0, trainMiss = 0, testMiss = 0;
		for(_AdaptStruct u: m_mmbModel.getUsers()){
			// training users
			if(u.getUser().getFriendSize() == 0)
				trainMiss++;
			else
				trainSum += u.getUser().getFriendSize();
			if(u.getUser().getTestFriendSize() == 0)
				testMiss++;
			else{
				testSum += u.getUser().getTestFriendSize();
				testNonSum += u.getUser().getNonFriendSize();
			}
		}
		System.out.println(String.format("[Stat]%.1f training users have friends, %.1f testing users have friends.", 
				(m_allUserSize-trainMiss), (m_allUserSize-testMiss)));	
		System.out.println(String.format("[Stat]Avg training friend size is %.2f; avg testing friend size is %.2f; avg testing non-friend size is %.2f.\n",
				trainSum/(m_allUserSize-trainMiss), testSum/(m_allUserSize-testMiss), testNonSum/(m_allUserSize-testMiss)));	
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
	
	int m_testUser = 0, m_testPair = 0;

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
		m_testUser = 0; m_testPair = 0;
		for(int i=0; i<m_testSize; i++){
			_MMBAdaptStruct ui = m_testSet.get(i);
			if(m_testMap.containsKey(ui.getUserID()) && ui.getUser().getTestFriendSize() != 0 && ui.getUser().getNonFriendSize() !=0){
				linkPrediction4TestUsers(i, ui);
				m_testUser++;
			}
		}
		System.out.format("[Info]Finish link prediction on (%d,%d) testing users/pairs.\n", m_testUser, m_testPair);
	}
	
	// for testing user, construct user pair among all the users
	@Override
	protected void linkPrediction4TestUsers(int i, _MMBAdaptStruct ui){
		String[] nonFriends = ui.getUser().getNonFriends();
		String[] testFriends = ui.getUser().getTestFriends();
		m_testPair += testFriends.length + nonFriends.length;
		ArrayList<_RankItem> neighbors = new ArrayList<_RankItem>();
		for(String frd: testFriends){
			int neiIndex = m_userIDIndexMap.get(frd);
			neighbors.add(new _RankItem(neiIndex, m_simMtx[i][neiIndex], 1));
		}
		for(String nonfrd: nonFriends){
			int neiIndex = m_userIDIndexMap.get(nonfrd);
			neighbors.add(new _RankItem(neiIndex, m_simMtx[i][neiIndex], 0));
		}
		Collections.sort(neighbors, new Comparator<_RankItem>(){
			@Override
			public int compare(_RankItem p1, _RankItem p2){
				if(p1.m_value  < p2.m_value)
					return 1;
				else if(p1.m_value > p2.m_value)
					return -1;
				else{
					return -p1.m_label + p2.m_label;
				}
			}
		});			
		m_frdTestMtx[i] = rankFriends(ui, neighbors);
	}
	
	// decide if the neighbors based on similarity are real friends
	@Override
	protected int[] rankFriends(_MMBAdaptStruct ui, ArrayList<_RankItem> neighbors){
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
