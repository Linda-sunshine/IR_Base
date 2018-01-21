package Application;

import java.util.ArrayList;

import structures.MyPriorityQueue;
import structures._RankItem;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.MMB._MMBAdaptStruct;

public class LinkPredictionWithMMBPerEdge extends LinkPredictionWithMMB{

	// calculate training/testing size, construct training set/testing set
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
		
		_MMBAdaptStruct ui;
	
		// for each user, rank their neighbors.
		for(int i=0; i<m_testSize; i++){
			ui = m_testSet.get(i);
			linkPrediction4TestUsers(i, ui);
		}
		System.out.format("[Info]Finish link prediction on %d testing users.\n", m_testCount);
	}
	
	int m_testCount = 0;
	// for testing user, construct user pair among all the users
	@Override
	protected void linkPrediction4TestUsers(int i, _MMBAdaptStruct ui){
		int count = 0;
		double sim = 0;
		int rankSize = m_allUserSize - 1 - ui.getUser().getFriendSize();
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(rankSize);
		int skip = 0;
		for(int j=0; j<m_testSize; j++){
			_MMBAdaptStruct uj = m_testSet.get(j);
			
			if(j == i){
				skip++;
				continue;
			}
			if(j > i){
				sim = calcSimilarity(ui, uj);
				m_simMtx[i][j] = sim;
				m_simMtx[j][i] = sim;
			}
			// if uj is ui's training friend
			if(ui.getUser().hasFriend(uj.getUserID())){
				skip++;
				continue;
			}
			count++;
			neighbors.add(new _RankItem(j, m_simMtx[i][j]));
		}
		if(count != rankSize)
			System.out.format("rank: %d, count: %d, skip: %d\n", rankSize, count, skip);
		if(ui.getUser().getTestFriendSize() == 0)
			return;
		m_frdTestMtx[i] = rankFriends(ui, neighbors);
		m_testCount++;
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
