package CoLinAdapt;

import java.util.ArrayList;
import structures.MyLinkedList;
import structures.MyPriorityQueue;
import structures._RankItem;
import structures._Review;
import structures._User;
import utils.Utils;

public class CoLinAdaptSchedule extends LinAdaptSchedule {
	double[] m_similarity;//It contains all user pair's similarity.
	
	public CoLinAdaptSchedule(ArrayList<_User> users, int featureNo, int featureGroupNo, int[] featureGroupIndexes){
		super(users, featureNo, featureGroupNo, featureGroupIndexes);
	}
	
	//Fill in the user related information map and array.
	public void initSchedule() {
		_User user;
		for (int i = 0; i < m_users.size(); i++) {
			user = m_users.get(i);
			m_userIDs[i] = user.getUserID();
			m_userIDIndexMap.put(user.getUserID(), i);
			user.initCoLinAdapt(m_featureGroupNo, m_featureNo, m_globalWeights, m_featureGroupIndexes); // Init each user's CoLinAdapt model.
		}
	}
	//Specify the neighbors of the current user.
	public void constructNeighborhood(){
		ArrayList<_User> neighbors;
		_User user;
		for(int i=0; i<m_users.size(); i++){
			user = m_users.get(i);
			neighbors = new ArrayList<_User>();
			for(int j=0; j<m_users.size(); j++){
				if(j != i)
					neighbors.add(m_users.get(j));
			}
			//For testing purpose, we use all others as neighbors.
			user.setNeighbors(neighbors); // Pass the references to the user as neighbors.
			user.setCoLinAdaptNeighbors(); //Pass neighbors to CoLinAdapt model.
		}
	}
	
	public void constructNeighborhood(int topK){
		_User user;
		ArrayList<_User> neighbors = new ArrayList<_User>();
//		ArrayList<Integer> neighborIndexes = new ArrayList<Integer>();
		ArrayList<Double> neighborSims = new ArrayList<Double>();
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(topK);
		
		for(int i=0; i<m_users.size(); i++){
			user = m_users.get(i);
			for(int j=0; j<m_users.size(); j++){
				if(i != j)
					queue.add(new _RankItem(j, m_similarity[getIndex(i, j)]));//Sort all the neighbors based on similarity.
			}
			// Add the neighbors.
			for(_RankItem item: queue){
				neighbors.add(m_users.get(item.m_index));
//				neighborIndexes.add(item.m_index);
				neighborSims.add(item.m_value);
			}
			user.setNeighbors(new ArrayList<_User>(neighbors));//Set the neighbors for the user.
			user.setCoLinAdaptNeighbors(); //Pass neighbors to the coLinAdapt model.
//			user.setCoLinAdaptNeighborIndexes(neighborIndexes);
			user.setCoLinAdpatNeighborSims(new ArrayList<Double>(neighborSims));
			queue.clear();
			neighbors.clear();
			neighborSims.clear();
			neighborSims.clear();
			
		}
	}
	//In the online mode, train them one by one and consider the order of the reviews.
	public void onlineTrain(){
		_Review tmp, next;
		int userIndex, predL;
		CoLinAdapt model;
		int count = 0;
		m_trainQueue = new MyLinkedList<_Review>();//We use this to maintain the review pool.
		//Construct the initial pool.
		for(_User u: m_users)
			m_trainQueue.add(u.getOneReview()); //Collect one review from each user.
		
		while(!m_trainQueue.isEmpty()){
			//Get the head review: the most recent one.
			tmp = m_trainQueue.poll(); 
			userIndex = m_userIDIndexMap.get(tmp.getUserID());
			next = m_users.get(userIndex).getOneReview();
			if(next != null)
				m_trainQueue.add(next);
			
			// Predict first.
			model = m_users.get(userIndex).getCoLinAdapt();
			model.setAs();
			predL = model.predict(tmp);
			model.addOnePredResult(predL, tmp.getYLabel());
			model.train(tmp);
			count++;
			if(count % 1000 == 0)
				System.out.print(".");
		}
//		System.out.println("total: " + count);
	}

	//In batch mode, we use half of one user's reviews as training set and we concatenate all users' reviews.
	public void batchTrainTest() {
		SyncCoLinAdapt sync = new SyncCoLinAdapt(m_featureGroupNo, m_featureNo, m_globalWeights, m_featureGroupIndexes, m_users);
//		CoLinAdapt model;
		ArrayList<_Review> reviews;
		int pivot = 0;

		ArrayList<_Review> trainSet = new ArrayList<_Review>();
		ArrayList<_Review> testSet = new ArrayList<_Review>();
		
		// Traverse all users and train their models based on the half of their reviews.
		for (int i = 0; i < m_users.size(); i++) {
//			model = m_users.get(i).getCoLinAdapt();
			reviews = m_users.get(i).getReviews();
			pivot = reviews.size() / 2;
			// Split the reviews into two parts, one for training and another for testing.
			for (int j = 0; j < reviews.size(); j++) {
				if (j < pivot)
					trainSet.add(reviews.get(j));
				else
					testSet.add(reviews.get(j));
			}
		}
		sync.init();
		sync.setSimilarities(m_similarity);
		sync.train(trainSet);// Train the model.
		sync.test(testSet);
	}
	
	//Calculate each user's performance.
	public void calcPerformance(){
		CoLinAdapt model;
		for(int i=0; i<m_users.size(); i++){
			model = m_users.get(i).getCoLinAdapt();
			model.m_perfStat.calculatePRF();
			addOneUserPRF(model.m_perfStat.getOneUserPRF());
		}
		for(int i=0; i<m_avgPRF.length; i++){
			for(int j=0; j<m_avgPRF[0].length; j++){
				m_avgPRF[i][j] /= m_users.size();
			}
		}
	}
	
	public double calcSim4TwoUsers(_User ui, _User uj){
//		return 0.2;
		return 	Utils.cosine(ui.getSparse(), uj.getSparse());
	}
	
	public int getIndex(int i, int j){
		//Swap i and j.
		if(i < j){
			int t = j;
			j = i;
			i = t;
		}
		return i*(i-1)/2+j;
	}
	
	public void calcluateSimilarities(){
		m_similarity = new double[m_users.size()*(m_users.size()-1)/2];
		//Pre-compute the similarities among all users.
		for(int i=1; i<m_users.size(); i++){
			for(int j=0; j<i; j++)
				m_similarity[getIndex(i, j)] = calcSim4TwoUsers(m_users.get(i), m_users.get(j));
		}		
	}
}
