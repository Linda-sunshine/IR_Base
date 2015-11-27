package CoLinAdapt;

import java.util.ArrayList;
import java.util.TreeMap;

import structures.MyLinkedList;
import structures._Review;
import structures._User;

public class CoLinAdapt extends LinAdapt {
	
	public CoLinAdapt(ArrayList<_User> users, int featureNo, int featureGroupNo, int[] featureGroupIndexes){
		super(users, featureNo, featureGroupNo, featureGroupIndexes);
	}
	
	//Specify the neighbors of the current user.
	public void constructNeighborhood(){
		
	}
	//In the online mode, train them one by one and consider the order of the reviews.
	public void onlineTrain(){
		_Review tmp;
		int userIndex, predL;
		OneLinAdapt model;
		m_trainQueue = new MyLinkedList<_Review>();//We use this to maintain the review pool.
		//Construct the initial pool.
		for(_User u: m_users)
			m_trainQueue.add(u.getOneReview()); //Collect one review from each user.
		
		while(!m_trainQueue.isEmpty()){
			//Get the head review: the most rescent one.
			tmp = m_trainQueue.poll(); 
			userIndex = m_userIDIndexMap.get(tmp.getUserID());
			m_trainQueue.add(m_users.get(userIndex).getOneReview());
			
			// Predict first.
			model = m_userModels[userIndex];
			predL = model.predict(tmp);
//			m_overallPerf.addOnePrediction(tmp.getYLabel(), predL);
			
			//Adapt based on the new review.
			ArrayList<_Review> trainSet = new ArrayList<_Review>();
			trainSet.add(tmp);
			model.train(trainSet);
		}
	}
}
