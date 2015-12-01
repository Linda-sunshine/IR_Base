package CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;

import structures.MyLinkedList;
import structures._Review;
import structures._User;

public class CoLinAdaptSchedule extends LinAdaptSchedule {
	double[] m_As; 
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
			user.constructNeighbors(neighbors); // Pass the references to the user as neighbors.
			user.passNeighbors2Model(); //Pass neighbors to CoLinAdapt model.
		}
	}
	
//	//Init gradients for the users.
//	public void initGradients(){
//		_User user;
//		for(int i=0; i<m_users.size(); i++){
//			user = m_users.get(i);
//			user.initGradients4CoLinAdapt();
//		}
//	}
	//In the online mode, train them one by one and consider the order of the reviews.
	public void onlineTrain(){
		_Review tmp, next;
		int userIndex, predL;
		CoLinAdapt model;
		m_trainQueue = new MyLinkedList<_Review>();//We use this to maintain the review pool.
		//Construct the initial pool.
		for(_User u: m_users)
			m_trainQueue.add(u.getOneReview()); //Collect one review from each user.
		
		while(!m_trainQueue.isEmpty()){
			//Get the head review: the most recent one.
			tmp = m_trainQueue.poll(); 
			userIndex = m_userIDIndexMap.get(tmp.getUserID());
			next = m_users.get(userIndex).getOneReview();
			if(next.equals(null))
				m_trainQueue.add(m_users.get(userIndex).getOneReview());
			
			// Predict first.
			model = m_users.get(userIndex).getCoLinAdapt();
			predL = model.predict(tmp);
			model.addOnePredResult(predL, tmp.getYLabel());
			
			model.train(tmp);
		}
	}
	
	public void updateNeighbors(double[] As){
		
	}
}
