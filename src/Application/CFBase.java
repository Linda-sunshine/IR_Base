package Application;

import java.util.ArrayList;
import java.util.HashMap;

import structures._Review;
import structures._User;

public class CFBase {
	
	// time is the number of random reviews = m_time*(reviews.size()-1) 
	protected int m_time; 
	protected ArrayList<_User> m_users;
	
	public CFBase(ArrayList<_User> users){
		m_users = users;
	}
	
	// For each user, construct candidate items for ranking
	// candidate size = time * review size
	public void constructRandomNeighbors(int t, HashMap<String, ArrayList<Integer>> userIDRdmNeighbors){
		m_time = t;
		_Review review;
		ArrayList<Integer> indexes;
		for(_User u: m_users){
			indexes = new ArrayList<Integer>();
			for(int i=u.getReviewSize(); i<u.getReviewSize()*m_time; i++){
				int randomIndex = (int) (Math.random() * m_totalReviews.size());
				review = m_totalReviews.get(randomIndex);
				while(u.getReviews().contains(review)){
					randomIndex = (int) (Math.random() * m_totalReviews.size());
					review = m_totalReviews.get(randomIndex);
				}
				indexes.add(randomIndex);
			}
			userIDRdmNeighbors.put(u.getUserID(), indexes);
		}
	}

	// for one item of a user, find the other users who have reviewed this item.
	// collection their other purchased items for ranking.
	public void constructRandomNeighborsAll(HashMap<String, ArrayList<Integer>> userIDRdmNeighbors){
		_User nei;
		String itemID;
		ArrayList<Integer> indexes;
		for(_User u: m_users){
			indexes = new ArrayList<Integer>();
			for(int i=0; i<u.getReviewSize(); i++){
				itemID = u.getReviews().get(i).getItemID();
				// access all the users who have purchased this item
				for(int userIndex: m_itemIDUserIndex.get(itemID)){
					nei = m_users.get(userIndex);
					// the users' other purchased items will be considered as candidate item for ranking
					for(_Review r: nei.getReviews()){
						if(!r.getItemID().equals(itemID)){
							indexes.add(m_reviewIndexMap.get(r));
						}
					}
				}
			}
			userIDRdmNeighbors.put(u.getUserID(), indexes);
		}
	}

}
