package Application;

import java.util.ArrayList;

import structures._Review;
import structures._User;

public class CollaborativeFilteringWithAllNeighbors extends CollaborativeFiltering{

	public CollaborativeFilteringWithAllNeighbors(ArrayList<_User> users) {
		super(users);
	}
	
	public CollaborativeFilteringWithAllNeighbors(ArrayList<_User> users, int fs) {
		super(users, fs);
	}

	//calculate the ranking score for each review of each user.
	//The ranking score is calculated based on all the users who have reviewed the item
	@Override
	public double calculateRankScore(_User u, _Review r){
		int userIndex = m_userIDIndex.get(u.getUserID());
		double rankSum = 0;
		double simSum = 0;
		String itemID = r.getItemID();
			
		//select top k users who have purchased this item.
		ArrayList<Integer> candidates = m_itemIDUserIndex.get(itemID);
		if(m_avgFlag){
			for(int c: candidates){
				double label = m_users.get(c).getItemIDRating().get(itemID)+1;
				rankSum += label;
				simSum++;
			}
			return rankSum/ simSum;
		} else{
			//Calculate the ranking value given by all neighbors and similarity;
			for(int c: candidates){
				if(c != userIndex){
					int label = m_users.get(c).getItemIDRating().get(itemID)+1;
					double sim = getSimilarity(userIndex, c); 
					rankSum += m_equalWeight ? label : sim * label;//If equal weight, add label, otherwise, add weighted label.
					simSum += m_equalWeight ? 1 : sim;
				}
			}
		}
		if(simSum == 0) 
			return 0;
		else
			return rankSum/simSum;
	}
}
