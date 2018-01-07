package Application;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import structures._CFUser;
import structures._Review;
import structures._User;

public class CollaborativeFilteringWithAllNeighbors extends CollaborativeFiltering{

	public CollaborativeFilteringWithAllNeighbors(ArrayList<_User> users){
		super(users);
	}
	
	public CollaborativeFilteringWithAllNeighbors(ArrayList<_CFUser> users, int fs) {
		super(users, fs);
	}

	//calculate the ranking score for each review of each user.
	//The ranking score is calculated based on all the users who have reviewed the item
	@Override
	public double calculateRankScore(_CFUser u, String item){
		int userIndex = m_userIDIndex.get(u.getUserID());
		double rankSum = 0;
		double simSum = 0;
			
		ArrayList<String> neighbors = m_trainMap.get(item);
		if(m_avgFlag){
			for(String nei: neighbors){
				if(!nei.equals(u.getUserID())){
					int index = m_userIDIndex.get(nei);
					double label = m_users.get(index).getItemIDRating().get(item)+1;
					rankSum += label;
					simSum++;
				}
			}
			return rankSum/ simSum;
		} else{
			//Calculate the ranking value given by all neighbors and similarity;
			for(String nei: neighbors){
				if(!nei.equals(u.getUserID())){
					int neiIndex = m_userIDIndex.get(nei);
					int label = m_users.get(neiIndex).getItemIDRating().get(item)+1;
					double sim = getSimilarity(userIndex, neiIndex); 
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
	
	// for one item of a user, find the other users who have reviewed this item.
	// collection their other purchased items for ranking.
	@Override
	public void constructRankingNeighbors(){
		double sum = 0;
		for(_CFUser user: m_users){
			ArrayList<_Review> testReviews = user.getTestReviews();
			Set<String> items = new HashSet<String>();
			
			// for each test review, construct the ranking array
			for(_Review tr: testReviews){
				
				String itemID = tr.getItemID();
				// if this item is not in the training map, ignore it.
				if(!m_trainMap.containsKey(itemID))
					continue;
				
				// add the user's own reviewed item
				items.add(tr.getItemID());
				
				// and access all the users who have rated this item in training
				ArrayList<String> trainNeis = m_trainMap.get(itemID);
				for(String nid: trainNeis){
					_CFUser nei = m_userMap.get(nid);
					for(_Review r: nei.getTrainReviews()){
						items.add(r.getItemID());
					}
				}
			}
			user.setRankingItems(items);
			sum += items.size();
			m_validUser += items.size() == 0 ? 0 : 1;
		}
		System.out.format("[Stat]Avg candidate item is %.2f.\n", sum/m_validUser);
	}
	
}
