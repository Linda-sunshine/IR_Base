package Application;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import structures._Review;
import structures._User;

public class CollaborativeFilteringWithAllNeighbors extends CollaborativeFiltering{
	int m_pop;
	
	public CollaborativeFilteringWithAllNeighbors(ArrayList<_User> users, int fs, int pop) {
		super(users, fs);
		m_pop = pop;
	}

	// for one item of a user, find the other users who have reviewed this item.
	// collection their other purchased items for ranking.
	@Override
	public void constructRankingNeighbors(){
		double sum = 0;
		double avgRvwSize = 0, rvwSize = 0;

		for(_User user: m_users){
			rvwSize = 0;
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
				rvwSize++;
				
				// and access all the users who have rated this item in training
				ArrayList<String> trainNeis = m_trainMap.get(itemID);
				for(String nid: trainNeis){
					if(nid.equals(user.getUserID())){
						System.out.println("The user has rated the same item twice!");
						continue;
					}
					int nIdx = m_userIDIndex.get(nid);
					_User nei = m_userMap.get(nIdx);
					for(_Review r: nei.getTrainReviews()){
						if(m_trainMap.get(r.getItemID()).size() < m_pop)
							continue;
						items.add(r.getItemID());
					}
				}
			}
			user.setRankingItems(items);
			sum += items.size();
			if(items.size()>=1){
				m_validUser++;
				avgRvwSize += rvwSize;
			}
		}
		System.out.format("[Stat]Pop: %d, Valid user: %s, avg candidate item: %.2f, avg rvw size: %.2f.\n", m_pop, m_validUser, sum/m_validUser, avgRvwSize/m_validUser);
	}
}
