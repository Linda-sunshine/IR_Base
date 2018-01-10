package Application;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
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
			if(simSum == 0){
				System.err.println("bug in candidate!");
				return 0;
			}
			return rankSum/ simSum;
		} else{
			//Calculate the ranking value given by all neighbors and similarity;
			for(String nei: neighbors){
				if(!nei.equals(u.getUserID())){
					int neiIndex = m_userIDIndex.get(nei);
					int label = m_users.get(neiIndex).getItemRating(item)+1;
					double sim = getSimilarity(userIndex, neiIndex); 
					rankSum += m_equalWeight ? label : sim * label;//If equal weight, add label, otherwise, add weighted label.
					simSum += m_equalWeight ? 1 : sim;				
				}
			}
		}
		if(simSum == 0){
			System.err.println("bug in candidate!");
			return 0;
		} else
			return rankSum/simSum;
	}
	
	// for one item of a user, find the other users who have reviewed this item.
	// collection their other purchased items for ranking.
	@Override
	public void constructRankingNeighbors(){
		double sum = 0;
		double avgRvwSize = 0, rvwSize = 0;

		for(_CFUser user: m_users){
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
					_CFUser nei = m_userMap.get(nid);
					for(_Review r: nei.getTrainReviews()){
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
		System.out.format("[Stat]Valid user: %s, avg candidate item: %.2f, avg rvw size: %.2f.\n", m_validUser, sum/m_validUser, avgRvwSize/m_validUser);
	}
	
	// save the user-item pair to graphlab for model training.
	public void saveUserItemPairs(String dir){
		int trainUser = 0, testUser = 0, trainPair = 0, testPair = 0;
		try{
			PrintWriter trainWriter = new PrintWriter(new File(dir+"/train_all.csv"));
			PrintWriter testWriter = new PrintWriter(new File(dir+"/test_all.csv"));
			trainWriter.write("user_id,item_id,rating\n");
			testWriter.write("user_id,item_id,rating\n");
			for(_CFUser u: m_users){
				trainUser++;
				// print out the training pairs
				for(_Review r: u.getTrainReviews()){
					trainPair++;
					trainWriter.write(String.format("%s,%s,%d\n", u.getUserID(), r.getItemID(), u.getItemRating(r.getItemID())+1));
				}
				String[] rankingItems = u.getRankingItems();
				if(rankingItems == null)
					continue;
				testUser++;
				for(String item: rankingItems){
					testPair++;
					testWriter.write(String.format("%s,%s,%d\n", u.getUserID(), item, u.getItemRating(item)+1));
				}
			}
			trainWriter.close();
			testWriter.close();
			System.out.format("[Info]Finish writing (%d,%d) training users/pairs, (%d,%d) testing users/pairs.\n", trainUser, trainPair, testUser, testPair);
		} catch(IOException e){
			e.printStackTrace();
		}
		
	}
}
