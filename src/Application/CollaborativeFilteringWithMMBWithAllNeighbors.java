package Application;

import java.util.ArrayList;

import structures._User;

public class CollaborativeFilteringWithMMBWithAllNeighbors extends CollaborativeFilteringWithMMB{

	public CollaborativeFilteringWithMMBWithAllNeighbors(
			ArrayList<_User> users, int fs) {
		super(users, fs);
	}
	
	//calculate the ranking score for each review of each user.
	//The ranking score is calculated based on all the users who have reviewed the item
	@Override
	public double calculateRankScore(_User u, String item){
		int userIndex = m_userIDIndex.get(u.getUserID());
		double rankSum = 0;
		double simSum = 0;
			
		ArrayList<String> neighbors = m_trainMap.get(item);
		if(m_avgFlag){
			for(String nei: neighbors){
//				if(!nei.equals(u.getUserID())){
					int index = m_userIDIndex.get(nei);
					double label = m_users.get(index).getItemRating(item)+1;
					rankSum += label;
					simSum++;
//				}
			}
			if(simSum == 0){
				System.err.println("bug in candidate!");
				return 0;
			} else
				return rankSum/ simSum;
		} else{
			//Calculate the ranking value given by all neighbors and similarity;
			for(String nei: neighbors){
//				if(!nei.equals(u.getUserID())){
					int neiIndex = m_userIDIndex.get(nei);
					int label = m_users.get(neiIndex).getItemRating(item);
					if(label == -1)
						System.out.println("[error]Wrong neighbor!");
					else{
						label++;
						double sim = getSimilarity(userIndex, neiIndex); 
						rankSum += m_equalWeight ? label : sim * label;//If equal weight, add label, otherwise, add weighted label.
						simSum += m_equalWeight ? 1 : sim;
					}				
//				}
			}
		}
		if(simSum == 0){
			System.err.println("bug in candidate!");
			return 0;
		} else
			return rankSum/simSum;
	}
}
