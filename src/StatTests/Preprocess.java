package StatTests;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import structures._Review;
import structures._User;

public class Preprocess {

	// This pair is used to represent the pair for a particular restaurant.
	// <user id, rating>.
	class Pair{
		String m_userID;
		int m_rating;
		
		public Pair(String uid, int rating){
			m_userID = uid;
			m_rating = rating;
		}
		
		public String getUserID(){
			return m_userID;
		}
		public int getRating(){
			return m_rating;
		}
	}
//	HashSet<String> m_restaurants;
	HashMap<String, ArrayList<Pair>> m_rstMap;
	ArrayList<_User> m_users;
	
	public Preprocess(ArrayList<_User> users){
		m_users = users;
//		HashSet<String> restaurants = new HashSet<String>();
		m_rstMap = new HashMap<String, ArrayList<Pair>>();
	}
	
	public void getRestaurantsStat(){
		for(_User u: m_users){
			for(_Review r: u.getReviews()){
				if(!m_rstMap.containsKey(r.getItemID())){
					m_rstMap.put(r.getItemID(), new ArrayList<Pair>());
				}
				m_rstMap.get(r.getItemID()).add(new Pair(u.getUserID(), r.getYLabel()));
			}
		}
	}
	
	public void printRestaurantStat(String filename){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			for(String rstID: m_rstMap.keySet()){
				if(m_rstMap.get(rstID).size() == 1)
					continue;
				for(Pair p: m_rstMap.get(rstID)){
					writer.write(String.format("%s,%s,%d\n", rstID, p.getUserID(), p.getRating()));
				}
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
}
