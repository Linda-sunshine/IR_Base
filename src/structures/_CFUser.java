package structures;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import structures._Review.rType;

public class _CFUser extends _User {
	private boolean m_isValid;
	private ArrayList<_Review> m_trainReviews;
	private ArrayList<_Review> m_testReviews;
	
	private final HashMap<String, Integer> m_itemIDRating;
	private String[] m_rankingItems;
	
	public _CFUser(String uid) {
		super(uid);
		m_isValid = true;
		m_itemIDRating = new HashMap<String, Integer>();
		constructTrainTestReviews();
	}
	
	public _CFUser(_User u){
		super(u.getUserID(), u.getClassNo(), u.getReviews(), u.getCategory());
		m_isValid = true;
		m_itemIDRating = new HashMap<String, Integer>();
		constructTrainTestReviews();
	}
	
	public void addOneItemIDRatingPair(String item, int r){
		if(!m_itemIDRating.containsKey(item))
			m_itemIDRating.put(item, r);
//			System.out.format("%s user reviewed the product %s more than once.\n", m_userID, item);
	}
	
	public boolean containsRvw(String item){
		return m_itemIDRating.containsKey(item);
	}
	
	public void constructTrainTestReviews(){
		m_trainReviews = new ArrayList<>();
		m_testReviews = new ArrayList<>();
		for(_Review r: m_reviews){
			if(r.getType() == rType.ADAPTATION)
				m_trainReviews.add(r);
			else
				m_testReviews.add(r);
		}
	}
	public int getItemRating(String item){
		// rating is 0 or 1, thus non-existing is -1
		if(m_itemIDRating.containsKey(item))
			return m_itemIDRating.get(item);
		else
			return -1;
	}
	
	public boolean isValid(){
		return m_isValid;
	}

	public int getRankingItemSize(){
		if(m_rankingItems == null)
			return 0;
		else 
			return m_rankingItems.length;
	}
	public String[] getRankingItems(){
		return m_rankingItems;
	}
	
	public ArrayList<_Review> getTrainReviews(){
		return m_trainReviews;
	}
	
	public int getTrainReviewSize(){
		return m_trainReviews.size();
	}
	
	public ArrayList<_Review> getTestReviews(){
		return m_testReviews;
	}
	
	public int getTestReviewSize(){
		return m_testReviews.size();
	}

	public HashMap<String, Integer> getItemIDRating(){
		return m_itemIDRating;
	}
	
	public void setRankingItems(Set<String> items){
		if(items.size() == 0)
			m_isValid = false; 
		else{
			m_rankingItems = new String[items.size()];
			int index = 0;
			for(String item: items){
				m_rankingItems[index++] = item;
			}
		}
	}
	
}
