package structures;

public class _Review extends _Doc {
	String m_userID;
	String m_reviewID;
	String m_category; 
	
	//Constructor for test purpose.
	public _Review(int ID, String source, int ylabel, String userID, String reviewID, String category, long timeStamp){
		super(ID, source, ylabel);
		m_userID = userID;
		m_reviewID = reviewID;
		m_category = category;
		m_timeStamp = timeStamp;
	}
	
	//Compare the timestamp of two documents and sort them based on timestamps.
	public int compareTo(_Doc d){
		if(m_timeStamp < d.m_timeStamp)
			return -1;
		else if(m_timeStamp == d.m_timeStamp)
			return 0;
		else 
			return 1;
	}

	//Access the userID of the review.
	public String getUserID(){
		return m_userID;
	}
}
