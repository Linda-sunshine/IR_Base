package structures;

public class _Review extends _Doc {
	public enum rType {
		TRAIN, // for training the global model
		ADAPTATION, // for training the personalized model
		TEST // for testing
	}
	
	String m_userID;
	String m_category; 
	rType m_type; // specification of this review
	
	//Constructor for route project.
	public _Review(int ID, String source, int ylabel){
		super(ID, source, ylabel);
	}
	
	public _Review(int ID, String source, int ylabel, String userID, String productID, String category, long timeStamp){
		super(ID, source, ylabel);
		m_userID = userID;
		m_itemID = productID;
		m_category = category;
		m_timeStamp = timeStamp;
		m_type = rType.TRAIN; // by default, every review is used for training the global model
	}
	
	public rType getType() {
		return m_type;
	}
	
	public void setType(rType type) {
		m_type = type;
	}
	
	//Compare the timestamp of two documents and sort them based on timestamps.
	@Override
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
	
	@Override
	public String toString() {
		return String.format("%s-%s-%s-%s", m_userID, m_itemID, m_category, m_type);
	}
	
	// Added by Lin for experimental purpose.
	public String getCategory(){
		return m_category;
	}
}
