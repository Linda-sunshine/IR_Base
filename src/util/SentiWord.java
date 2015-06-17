package util;


public class SentiWord extends Word {
	private int topic;
	private int sentiment;
	public Integer lexicon = null;
	
	public SentiWord(int wordNo){
		this.wordNo = wordNo;
	}
	
	public SentiWord(int wordNo, int topic){
		this.topic = topic;
		this.wordNo = wordNo;
	}
	
	public int getTopic() {
		return topic;
	}
	public void setTopic(int topic) {
		this.topic = topic;
	}

	public void setSentiment(int sentiment) {
		this.sentiment = sentiment;
	}

	public int getSentiment() {
		return sentiment;
	}
}