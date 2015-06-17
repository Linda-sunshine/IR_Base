package util;

public class Word implements Comparable<Word> {
	public static enum POS { NN, JJ, RB, VB, MISC };
	
	private int topic;
	public int wordNo;
	public POS pos = null;
	private LinkNode<Integer> classState = null;
	
	public Word(){
	}
	
	public Word(int wordNo){
		this.wordNo = wordNo;
	}
	
	public Word(int wordNo, int topic){
		this.topic = topic;
		this.wordNo = wordNo;
	}
	
	public int getTopic() {
		return topic;
	}
	
	public void setTopic(int topic) {
		this.topic = topic;
	}
	
	public void setClass(int classValue) {
		if (classState == null) this.classState = new LinkNode<Integer>(classValue);
		else this.classState.value = classValue;
	}
	
	public int getWordClass() {
		return this.classState.value;
	}
	
	public LinkNode<Integer> getClassState() {
		return this.classState;
	}
	
	public int getWordNo() {
		return wordNo;
	}
	
	public void setWordNo(int wordNo) {
		this.wordNo = wordNo;
	}
	
	public boolean equals(Word word) {
		return (word.wordNo == this.wordNo);
	}

	public int compareTo(Word word) {
		return this.wordNo - word.wordNo;
	}
}

