package topicmodels.multithreads;

import structures._Doc;

public interface TopicModelWorker extends Runnable {

	public void addDoc(_Doc d);
	
	public double calculate_E_step(_Doc d);
	
	public double accumluateStats();
	
	public void resetStats();
}
