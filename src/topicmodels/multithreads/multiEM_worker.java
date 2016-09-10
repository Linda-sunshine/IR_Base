package topicmodels.multithreads;

import structures._Doc;
import topicmodels.multithreads.updateParam_worker.RunType;

public interface multiEM_worker extends Runnable{
	public void setType(RunType type);
	
	public void addParameter(double[]param, int paramIndex);
	
	public void clearParameter();
	
	public void calculate_M_step();
	
	public void returnParameter(double[] param, int index);
	
	public double getLogLikelihood();

	public void addDoc(_Doc d);
	
	public void clearCorpus();
	
	public double calculate_E_step(_Doc d);
	
	public double inference(_Doc d);
	
	public double accumluateStats(double[][] word_topic_sstat);
	
	public void resetStats();
		
	public double getPerplexity();
}
