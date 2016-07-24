package topicmodels;

import java.io.File;

import structures._Corpus;

public class DCMLDA4AC extends LDA_Gibbs_Debug{
	protected double[] m_alpha;
	protected double[][] m_beta;
	
	protected double m_totalAlpha;
	protected double[] m_totalBeta;
	
	protected int m_newtonIter;
	protected double m_newtonConverge;
	
	public DCMLDA4AC(int number_of_iteration, double converge, double beta, _Corpus c, double lambda, int number_of_topics, double alpha, double  burnIn, int lag, double ksi, double tau, int newtonIter, double newtonConverge){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, ksi, tau);
	
		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];
		
		m_totalAlpha = 0;
		m_totalBeta = new double[number_of_topics];
		
		m_newtonIter = newtonIter;
		m_newtonConverge = newtonConverge;
	}
	
	public String toString(){
		return String.format("DCMLDA4AC[k:%d, alphaA:%.2f, beta:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta);
	}
	
	public void EM(){
		System.out.format("Starting %s ... \n", toString());
		
		long startTime = System.currentTimeMillis();
		
		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);
		
		String filePrefix = "./data/results/DCM_LDA";
		File weightFolder = new File(filePrefix+"");
		if(!weightFolder.exists()){
			weightFolder.mkdir();
		}
		
		double delta = 0, last = 0, current = 0;
		
		int i=0, displayCount = 0;
		do{
			
			
		}while(++i<number_of_iteration);
		
		finalEst();
		
		long endTime = System.currentTimeMillis() - startTime;
		
		System.out.format("likelihood %.3f after step %s converge to %f after %d seconds ...\n", current, i, delta, endTime/1000);
	}
	
	
	
}
