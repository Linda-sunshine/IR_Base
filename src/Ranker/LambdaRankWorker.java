/**
 * 
 */
package Ranker;

import java.util.ArrayList;
import java.util.Arrays;

import structures._QUPair;
import structures._Query;
import utils.Utils;
import Ranker.LambdaRank.OptimizationType;
import Ranker.LambdaRankParallel.OperationType;
import Ranker.evaluator.Evaluator;
import Ranker.evaluator.MAP_Evaluator;
import Ranker.evaluator.NDCG_Evaluator;

/**
 * @author hongning
 *
 */
class LambdaRankWorker implements Runnable {
	ArrayList<_Query> m_queries;//list of pointers to queries
	double[] m_weight; // feature weight
	double[] m_g;//gradient		
	int[] m_order;
	
	Evaluator m_eval;
	OperationType m_type;
	double m_step, m_shrinkage, m_lambda;
	int m_maxIter, m_windowSize;
	
	double m_obj, m_perf;
	int m_misorder, m_trainingSize;
	
	public LambdaRankWorker(int maxIter, int featureSize, int windowSize, double initStep, double shrinkage,
			double lambda, OptimizationType otype) {
		m_weight = new double[featureSize];
		m_g = new double[featureSize];
		m_queries = new ArrayList<_Query>();
		m_step = initStep;
		m_maxIter = maxIter;
		m_windowSize = windowSize;
		m_shrinkage = shrinkage;
		m_lambda = lambda;
		
		if (otype.equals(OptimizationType.OT_MAP))
			m_eval = new MAP_Evaluator();
		else if (otype.equals(OptimizationType.OT_NDCG))
			m_eval = new NDCG_Evaluator(LambdaRank.NDCG_K);
		else
			m_eval = new Evaluator();
		m_eval.setRate(0.5);
	}
	
	public void addQuery(_Query q) {
		m_queries.add(q);
	}
	
	public void clearQueries() {
		m_queries.clear();
	}
	
	public void setType(OperationType type){
		m_type = type;
	}
	
	public void setWeight(double[] weight) {
		System.arraycopy(weight, 0, m_weight, 0, m_weight.length);
	}
	
	public double[] getWeight() {
		return m_weight;
	}
	
	public int getQuerySize() {
		return m_queries.size();
	}
	
	@Override
	public void run() {
		if (m_type==OperationType.OT_train)
			train();
		else if (m_type==OperationType.OT_evaluate)
			evaluate();			
	}
	
	void init() {
		m_trainingSize = m_queries.size();
		m_order = new int[m_trainingSize];
		for(int i=0; i<m_order.length; i++)
			m_order[i] = i;
	}
	
	protected int gradientUpdate(_Query query){
		double diff;
		int i, trainSize = 0;
		
		//Step 1: calculate the ranking score
		for(_QUPair pair : query.m_docList)
			pair.score(m_weight);
		m_eval.eval(query);
		
		//Step 2: accumulate the lambdas for each URL
		for(_QUPair pair : query.m_docList){			
			diff = 0;
			if (pair.m_worseURLs!=null){
				for(_QUPair worseURL:pair.m_worseURLs){//force to moving up
					diff += Utils.logistic(worseURL.m_score-pair.m_score) * m_eval.delta(pair, worseURL);
					trainSize ++;
				}
			}
			
			if (pair.m_betterURLs!=null){
				for(_QUPair betterURL:pair.m_betterURLs){//force to moving down
					diff -= Utils.logistic(pair.m_score-betterURL.m_score) * m_eval.delta(betterURL, pair);
					trainSize ++;
				}
			}

			//Step 3: update weight according to this URL
			if (diff!=0){
				for(i=0; i<pair.m_rankFv.length; i++)
					m_g[i] -= diff * pair.m_rankFv[i];
			}
		}
		
		return trainSize;
	}
	
	public void train(){
		double mu;		
		int qid, i, j, pSize;
		for(int n=0; n<m_maxIter; n++){
			Utils.shuffle(m_order, m_trainingSize);
			qid = 0;
			while(qid<m_trainingSize){
				pSize = 0;
				Arrays.fill(m_g, 0.0);
				
				for(j=0; j<m_windowSize; j++){//collect the gradients in mini-batch
					pSize += gradientUpdate(m_queries.get(m_order[qid%m_trainingSize]));
					qid ++;
				}
				
				//Step 4: gradient from regularization
				for(i=0; i<m_weight.length; i++)
					m_g[i] = m_g[i]/pSize + m_lambda * m_weight[i];
				
				mu = Math.random()*m_step;
				for(i=0; i<m_weight.length; i++)
					m_weight[i] -= mu * m_g[i];
			}			
			
			m_step *= m_shrinkage;
		}
	}
	
	protected void evaluate(){
		double r;
		
		m_obj = 0;
		m_perf = 0;
		m_misorder = 0;
		
		for(_Query query:m_queries){
			//calculate ranking score with latest weight
			for(_QUPair pair : query.m_docList)
				pair.score(m_weight);
			
			if ((r=m_eval.eval(query))>=0)//ranking score should already be calculated
				m_perf += r;
			
			for(_QUPair pair : query.m_docList){
				if (pair.m_worseURLs!=null){
					for(_QUPair worseURL:pair.m_worseURLs){
						if ((r=Utils.logistic(pair.m_score-worseURL.m_score))>0)
							m_obj += Math.log(r);
						if (pair.m_score<=worseURL.m_score)
							m_misorder++;
					}
				}
				
				if (pair.m_betterURLs!=null){
					for(_QUPair betterURL:pair.m_betterURLs){
						if ((r=Utils.logistic(betterURL.m_score-pair.m_score))>0)
							m_obj += Math.log(r);
						if (pair.m_score>=betterURL.m_score)
							m_misorder++;
					}
				}
			}
		}		
		m_misorder /= 2;
	}
}

