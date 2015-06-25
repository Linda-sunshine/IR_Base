/**
 * 
 */
package Ranker;

import java.util.ArrayList;
import java.util.Arrays;

import structures._QUPair;
import structures._Query;
import utils.Utils;
import Ranker.evaluator.Evaluator;
import Ranker.evaluator.NDCG_Evaluator;
import cern.jet.random.tdouble.Normal;

/**
 * @author wang296
 * Using stochastic gradient descent for logistic regression training
 */
public class LambdaRank {

	double m_lambda;
	int m_trainingSize;
	
	ArrayList<_Query> m_queries;//list of pointers to queries
	int[] m_order;//randomly shuffle the order for stochastic gradient descent
	double[] m_weight; // feature weight
	double[] m_g;//gradient
	
	//Evaluator m_eval = new MAP_Evaluator();//relevance should be larger than threshold 
	Evaluator m_eval = new NDCG_Evaluator(20);
	//Evaluator m_eval = new Evaluator();
	
	public LambdaRank(int featureSize, double lambda, ArrayList<_Query> queries) {
		super();
		m_lambda = lambda;
		m_queries = queries;
		m_weight = new double[featureSize];
		m_eval.setRate(0.5);
	}
	
	public double score(double[] fv) {
		return Utils.dotProduct(fv, m_weight);
	}
	
	public double[] getWeights() {
		return m_weight;
	}
	
	protected void initWeight(double lambda){
		lambda = 1.0/Math.sqrt(lambda);
		for(int i=0; i<m_weight.length; i++)
			m_weight[i] = Normal.staticNextDouble(0, lambda)/100;
	}
	
	protected void init(){//create the pointer array to the training queries
		m_trainingSize = m_queries.size();
		m_order = new int[m_trainingSize];
		m_g = new double[m_weight.length];
		
		initWeight(m_lambda);
	}
	
	//for lambdaRank
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
	
	protected double evaluate(double lambda, boolean print){
		double obj = 0, perf = 0, total = 0, r;
		int misorder = 0;
		
		for(_Query query:m_queries){
			if ((r=m_eval.eval(query))>=0){//ranking score should already be calculated
				perf += r;
				total ++;
			}
			
			for(_QUPair pair : query.m_docList){
				if (pair.m_worseURLs!=null){
					for(_QUPair worseURL:pair.m_worseURLs){
						if ((r=Utils.logistic(pair.m_score-worseURL.m_score))>0)
							obj += Math.log(r);
						if (pair.m_score<=worseURL.m_score)
							misorder++;
					}
				}
				
				if (pair.m_betterURLs!=null){
					for(_QUPair betterURL:pair.m_betterURLs){
						if ((r=Utils.logistic(betterURL.m_score-pair.m_score))>0)
							obj += Math.log(r);
						if (pair.m_score>=betterURL.m_score)
							misorder++;
					}
				}
			}
		}		
		
		if (print){
			obj = obj - 0.5 * lambda * Utils.L2Norm(m_weight);//to be maximized
			System.out.format("%d\t%.4f\t%.4f\n", misorder/2, obj, perf/total);
		}
		return perf/total;
	}
	
	public void train(int maxIter, int k, double initStep){
		init();		
		double step = initStep, mu;		
		int qid, i, j, pSize;
		for(int n=0; n<maxIter; n++){
			Utils.shuffle(m_order, m_trainingSize);
			qid = 0;
			while(qid<m_trainingSize){
				pSize = 0;
				Arrays.fill(m_g, 0.0);
				
				for(j=0; j<k; j++){//collect the gradients in mini-batch
					pSize += gradientUpdate(m_queries.get(m_order[qid%m_trainingSize]));
					qid ++;
				}
				
				//Step 4: gradient from regularization
				for(i=0; i<m_weight.length; i++)
					m_g[i] = m_g[i] + m_lambda * m_weight[i];
				
				mu = Math.random()*step;
				for(i=0; i<m_weight.length; i++)
					m_weight[i] -= mu * m_g[i];
			}
						
			if (n%4==0){
				step /= 1.15;
				if (n%12==0){
					evaluate(m_lambda, true);
				}
			}
		}
	}
}
