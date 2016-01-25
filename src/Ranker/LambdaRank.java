package Ranker;

import java.util.ArrayList;
import java.util.Arrays;

import Ranker.evaluator.Evaluator;
import Ranker.evaluator.MAP_Evaluator;
import Ranker.evaluator.NDCG_Evaluator;
import cern.jet.random.tdouble.Normal;
import structures._QUPair;
import structures._Query;
import utils.Utils;

/**
 * @author wang296
 * Using stochastic gradient descent for logistic regression training
 */
public class LambdaRank {

	public enum OptimizationType {
		OT_MAP,
		OT_NDCG,
		OT_PAIR
	}
	
	public static final int NDCG_K = 40; 
	
	double m_lambda;
	int m_trainingSize;
	int[] m_signs; // sign of feature weights during random initialization
	
	ArrayList<_Query> m_queries;//list of pointers to queries
	int[] m_order;//randomly shuffle the order for stochastic gradient descent
	double[] m_weight; // feature weight
	double[] m_g;//gradient
	
	OptimizationType m_oType;
	Evaluator m_eval;
	
	public LambdaRank(int featureSize, double lambda, ArrayList<_Query> queries, OptimizationType otype) {
		m_lambda = lambda;
		m_queries = queries;
		m_weight = new double[featureSize];
		
		m_oType = otype;
		if (otype.equals(OptimizationType.OT_MAP))
			m_eval = new MAP_Evaluator();
		else if (otype.equals(OptimizationType.OT_NDCG))
			m_eval = new NDCG_Evaluator(NDCG_K);
		else
			m_eval = new Evaluator();
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
			m_weight[i] = Normal.staticNextDouble(0, lambda);
		
		if (m_signs!=null) {//enforce sign over the initial setting of feature weights
			for(int i=0; i<m_weight.length; i++) {
				if (m_signs[i]*m_weight[i]<0)
					m_weight[i] = -m_weight[i];
			}
		}
	}
	
	public void setSigns(int[] signs) {
		m_signs = signs;
	}
	
	protected void init(){//create the pointer array to the training queries
		m_trainingSize = m_queries.size();
		m_order = new int[m_trainingSize];
		for(int i=0; i<m_trainingSize; i++)
			m_order[i] = i;//shuffling the orders for SGD model training
		
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
	
	protected double evaluate(){
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
		
		perf /= total;
		obj -= 0.5 * m_lambda * Utils.L2Norm(m_weight);//to be maximized
		System.out.format("%d\t%.2f\t%.4f\n", misorder/2, obj, perf);
		
		return perf;
	}
	
	public void train(int maxIter, int windowSize, double initStep, double shrinkage){
		//output the settings
		System.out.println("[Info]LambdaRank configuration:");
		System.out.format("\tOptimization Type %s, Lambda %.3f, Shrinkage %.3f, WindowSize %d\n", m_oType, m_lambda, shrinkage, windowSize);
		System.out.format("\tInitial step size %.1f, Steps %d\n", initStep, maxIter);
		System.out.println("Misorder\tLogLilikelihood\tPerf");
		
		init();		
		double step = initStep, mu;		
		int qid, i, j, pSize;
		for(int n=0; n<maxIter; n++){
			Utils.shuffle(m_order, m_trainingSize);
			qid = 0;
			while(qid<m_trainingSize){
				pSize = 0;
				Arrays.fill(m_g, 0.0);
				
				for(j=0; j<windowSize; j++){//collect the gradients in mini-batch
					pSize += gradientUpdate(m_queries.get(m_order[qid%m_trainingSize]));
					qid ++;
				}
				
				//Step 4: gradient from regularization
				for(i=0; i<m_weight.length; i++)
					m_g[i] = m_g[i]/pSize + m_lambda * m_weight[i];
				
				mu = Math.random()*step;
				for(i=0; i<m_weight.length; i++)
					m_weight[i] -= mu * m_g[i];
			}			
			
			step *= shrinkage;
			if (n%50==0)
				evaluate();
		}
	}
}