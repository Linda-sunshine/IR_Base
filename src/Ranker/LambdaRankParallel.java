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
import Ranker.evaluator.MAP_Evaluator;
import Ranker.evaluator.NDCG_Evaluator;
import cern.jet.random.tdouble.Normal;

/**
 * @author wang296
 * Using stochastic gradient descent for logistic regression training
 */
public class LambdaRankParallel {
	public enum OptimizationType {
		OT_MAP,
		OT_NDCG
	}
	
	enum OperationType {
		OT_train,
		OT_evaluate,
		OT_test
	}
	
	class LambdaRankWorker implements Runnable {
		ArrayList<_Query> m_queries;//list of pointers to queries
		double[] m_weight; // feature weight
		double[] m_g;//gradient		
		int[] m_order;
		
		Evaluator m_eval;
		OperationType m_type;
		double m_step;
		int m_maxIter;
		
		double m_obj, m_perf;
		int m_misorder, m_trainingSize;
		
		public LambdaRankWorker(double initStep, int maxIter) {
			m_weight = new double[m_featureSize];
			m_g = new double[m_featureSize];
			m_queries = new ArrayList<_Query>();
			m_step = initStep;
			m_maxIter = maxIter;
			
			if (m_OType.equals(OptimizationType.OT_MAP))
				m_eval = new MAP_Evaluator();
			else
				m_eval = new NDCG_Evaluator(20);
		}
		
		public void addQuery(_Query q) {
			m_queries.add(q);
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

	double m_lambda;
	int m_featureSize;
	int m_windowSize = 30;
	double m_shrinkage = 0.95;
	OptimizationType m_OType = OptimizationType.OT_MAP;
	final int MAX_TRAIN_ITER = 50;
	
	ArrayList<_Query> m_collection;//list of pointers to queries
	double[] m_model; // feature weight
	
	LambdaRankWorker[] m_workers; // a list of threads for lambdarank model
	Thread[] m_threadpool;
	
	public LambdaRankParallel(int featureSize, double lambda, ArrayList<_Query> queries) {
		m_lambda = lambda;
		m_collection = queries;
		m_featureSize = featureSize;
		m_model = new double[featureSize];		
	}
	
	public double score(double[] fv) {
		return Utils.dotProduct(fv, m_model);
	}
	
	public double[] getModel() {
		return m_model;
	}
	
	protected void initModel(double lambda){
		lambda = 1.0/Math.sqrt(lambda);
		for(int i=0; i<m_model.length; i++)
			m_model[i] = Normal.staticNextDouble(0, lambda);
	}
	
	protected int initWorkers(double initStep, int maxIter){
		// Step 1: create the workers
		int workerSize = 1; //Runtime.getRuntime().availableProcessors();
		int i;
		m_workers = new LambdaRankWorker[workerSize];
		for(i=0; i<workerSize; i++)
			m_workers[i] = new LambdaRankWorker(initStep, maxIter);
		m_threadpool = new Thread[workerSize];
		
		// Step 2: allocate the training instances evenly
		i = 0;
		for(_Query q:m_collection){
			m_workers[i%workerSize].addQuery(q);
			i++;
		}
		
		// Step 3: initialize the global weights randomly
		initModel(m_lambda);
		
		// Step 4: initialize the workers
		for(LambdaRankWorker worker:m_workers)
			worker.init();
		
		return workerSize;
	}
	
	protected void WaitTillFinish(){
		for(int i=0; i<m_threadpool.length; i++){
			m_threadpool[i] = new Thread(m_workers[i]);//everytime, we have to recreate the thread
			m_threadpool[i].start();
		}
		
		for(int i=0; i<m_threadpool.length; i++){
			try {
				m_threadpool[i].join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	public void train(double initStep, int maxIter, int iteration){
		// Step 0: output the settings
		System.out.println("[Info]LambdaRank configuration:");
		System.out.format("\tOptimization Type %s, Lambda %.3f, Shrinkage %.3f, WindowSize %d\n", m_OType, m_lambda, m_shrinkage, m_windowSize);
		System.out.format("\tInitial step size %.1f, Steps %d, Iteration %d\n", initStep, maxIter, iteration);
		System.out.println("Iter\tMisorder\tLogLilikelihood\tPerf");
		
		// Step 1: setup the workers
		int workerSize = initWorkers(initStep, maxIter);
		
		// Step 2: start multi-thread training
		double weight = 1.0 / workerSize, performance = 0;
		long starttimer = System.currentTimeMillis();
		for(int i=0; i<iteration; i++){
			// training operation
			for(LambdaRankWorker worker:m_workers){
				worker.setType(OperationType.OT_train);
				worker.setWeight(m_model);
			}
			WaitTillFinish();
			
			// aggregate the learned weights from workers
			Arrays.fill(m_model, 0);
			for(LambdaRankWorker worker:m_workers){
				//collect the results (simple average over all workers)
				Utils.add2Array(m_model, worker.getWeight(), weight);
			}
			
			//evaluate training performance
			for(LambdaRankWorker worker:m_workers)
				worker.setType(OperationType.OT_evaluate);//evaluation on training queries
			WaitTillFinish();
			
			double obj = 0, perf = 0;
			int misorder = 0, querySize = 0;
			for(LambdaRankWorker worker:m_workers){
				obj += worker.m_obj;
				perf += worker.m_perf;
				misorder += worker.m_misorder;
				querySize += worker.getQuerySize();
			}
			perf /= querySize;
			obj -= 0.5 * m_lambda * Utils.L2Norm(m_model);//to be maximized
			
			if (performance>0){
				if (performance<perf)//become better
					iteration--;
				else if (performance>perf)//become worse
					iteration = Math.min(MAX_TRAIN_ITER, iteration+2);
			}
			performance = perf;
			
			System.out.format("%d\t%d\t%.4f\t%.4f\n", i, misorder, obj, perf);
		}

		System.out.format("[Info]Training procedure takes %.2f seconds to finish...\n", (System.currentTimeMillis()-starttimer)/1000.0);
	}
}
