/**
 * 
 */
package Ranker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import structures._Query;
import utils.Utils;

/**
 * @author wang296
 * Using stochastic gradient descent for logistic regression training
 */
public class LambdaRankParallel extends LambdaRank {
	enum OperationType {
		OT_train,
		OT_evaluate,
		OT_test
	}
	
	int m_iteration;
	final int MAX_TRAIN_ITER = 20;
	
	LambdaRankWorker[] m_workers; // a list of threads for lambdarank model
	Thread[] m_threadpool;
	
	public LambdaRankParallel(int featureSize, double lambda, ArrayList<_Query> queries, OptimizationType otype, int iteration) {
		super(featureSize, lambda, queries, otype);
		m_iteration = iteration;
	}
	
	void allocateQueries() {
		for(LambdaRankWorker worker:m_workers)
			worker.clearQueries();
		
		int workerSize = m_workers.length, workerId;
		Random rand = new Random();
		for(_Query q:m_queries){
			workerId = rand.nextInt(workerSize);
			m_workers[workerId].addQuery(q);
		}
		
		for(LambdaRankWorker worker:m_workers)
			worker.init();
	}
	
	protected int initWorkers(int windowSize, int maxIter, double initStep, double shrinkage){
		// Step 1: create the workers
		int workerSize = Runtime.getRuntime().availableProcessors();
		int i;
		m_workers = new LambdaRankWorker[workerSize];
		for(i=0; i<workerSize; i++)
			m_workers[i] = new LambdaRankWorker(maxIter, m_weight.length, windowSize, initStep, shrinkage, m_lambda/workerSize, m_oType);
		m_threadpool = new Thread[workerSize];
		
		// Step 2: initialize the global weights randomly
		initWeight(m_lambda);		
		
		return workerSize;
	}
	
	protected void WaitTillFinish(OperationType opt){
		if (opt == OperationType.OT_train)
			allocateQueries();
		
		for(LambdaRankWorker worker:m_workers) {
			worker.setWeight(m_weight);
			worker.setType(opt);//evaluation on training queries
		}
		
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
	
	public void train(int maxIter, int windowSize, double initStep, double shrinkage){
		int iteration = m_iteration;
		
		// Step 0: output the settings
		System.out.println("[Info]LambdaRank configuration:");
		System.out.format("\tOptimization Type %s, Lambda %.3f, Shrinkage %.3f, WindowSize %d\n", m_oType, m_lambda, shrinkage, windowSize);
		System.out.format("\tInitial step size %.1f, Steps %d, Iteration %d\n", initStep, maxIter, iteration);
		System.out.println("Iter\tMisorder\tLogLilikelihood\tPerf");
		
		// Step 1: setup the workers
		int workerSize = initWorkers(windowSize, maxIter, initStep, shrinkage);
		
		// Step 2: start multi-thread training
		double weight = 1.0 / workerSize, performance = 0;
		int querySize = m_queries.size();
		long starttimer = System.currentTimeMillis();
		double obj = 0, perf = 0;
		int misorder = 0;	
		
//		//evaluate initial performance
//		WaitTillFinish(OperationType.OT_evaluate);
//		for(LambdaRankWorker worker:m_workers){
//			obj += worker.m_obj;
//			perf += worker.m_perf;
//			misorder += worker.m_misorder;
//		}
//		perf /= querySize;
//		obj -= 0.5 * m_lambda * Utils.L2Norm(m_weight);//to be maximized		
//		System.out.format("0\t%d\t%.2f\t%.4f\n", misorder, obj, perf);
		
		for(int i=0; i<iteration; i++){
			// training operation
			WaitTillFinish(OperationType.OT_train);
			
			// aggregate the learned weights from workers
			Arrays.fill(m_weight, 0);
			for(LambdaRankWorker worker:m_workers)
				Utils.add2Array(m_weight, worker.getWeight(), weight);
			
			//evaluate training performance
			WaitTillFinish(OperationType.OT_evaluate);
			
			obj = 0; perf = 0; misorder = 0;
			for(LambdaRankWorker worker:m_workers){
				obj += worker.m_obj;
				perf += worker.m_perf;
				misorder += worker.m_misorder;
			}
			perf /= querySize;
			obj -= 0.5 * m_lambda * Utils.L2Norm(m_weight);//to be maximized
			
			if (performance>0){
				if (performance<=perf)//become better
					iteration--;
				else//become worse
					iteration = Math.min(MAX_TRAIN_ITER, iteration+2);
			}
			performance = perf;
			
			System.out.format("%d\t%d\t%.2f\t%.4f\n", 1+i, misorder, obj, perf);
		}

		System.out.format("[Info]Training procedure takes %.2f seconds to finish...\n", (System.currentTimeMillis()-starttimer)/1000.0);
	}
}
