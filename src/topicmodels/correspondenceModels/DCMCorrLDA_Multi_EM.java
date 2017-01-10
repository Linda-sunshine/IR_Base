package topicmodels.correspondenceModels;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4DCM;
import structures._Word;
import topicmodels.multithreads.multiEM_worker;
import utils.Utils;

public class DCMCorrLDA_Multi_EM extends DCMCorrLDA{
	public enum RunType {
		RT_inference, RT_E, RT_M, RT_EM
	}

	RunType m_type = RunType.RT_EM;// EM is the default type

	protected DCMCorrLDA_MultiEM_worker[] m_multiEM_worker;
	
	public class DCMCorrLDA_MultiEM_worker implements multiEM_worker {
		protected double[] m_alphaStat;
		protected ArrayList<double[]> m_param;
		protected ArrayList<_Doc> m_corpus;
		protected ArrayList<Integer> m_paramIndex;
		protected double m_likelihood;
		
		public DCMCorrLDA_MultiEM_worker(int number_of_topics,
				int vocabulary_size) {
			
			m_alphaStat = new double[number_of_topics];
			m_param = new ArrayList<double[]>();
			m_paramIndex = new ArrayList<Integer>();
			m_likelihood = 0;
			m_corpus = new ArrayList<_Doc>();
		}

		public void run() {
			
			if(m_type == RunType.RT_EM){
				System.out.println("EM error mode");
			}
			
			if(m_type == RunType.RT_E){
//				System.out.println("E step");
				for(_Doc d:m_corpus){
					calculate_E_step(d);
				}
			}else if(m_type == RunType.RT_M){
			
				for(int i=0; i<m_paramIndex.size(); i++){
					updateParam(m_param.get(i), m_paramIndex.get(i));
				}
				
				for(int i=0; i<m_paramIndex.size(); i++){
					int paramLength = m_param.get(i).length;
					int paramIndex = m_paramIndex.get(i);
					System.arraycopy(m_param.get(i), 0, m_beta[paramIndex], 0, paramLength);
				}
			}
		}

		public void addParameter(double[] t_param, int t_index) {
			int paramLen = t_param.length;
			double[] param = new double[paramLen];
			System.arraycopy(t_param, 0, param, 0, paramLen);
			m_param.add(param);
			m_paramIndex.add(t_index);
		}

		public void clearParameter() {
			for(double[] param: m_param)
				Arrays.fill(param, 0);
		}

		public void updateParam(double[] param, int tid) {
//			System.out.println("topic optimization\t"+tid);
			double diff = 0;
			int iteration = 0;
			double smoothingBeta = 0.1;
			double totalBeta = 0;
			
			do{
				diff = 0;
				
				double deltaBeta = 0;
				double wordNum = 0;
				
				double[] wordNum4V = new double[vocabulary_size];
				double totalBetaDenominator = 0;
				double[] totalBetaNumerator = new double[vocabulary_size];
				
				Arrays.fill(totalBetaNumerator, 0);
				Arrays.fill(wordNum4V, 0);
				
				totalBeta = Utils.sumOfArray(param);
				double digBeta = Utils.digamma(totalBeta);
				
				for(_Doc d:m_trainSet){
					if(d instanceof _ParentDoc){
						_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
						totalBetaDenominator += Utils.digamma(totalBeta+pDoc.m_topic_stat[tid])-digBeta;
						for(int v=0; v<vocabulary_size; v++){
							wordNum += pDoc.m_wordTopic_stat[tid][v];
							wordNum4V[v] += pDoc.m_wordTopic_stat[tid][v];
							
							totalBetaNumerator[v] += Utils.digamma(param[v]+pDoc.m_wordTopic_stat[tid][v]);
							totalBetaNumerator[v] -= Utils.digamma(param[v]);
						}
					}
					
				}
				
				for(int v=0; v<vocabulary_size; v++){
					if(wordNum == 0)
						break;
					if(wordNum4V[v] == 0){
						deltaBeta = 0;
					}else{
						deltaBeta = totalBetaNumerator[v]/totalBetaDenominator;
					}
						
					double newBeta = param[v]*deltaBeta+d_beta;
					double t_diff = Math.abs(param[v]-newBeta);
					if(t_diff>diff)
						diff = t_diff;
					
					param[v] = newBeta;
				}
				
				iteration ++;
				if(iteration > m_newtonIter)
					break;
				
				// System.out.println("beta iteration\t"+iteration);
			}while(diff > m_newtonConverge);
			
//			System.out.println("iteration\t"+iteration);
		}

		public void returnParameter(double[] param, int index) {
			if(param.length == m_param.get(index).length){
				System.arraycopy(param, 0, m_param.get(index), 0, param.length);
			}
		}

		public double getLogLikelihood() {
			return m_likelihood;

		}

		public void addDoc(_Doc d) {
			m_corpus.add(d);
		}

		public void clearCorpus() {
			// TODO Auto-generated method stub
			m_corpus.clear();
		}

		public double calculate_E_step(_Doc d) {
			d.permutation();

			if (d instanceof _ParentDoc) {
				sampleInParentDoc((_ParentDoc) d);
			} else if (d instanceof _ChildDoc) {
				sampleInChildDoc((_ChildDoc) d);
			}

			return 0;
		}

		protected void sampleInParentDoc(_Doc d) {
			_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
			int wid, tid;
			double normalizedProb;

			for (_Word w : pDoc.getWords()) {
				tid = w.getTopic();
				wid = w.getIndex();

				pDoc.m_sstat[tid]--;
				pDoc.m_topic_stat[tid]--;
				pDoc.m_wordTopic_stat[tid][wid]--;

				normalizedProb = 0;

				for (tid = 0; tid < number_of_topics; tid++) {
					double pWordTopic = parentWordByTopicProb(tid, wid, pDoc);
					double pTopicPDoc = parentTopicInDocProb(tid, pDoc);
					double pTopicCDoc = parentChildInfluenceProb(tid, pDoc);

					m_alphaStat[tid] = pWordTopic * pTopicPDoc * pTopicCDoc;
					normalizedProb += m_alphaStat[tid];
				}

				normalizedProb *= m_rand.nextDouble();
				for (tid = 0; tid < number_of_topics; tid++) {
					normalizedProb -= m_alphaStat[tid];
					if (normalizedProb <= 0)
						break;
				}

				if (tid == number_of_topics)
					tid--;

				w.setTopic(tid);
				pDoc.m_sstat[tid]++;
				pDoc.m_topic_stat[tid]++;
				pDoc.m_wordTopic_stat[tid][wid]++;
			}
		}

		protected void sampleInChildDoc(_ChildDoc d) {
			int wid, tid;
			double normalizedProb;

			_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d.m_parentDoc;

			for (_Word w : d.getWords()) {
				tid = w.getTopic();
				wid = w.getIndex();

				pDoc.m_wordTopic_stat[tid][wid]--;
				pDoc.m_topic_stat[tid]--;
				d.m_sstat[tid]--;

				normalizedProb = 0;
				for (tid = 0; tid < number_of_topics; tid++) {
					double pWordTopic = childWordByTopicProb(tid, wid, pDoc);
					double pTopic = childTopicInDocProb(tid, d, pDoc);

					m_alphaStat[tid] = pWordTopic * pTopic;
					normalizedProb += m_alphaStat[tid];
				}

				normalizedProb *= m_rand.nextDouble();
				for (tid = 0; tid < number_of_topics; tid++) {
					normalizedProb -= m_alphaStat[tid];
					if (normalizedProb <= 0)
						break;
				}

				if (tid == number_of_topics)
					tid--;

				w.setTopic(tid);
				d.m_sstat[tid]++;
				pDoc.m_topic_stat[tid]++;
				pDoc.m_wordTopic_stat[tid][wid]++;
			}
		}

		public double inference(_Doc d) {
			// TODO Auto-generated method stub
			return 0;
		}


		public double accumluateStats(double[][] word_topic_sstat) {
			// TODO Auto-generated method stub
			return 0;
		}

		public void resetStats() {
			for (int i = 0; i < m_alphaStat.length; i++)
				Arrays.fill(m_alphaStat, 0);
		}


		public double getPerplexity() {
			// TODO Auto-generated method stub
			return 0;
		}

		public void setType(RunType type) {
			// TODO Auto-generated method stub
			m_type = type;
		}

		@Override
		public void calculate_M_step() {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void setType(
				topicmodels.multithreads.updateParam_worker.RunType type) {
			// TODO Auto-generated method stub

		}
		
	}

	public DCMCorrLDA_Multi_EM(int number_of_iteration, double converge,
			double beta, _Corpus c, double lambda, int number_of_topics,
			double alpha_a, double alpha_c, double burnIn, int lag, double ksi,
			double tau, int newtonIter, double newtonConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha_a, alpha_c, burnIn, ksi, tau, lag, newtonIter,
				newtonConverge);
		// TODO Auto-generated constructor stub
//		 m_multithread = true;
	}

	public String toString() {
		return String
				.format("DCMCorrLDA_Multi_EM[k:%d, alphaA:%.2f, beta:%.2f, Gibbs Sampling]",
						number_of_topics, d_alpha, d_beta);
	}

	protected void initialize_probability(Collection<_Doc> collection) {
		super.initialize_probability(collection);

		int cores = Runtime.getRuntime().availableProcessors();

		m_threadpool = new Thread[cores];
		m_multiEM_worker = new DCMCorrLDA_MultiEM_worker[cores];

		for (int i = 0; i < cores; i++)
			m_multiEM_worker[i] = new DCMCorrLDA_MultiEM_worker(
					number_of_topics, vocabulary_size);

		int workerID = 0;
		for (int k = 0; k < number_of_topics; k++) {
			m_multiEM_worker[workerID % cores].addParameter(m_beta[k], k);
			workerID++;
		}
		
		workerID = 0;
		for (_Doc d : collection) {
			if (d instanceof _ParentDoc) {
				m_multiEM_worker[workerID % cores].addDoc(d);
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
				for (_ChildDoc cDoc : pDoc.m_childDocs) {
					m_multiEM_worker[workerID % cores].addDoc(cDoc);
				}
				workerID++;
			}
		}

		super.initialize_probability(collection);

	}

	protected void init() {
		// super.init();
		for (DCMCorrLDA_MultiEM_worker worker : m_multiEM_worker) {
			worker.resetStats();
		}
	}

	protected void updateBeta() {
		for (int i = 0; i < m_multiEM_worker.length; i++) {
			m_multiEM_worker[i].setType(RunType.RT_M);
			m_threadpool[i] = new Thread(m_multiEM_worker[i]);
			m_threadpool[i].start();
		}

		for (Thread thread : m_threadpool) {
			try {
				thread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	public void updateParameter(int iter, File weightIterFolder) {

		initialAlphaBeta();
//		updateAlpha();
//		updateAlphaC();

		updateBeta();

		for (int k = 0; k < number_of_topics; k++)
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);

		String fileName = iter + ".txt";
//		saveParameter2File(weightIterFolder, fileName);

	}

	protected void multithread_E_step() {
		for (int i = 0; i < m_multiEM_worker.length; i++) {
			m_multiEM_worker[i].setType(RunType.RT_E);
			m_threadpool[i] = new Thread(m_multiEM_worker[i]);
			m_threadpool[i].start();
		}

		for (Thread thread : m_threadpool) {
			try {
				thread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}

	}

	public void EM() {
		System.out.format("Starting %s...\n", toString());

		long starttime = System.currentTimeMillis();

		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);

		String filePrefix = "./data/results/DCMCorrLDA";
		File weightFolder = new File(filePrefix + "");
		if (!weightFolder.exists()) {
			// System.out.println("creating directory for weight"+weightFolder);
			weightFolder.mkdir();
		}

		double delta = 0, last = 0, current = 0;
		int i = 0, displayCount = 0;
		do {

			long eStartTime = System.currentTimeMillis();

			for (int j = 0; j < number_of_iteration; j++) {
				init();
				multithread_E_step();
			}
			long eEndTime = System.currentTimeMillis();

			System.out.println("per iteration e step time\t"
					+ (eEndTime - eStartTime) / 1000.0 + "\t seconds");

			long mStartTime = System.currentTimeMillis();
			updateParameter(i, weightFolder);
			long mEndTime = System.currentTimeMillis();

			System.out.println("per iteration m step time\t"
					+ (mEndTime - mStartTime) / 1000.0 + "\t seconds");

			if (m_converge > 0
					|| (m_displayLap > 0 && i % m_displayLap == 0 && displayCount > 6)) {
				// required to display log-likelihood
				current = calculate_log_likelihood();
				// together with corpus-level log-likelihood

				if (i > 0)
					delta = (last - current) / last;
				else
					delta = 1.0;
				last = current;
			}

			if (m_displayLap > 0 && i % m_displayLap == 0) {
				if (m_converge > 0) {
					System.out.format(
							"Likelihood %.3f at step %s converge to %f...\n",
							current, i, delta);
					infoWriter.format(
							"Likelihood %.3f at step %s converge to %f...\n",
							current, i, delta);

				} else {
					System.out.print(".");
					if (displayCount > 6) {
						System.out.format("\t%d:%.3f\n", i, current);
						infoWriter.format("\t%d:%.3f\n", i, current);
					}
					displayCount++;
				}
			}

			if (m_converge > 0 && Math.abs(delta) < m_converge)
				break;// to speed-up, we don't need to compute likelihood in
						// many cases
		} while (++i < this.number_of_iteration);

		finalEst();

		long endtime = System.currentTimeMillis() - starttime;
		System.out
				.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n",
						current, i, delta, endtime / 1000);
		infoWriter
				.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n",
						current, i, delta, endtime / 1000);
	}

	protected void finalEst(){
		for (int j = 0; j < number_of_iteration; j++) {
			init();
			multithread_E_step();
			calculate_M_step(j);
		}

		for(_Doc d:m_trainSet)
			estThetaInDoc(d);
	}

}
