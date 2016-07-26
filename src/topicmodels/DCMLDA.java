package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._Word;
import utils.Utils;

public class DCMLDA extends LDA_Gibbs {

	/*
	 * m_docWordTopicProb---D*V*K D: number of documents V: number of words K:
	 * number of topics
	 * 
	 * m_docWordTopicStats---D*V*K m_docTopicStats---D*K // this can be included
	 * in the d.m_sstat
	 */

	protected double[][][] m_docWordTopicProb;
	protected double[][][] m_docWordTopicStats;
	// double[][] m_docTopicStats;

	/**
	 * 
	 * m_alpha K m_beta K*V;
	 * 
	 */
	protected double[] m_alpha;
	protected double[][] m_beta;

	protected double[] m_alphaAuxilary;

	protected double m_totalAlpha;
	protected double[] m_totalBeta;

	protected int m_newtonIter;
	protected double m_newtonConverge;

	public DCMLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha,
			double burnIn, int lag, int newtonIter, double newtonConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag);

		int corpusSize = c.getSize();
		
		m_docWordTopicProb = new double[corpusSize][number_of_topics][vocabulary_size];
		m_docWordTopicStats = new double[corpusSize][number_of_topics][vocabulary_size];

		// m_docTopicStats = new double[corpusSize][number_of_topics];

		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];

		m_totalAlpha = 0;
		m_totalBeta = new double[number_of_topics];

		m_newtonIter = newtonIter;
		m_newtonConverge = newtonConverge;

		m_alphaAuxilary = new double[number_of_topics];

	}

	public void EM() {
		System.out.format("Starting %s...\n", toString());

		long starttime = System.currentTimeMillis();

		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);

		String filePrefix = "./data/results/DCM_LDA";
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
				for (_Doc d : m_trainSet)
					calculate_E_step(d);
			}
			long eEndTime = System.currentTimeMillis();

			System.out.println("per iteration e step time\t"
					+ (eEndTime - eStartTime));

			long mStartTime = System.currentTimeMillis();
			calculate_M_step(i, weightFolder);
			long mEndTime = System.currentTimeMillis();

			// System.out.println("per iteration m step time\t"
			// + (mEndTime - mStartTime));

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

	@Override
	protected void initialize_probability(Collection<_Doc> collection) {

		// initialize topic-word allocation, p(w|z)
		for (_Doc d : collection) {
			int docID = d.getID();
			for (int k = 0; k < number_of_topics; k++) {
				Arrays.fill(m_docWordTopicStats[docID][k], 0);
			}

			// allocate memory and randomize it
			d.setTopics4Gibbs(number_of_topics, 0);

			for (_Word w : d.getWords()) {
				int wid = w.getIndex();
				int tid = w.getTopic();
				m_docWordTopicStats[docID][tid][wid]++;

			}
		}

		initialAlphaBeta();

		imposePrior();
	}

	public double calculate_E_step(_Doc d) {
		d.permutation();
		double p;
		int wid, tid;
		int docID = d.getID();
		for(_Word w:d.getWords()) {
			wid = w.getIndex();
			tid = w.getTopic();
			
			//remove the word's topic assignment
			d.m_sstat[tid]--;
			m_docWordTopicStats[docID][tid][wid]--;
			
			//perform random sampling
			p = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double term1 = topicInDocProb(tid, d);
				term1 = wordTopicProb(tid, wid, d);
				m_topicProbCache[tid] = topicInDocProb(tid, d)
						* wordTopicProb(tid, wid, d);
				p += m_topicProbCache[tid];	
			}
			p *= m_rand.nextDouble();
			
			tid = -1;
			while(p>0 && tid<number_of_topics-1) {
				tid ++;
				p -= topicInDocProb(tid, d) * wordTopicProb(tid, wid, d);
			}
			
			//assign the selected topic to word
			w.setTopic(tid);
			d.m_sstat[tid]++;
			m_docWordTopicStats[docID][tid][wid]++;
		}
		
		return 0;
	}
	
	@Override
	protected int sampleTopic4Word(_Word w, _Doc d) {
		double p;
		int tid = w.getTopic();
		int wid = w.getIndex();

		p = 0;
		// p(z|d) * p(w|z)
		for (tid = 0; tid < number_of_topics; tid++) {
			double term1 = topicInDocProb(tid, d);
			term1 = wordTopicProb(tid, wid, d);
			m_topicProbCache[tid] = topicInDocProb(tid, d)
					* wordTopicProb(tid, wid, d);
			p += m_topicProbCache[tid];
		}

		p *= m_rand.nextDouble();

		tid = -1;
		while (p > 0 && tid < number_of_topics - 1) {
			tid++;
			p -= m_topicProbCache[tid];
		}

		return tid;

	};

	protected double topicInDocProb(int tid, _Doc d) {
		double term1 = d.m_sstat[tid];
		term1 = m_alpha[tid];

		return (d.m_sstat[tid] + m_alpha[tid]);
	}

	// p(w|z)
	protected double wordTopicProb(int tid, int wid, _Doc d) {
		int docID = d.getID();
	
		return (m_docWordTopicStats[docID][tid][wid] + m_beta[tid][wid])
				/ (d.m_sstat[tid] + m_totalBeta[tid]);
	}

	protected void updateStats(boolean preFlag, _Word w, _Doc d) {
		int docID = d.getID();
		int wid = w.getIndex();
		int tid = w.getTopic();

		if (!preFlag) {
			d.m_sstat[tid]++;
			m_docWordTopicStats[docID][tid][wid]++;

		} else {
			d.m_sstat[tid]--;
			m_docWordTopicStats[docID][tid][wid]--;

		}

	}

	public void calculate_M_step(int iter, File weightFolder) {

		for (_Doc d : m_trainSet)
			collectStats(d);

		File weightIterFolder = new File(weightFolder, "_" + iter);
		if (!weightIterFolder.exists()) {
			weightIterFolder.mkdir();
		}

		updateParameter(iter, weightIterFolder);

	}

	protected void collectStats(_Doc d) {
		int docID = d.getID();

		for (int k = 0; k < this.number_of_topics; k++) {
			double topicProb = topicInDocProb(k, d);
			d.m_topics[k] += d.m_sstat[k] + m_alpha[k];

			for (int v = 0; v < vocabulary_size; v++){
				m_docWordTopicProb[docID][k][v] += m_docWordTopicStats[docID][k][v] + m_beta[k][v];
			}
		}

	}

	protected void updateParameter(int iter, File weightIterFolder) {
		initialAlphaBeta();
		updateAlpha();

		for (int k = 0; k < number_of_topics; k++)
			updateBeta(k);

		for (int k = 0; k < number_of_topics; k++)
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);

		String fileName = iter + ".txt";
		saveParameter2File(weightIterFolder, fileName);

	}

	protected void initialAlphaBeta() {

		Arrays.fill(m_sstat, 0);
		Arrays.fill(m_alphaAuxilary, 0);
		for (int k = 0; k < number_of_topics; k++) {
			Arrays.fill(topic_term_probabilty[k], 0);
			Arrays.fill(word_topic_sstat[k], 0);
		}

		for (_Doc d : m_trainSet) {
			int docID = d.getID();
			for (int k = 0; k < number_of_topics; k++) {
				double tempProb = d.m_sstat[k] / d.getTotalDocLength();
				m_sstat[k] += tempProb;
				m_alphaAuxilary[k] += tempProb * tempProb;
				if (d.m_sstat[k] == 0)
					continue;
				for (int v = 0; v < vocabulary_size; v++) {
					tempProb = m_docWordTopicStats[docID][k][v]
							/ d.m_sstat[k];

					topic_term_probabilty[k][v] += tempProb;
					word_topic_sstat[k][v] += tempProb*tempProb;
				}
			}
		}

		int trainSetSize = m_trainSet.size();
		for (int k = 0; k < number_of_topics; k++) {
			m_sstat[k] /= trainSetSize;
			m_alphaAuxilary[k] /= trainSetSize;
			for (int v = 0; v < vocabulary_size; v++) {
				topic_term_probabilty[k][v] /= trainSetSize;
				word_topic_sstat[k][v] /= trainSetSize;
			}
		}
	
		 for (int k = 0; k < number_of_topics; k++){
			m_alpha[k] = m_sstat[k];
			for (int v = 0; v < vocabulary_size; v++) {
				m_beta[k][v] = topic_term_probabilty[k][v] + d_beta;
			}
		}
		
		m_totalAlpha = Utils.sumOfArray(m_alpha);
		for (int k = 0; k < number_of_topics; k++) {
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);
		}

	}

	protected void updateAlpha() {
		double diff = 0;
		double smallAlpha = 0.1;
		int iteration = 0;
		do {

			diff = 0;

			double totalAlphaDenominator = 0;
			m_totalAlpha = Utils.sumOfArray(m_alpha);
			double digAlpha = Utils.digamma(m_totalAlpha);

			double deltaAlpha = 0;

			for (_Doc d : m_trainSet) {
				totalAlphaDenominator += Utils.digamma(d.getTotalDocLength()
						+ m_totalAlpha)
						- digAlpha;
			}
			
			for (int k = 0; k < number_of_topics; k++) {
				double totalAlphaNumerator = 0;
				for (_Doc d : m_trainSet) {
					totalAlphaNumerator += Utils.digamma(m_alpha[k]
							+ d.m_sstat[k])
							- Utils.digamma(m_alpha[k]);
				}

				deltaAlpha = totalAlphaNumerator * 1.0
						/ totalAlphaDenominator;


				double newAlpha = m_alpha[k] * deltaAlpha;
				double t_diff = Math.abs(m_alpha[k] - newAlpha);
				if (t_diff > diff)
					diff = t_diff;

				m_alpha[k] = newAlpha;
			}
			
			iteration++;
	
			if(iteration > m_newtonIter)
				break;
			
		}while(diff>m_newtonConverge);

//		System.out.println("iteration\t" + iteration);
		m_totalAlpha = 0;
		for (int k = 0; k < number_of_topics; k++) {
			m_totalAlpha += m_alpha[k];
		}
	}

	protected void updateBeta(int tid) {

		double diff = 0;
		int iteration = 0;
		double smoothingBeta = 0.1;

		do {
			diff = 0;
			double deltaBeta = 0;
			double wordNum4Tid = 0;
			double[] wordNum4Tid4V = new double[vocabulary_size];
			double totalBetaDenominator = 0;
			double[] totalBetaNumerator = new double[vocabulary_size];
			Arrays.fill(totalBetaNumerator, 0);
			Arrays.fill(wordNum4Tid4V, 0);
			m_totalBeta[tid] = Utils.sumOfArray(m_beta[tid]);
			double digBeta4Tid = Utils.digamma(m_totalBeta[tid]);

			for (_Doc d : m_trainSet) {
				int docID = d.getID();
				totalBetaDenominator += Utils.digamma(m_totalBeta[tid]
						+ d.m_sstat[tid])
						- digBeta4Tid;
				for (int v = 0; v < vocabulary_size; v++) {
					wordNum4Tid += m_docWordTopicStats[docID][tid][v];
					wordNum4Tid4V[v] += m_docWordTopicStats[docID][tid][v];
					totalBetaNumerator[v] += Utils.digamma(m_beta[tid][v]
							+ m_docWordTopicStats[docID][tid][v]);
					totalBetaNumerator[v] -= Utils.digamma(m_beta[tid][v]);
				}
			}

			for (int v = 0; v < vocabulary_size; v++) {
				if (wordNum4Tid == 0)
					break;
				if (wordNum4Tid4V[v] == 0) {
					deltaBeta = 0;

				} else {
					deltaBeta = totalBetaNumerator[v] / totalBetaDenominator;

				}
				
				double newBeta = m_beta[tid][v] * deltaBeta + d_beta;

				double t_diff = Math.abs(m_beta[tid][v] - newBeta);
				if (t_diff > diff)
					diff = t_diff;

				m_beta[tid][v] = newBeta;

			}

			iteration++;

		} while (diff > m_newtonConverge);

//		System.out.println("iteration\t" + iteration);

	}

	protected void saveParameter2File(File fileFolder, String fileName) {
		try {
			File paramFile = new File(fileFolder, fileName);

			PrintWriter pw = new PrintWriter(paramFile);
			pw.println("alpha");
			for (int k = 0; k < number_of_topics; k++) {
				pw.print(m_alpha[k] + "\t");
			}

			pw.println("beta");
			for (int k = 0; k < number_of_topics; k++) {
				pw.print("topic" + k + "\t");
				for (int v = 0; v < vocabulary_size; v++) {
					pw.print(m_beta[k][v] + "\t");
				}
				pw.println();
			}

			pw.flush();
			pw.close();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}

	protected double calculate_log_likelihood() {
		double logLikelihood = 0.0;
		for (_Doc d : m_trainSet) {
			logLikelihood += calculate_log_likelihood(d);
		}
		return logLikelihood;
	}

	protected double calculate_log_likelihood(_Doc d) {
		double docLogLikelihood = 0.0;

		int docID = d.getID();

		for (int k = 0; k < number_of_topics; k++) {
			double term = Utils.lgamma(d.m_sstat[k] + m_alpha[k]);
			docLogLikelihood += term;

			term = Utils.lgamma(m_alpha[k]);
			docLogLikelihood -= term;

		}

		docLogLikelihood += Utils.lgamma(m_totalAlpha);
		docLogLikelihood -= Utils.lgamma(d.getTotalDocLength() + m_totalAlpha);

		for (int k = 0; k < number_of_topics; k++) {
			for (int v = 0; v < vocabulary_size; v++) {
				double term = Utils.lgamma(m_docWordTopicStats[docID][k][v]
						+ m_beta[k][v]);
				docLogLikelihood += term;

				term = Utils.lgamma(m_beta[k][v]);
				docLogLikelihood -= term;

			}
			docLogLikelihood += Utils.lgamma(m_totalBeta[k]);
			docLogLikelihood -= Utils.lgamma(d.m_sstat[k] + m_totalBeta[k]);
		}

		return docLogLikelihood;
	}

	protected void estThetaInDoc(_Doc d) {

		int docID = d.getID();
		for (int i = 0; i < number_of_topics; i++)
			Utils.L1Normalization(m_docWordTopicProb[docID][i]);
		Utils.L1Normalization(d.m_topics);

	}

	public void printTopWords(int k, String betaFile) {
		double logLikelihood = calculate_log_likelihood();
		System.out.format("final log likelihood %.3f\t", logLikelihood);
		
		String filePrefix = betaFile.replace("topWords.txt", "");
		debugOutput(filePrefix);

		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		printTopWords_liteVersion(k, betaFile);
	}
	
	protected void debugOutput(String filePrefix) {
		
		int topK = 10;
		File topicFolder = new File(filePrefix + "topicAssignment");

		if (!topicFolder.exists()) {
			System.out.println("creating directory" + topicFolder);
			topicFolder.mkdir();
		}

	
		File wordTopicDistributionFolder = new File(filePrefix
				+ "wordTopicDistribution");
		if (!wordTopicDistributionFolder.exists()) {
			System.out.println("creating word topic distribution folder\t"
					+ wordTopicDistributionFolder);
			wordTopicDistributionFolder.mkdir();
		}

		for (_Doc d : m_trainSet) {
			printParentTopicAssignment(d, topicFolder);
			printWordTopicDistribution(d, wordTopicDistributionFolder, topK);
		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";

		printParameter(parentParameterFile, childParameterFile, m_trainSet);


	}

	public void printTopWords_liteVersion(int k, String topWordPath) {
		System.out.println("TopWord FilePath:" + topWordPath);

		Arrays.fill(m_sstat, 0);
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < number_of_topics; i++)
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];
		}
		Utils.L1Normalization(m_sstat);

		try {
			PrintWriter topWordWriter = new PrintWriter(new File(topWordPath));

			for (int i = 0; i < m_beta.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							m_beta[i][j]));

				topWordWriter.format("Topic %d(%.5f):\t", i, m_sstat[i]);
				for (_RankItem it : fVector)
					topWordWriter.format("%s(%.5f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				topWordWriter.write("\n");
			}
			topWordWriter.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}

	protected void printParentTopicAssignment(_Doc d, File topicFolder) {
		String topicAssignmentFile = d.getName() + ".txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topicFolder,
					topicAssignmentFile));
			
			for (_Word w : d.getWords()) {
				int index = w.getIndex();
				int topic = w.getTopic();

				String featureName = m_corpus.getFeature(index);
				pw.print(featureName + ":" + topic + "\t");
			}
			
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	protected void printParameter(String parentParameterFile,
			String childParameterFile, ArrayList<_Doc> docList) {
		System.out.println("printing parameter");
		
		try{
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			
			for(_Doc d:docList){
				parentParaOut.print(d.getName()+"\t");
				parentParaOut.print("topicProportion\t");
				for(int k=0; k<number_of_topics; k++){
					parentParaOut.print(d.m_topics[k]+"\t");
				}
				
				parentParaOut.println();
			}
			
			parentParaOut.flush();
			parentParaOut.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	protected void printWordTopicDistribution(_Doc d,
			File wordTopicDistributionFolder, int k) {
		String wordTopicDistributionFile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(
					wordTopicDistributionFolder, wordTopicDistributionFile));

			int docID = d.getID();
			for (int i = 0; i < number_of_topics; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int v = 0; v < vocabulary_size; v++) {
					String featureName = m_corpus.getFeature(v);
					double wordProb = m_docWordTopicProb[docID][i][v];
					_RankItem ri = new _RankItem(featureName, wordProb);
					fVector.add(ri);
				}

				pw.format("Topic %d(%.5f):\t", i, d.m_topics[i]);
				for (_RankItem it : fVector)
					pw.format("%s(%.5f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				pw.write("\n");
			}

			pw.flush();
			pw.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public double inference(_Doc d) {
		double likelihood = 0;
		initTestDoc(d);
		
		int i = 0;
		do{
			calculate_E_step(d);
			if (i > m_burnIn && i % m_lag == 0) {
				collectStats(d);
			}
		}while(++i<number_of_iteration);
		
		estThetaInDoc(d);
		
		likelihood = calculate_log_likelihood4Perplexity(d);
		
		return likelihood;
	}
	
	public void initTestDoc(_Doc d) {
		int docID = d.getID();

		for (int k = 0; k < number_of_topics; k++) {
			Arrays.fill(m_docWordTopicStats[docID][k], 0);
		}

		d.setTopics4Gibbs(number_of_topics, 0);

		for (_Word w : d.getWords()) {
			int wid = w.getIndex();
			int tid = w.getTopic();
			m_docWordTopicStats[docID][tid][wid]++;

		}
	}

	protected double calculate_log_likelihood4Perplexity(_Doc d) {
		double likelihood = 0;
		
		int docID = d.getID();
		for (_Word w : d.getWords()) {
			int wid = w.getIndex();
			double wordLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				wordLikelihood += d.m_topics[k]
						* m_docWordTopicProb[docID][k][wid];
			}
			likelihood += Math.log(wordLikelihood);
		}
		
		return likelihood;
	}

}
