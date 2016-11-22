package topicmodels.DCM;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._Doc4DCMLDA;
import structures._RankItem;
import structures._Word;
import topicmodels.LDA.LDA_Gibbs;
import utils.Utils;

public class DCMLDA extends LDA_Gibbs {

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
	protected int m_corpusSize;

	public DCMLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha,
			double burnIn, int lag, int newtonIter, double newtonConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag);

		m_corpusSize = c.getSize();
		m_newtonIter = newtonIter;
		m_newtonConverge = newtonConverge;

	}

	public void LoadPrior(String fileName, double eta) {
		if (fileName == null || fileName.isEmpty()) {
			return;
		}

		try {

			if (word_topic_prior == null) {
				word_topic_prior = new double[number_of_topics][vocabulary_size];
			}

			for (int k = 0; k < number_of_topics; k++)
				Arrays.fill(word_topic_prior[k], 0);

			String tmpTxt;
			String[] lineContainer;
			String[] featureContainer;
			int tid = 0;

			HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
			for (int i = 0; i < m_corpus.getFeatureSize(); i++) {
				featureNameIndex.put(m_corpus.getFeature(i),
						featureNameIndex.size());
			}

			BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(fileName), "UTF-8"));

			while ((tmpTxt = br.readLine()) != null) {
				tmpTxt = tmpTxt.trim();
				if (tmpTxt.isEmpty())
					continue;

				lineContainer = tmpTxt.split("\t");

				tid = Integer.parseInt(lineContainer[0]);
				for (int i = 1; i < lineContainer.length; i++) {
					featureContainer = lineContainer[i].split(":");

					String featureName = featureContainer[0];
					double featureProb = Double
							.parseDouble(featureContainer[1]);

					int featureIndex = featureNameIndex.get(featureName);

					word_topic_prior[tid][featureIndex] = featureProb;
				}
			}

			System.out.println("prior is added");
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	protected void imposePrior() {
		if (word_topic_prior != null) {
			Arrays.fill(m_totalBeta, 0);
			for (int k = 0; k < number_of_topics; k++) {
				for (int v = 0; v < vocabulary_size; v++) {
					m_beta[k][v] = word_topic_prior[k][v];
					m_totalBeta[k] += m_beta[k][v];
				}
			}
		}
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

			long startTime = System.currentTimeMillis();
			for (int j = 0; j < number_of_iteration; j++) {
				init();
//				System.out.println("iteration\t" + j);
				for (_Doc d : m_trainSet)
					calculate_E_step(d);
			}
			long endTime = System.currentTimeMillis();

			System.out.println("per iteration e step time\t"
					+ (endTime - startTime) / 1000);

			startTime = System.currentTimeMillis();
			updateParameter(i, weightFolder);
			endTime = System.currentTimeMillis();

			 System.out.println("per iteration m step time\t"
					+ (endTime - startTime) / 1000);

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

		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];

		m_totalAlpha = 0;
		m_totalBeta = new double[number_of_topics];

		m_alphaAuxilary = new double[number_of_topics];

		for (_Doc d : collection) {
			((_Doc4DCMLDA)d).setTopics4Gibbs(number_of_topics, 0, vocabulary_size);
			// allocate memory and randomize it
			// ((_ChildDoc) d).setTopics4Gibbs_LDA(number_of_topics, 0);
		}

		initialAlphaBeta();

		imposePrior();
	}

	public double calculate_E_step(_Doc d) {
		_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA)d;
		
		DCMDoc.permutation();
		double p;
		int wid, tid;
		for(_Word w:DCMDoc.getWords()) {
			wid = w.getIndex();
			tid = w.getTopic();
			
			//remove the word's topic assignment
			DCMDoc.m_sstat[tid]--;
			DCMDoc.m_wordTopic_stat[tid][wid]--;
			
			if(m_collectCorpusStats)
				word_topic_sstat[tid][wid] --;
			
			//perform random sampling
			p = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double term1 = topicInDocProb(tid, DCMDoc);
				term1 = wordTopicProb(tid, wid, DCMDoc);
				m_topicProbCache[tid] = topicInDocProb(tid, DCMDoc)
						* wordTopicProb(tid, wid, DCMDoc);
				p += m_topicProbCache[tid];	
			}
			p *= m_rand.nextDouble();
			
			tid = -1;
			while(p>0 && tid<number_of_topics-1) {
				tid ++;
				p -= m_topicProbCache[tid];
			}
			
			//assign the selected topic to word
			w.setTopic(tid);
			DCMDoc.m_sstat[tid]++;
			DCMDoc.m_wordTopic_stat[tid][wid]++;
			
			if(m_collectCorpusStats)
				word_topic_sstat[tid][wid]++;
		}
		
		return 0;
	}

	protected double topicInDocProb(int tid, _Doc d) {
		double term1 = d.m_sstat[tid];
		term1 = m_alpha[tid];

		return (d.m_sstat[tid] + m_alpha[tid]);
	}

	// p(w|z)
	protected double wordTopicProb(int tid, int wid, _Doc d) {
		_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA) d;
	
		return (DCMDoc.m_wordTopic_stat[tid][wid] + m_beta[tid][wid])
				/ (DCMDoc.m_sstat[tid] + m_totalBeta[tid]);
	}

	public void calculate_M_step(int iter) {

		for (_Doc d : m_trainSet)
			collectStats(d);
		
		for(int k=0; k<number_of_topics; k++)
			for(int v=0; v<vocabulary_size; v++)
				topic_term_probabilty[k][v] += word_topic_sstat[k][v]+m_beta[k][v];

	}

	protected void collectStats(_Doc d) {
		_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA)d;
		
		for (int k = 0; k < this.number_of_topics; k++) {
			d.m_topics[k] += d.m_sstat[k] + m_alpha[k];

			for (int v = 0; v < vocabulary_size; v++){
				DCMDoc.m_wordTopic_prob[k][v] += DCMDoc.m_wordTopic_stat[k][v]+m_beta[k][v];
			}
		}

	}

	protected void updateParameter(int iter, File weightFolder) {
		
		File weightIterFolder = new File(weightFolder, "_" + iter);
		if (!weightIterFolder.exists()) {
			weightIterFolder.mkdir();
		}
		
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
		}

		for (_Doc d : m_trainSet) {
			_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA)d;
			for (int k = 0; k < number_of_topics; k++) {
				double tempProb = d.m_sstat[k] / d.getTotalDocLength();
				m_sstat[k] += tempProb;
				m_alphaAuxilary[k] += tempProb * tempProb;
				if (DCMDoc.m_sstat[k] == 0) continue;
				for (int v = 0; v < vocabulary_size; v++) {
					tempProb = DCMDoc.m_wordTopic_stat[k][v]
							/ DCMDoc.m_sstat[k];

					topic_term_probabilty[k][v] += tempProb;
				
				}
			}
		}

		int trainSetSize = m_trainSet.size();
		for (int k = 0; k < number_of_topics; k++) {
			m_sstat[k] /= trainSetSize;
			m_alphaAuxilary[k] /= trainSetSize;
			for (int v = 0; v < vocabulary_size; v++) {
				topic_term_probabilty[k][v] /= trainSetSize;
			}
		}
	
		 for (int k = 0; k < number_of_topics; k++){
			m_alpha[k] = m_sstat[k]+d_alpha;
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
			
		} while (diff > m_newtonConverge);

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
				_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA)d;
				totalBetaDenominator += Utils.digamma(m_totalBeta[tid]
						+ DCMDoc.m_sstat[tid])
						- digBeta4Tid;
				for (int v = 0; v < vocabulary_size; v++) {
					wordNum4Tid += DCMDoc.m_wordTopic_stat[tid][v];
					wordNum4Tid4V[v] += DCMDoc.m_wordTopic_stat[tid][v];
					totalBetaNumerator[v] += Utils.digamma(m_beta[tid][v]
							+ DCMDoc.m_wordTopic_stat[tid][v]);
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

		} while ((diff > m_newtonConverge) && (iteration < m_newtonIter));

		System.out.println("iteration\t" + iteration);

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
		
		_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA) d;

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
				double term = Utils.lgamma(DCMDoc.m_wordTopic_stat[k][v]
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

	protected void finalEst(){
		for (int j = 0; j < number_of_iteration; j++) {
			init();
			for (_Doc d : m_trainSet)
				calculate_E_step(d);
			calculate_M_step(j);
		}
		
		for(int i=0; i<number_of_topics; i++)
			Utils.L1Normalization(topic_term_probabilty[i]);
		
		for(_Doc d:m_trainSet)
			estThetaInDoc(d);
	}
	
	protected void estThetaInDoc(_Doc d) {
		
		_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA)d;
		
		for (int i = 0; i < number_of_topics; i++)
			Utils.L1Normalization(DCMDoc.m_wordTopic_prob[i]);
		Utils.L1Normalization(d.m_topics);

	}

	public void printTopWords(int k, String betaFile) {
		double logLikelihood = calculate_log_likelihood();
		System.out.format("final log likelihood %.3f\t", logLikelihood);

		System.out.println("TopWord FilePath:" + betaFile);

		Arrays.fill(m_sstat, 0);
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < number_of_topics; i++)
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];
		}
		Utils.L1Normalization(m_sstat);

		try {
			PrintWriter topWordWriter = new PrintWriter(new File(betaFile));

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

		likelihood = cal_logLikelihood4Partial(d);
		
		return likelihood;
	}
	
	public void initTestDoc(_Doc d) {

		_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA)d;
		
		for (int k = 0; k < number_of_topics; k++) {
			Arrays.fill(DCMDoc.m_wordTopic_prob[k], 0);
		}

		int testLength = (int) (m_testWord4PerplexityProportion * DCMDoc
				.getTotalDocLength());
		DCMDoc.setTopics4GibbsTest(number_of_topics, 0, testLength);

		for (_Word w : DCMDoc.getWords()) {
			int wid = w.getIndex();
			int tid = w.getTopic();
			DCMDoc.m_wordTopic_stat[tid][wid]++;

		}
	}

	protected double cal_logLikelihood4Partial(_Doc d) {
		double docLogLikelihood = 0;
		
		_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA)d;
		
		for (_Word w : DCMDoc.getTestWords()) {
			int wid = w.getIndex();
			double wordLogLikelihood = 0;

			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = DCMDoc.m_topics[k]*DCMDoc.m_wordTopic_prob[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			docLogLikelihood += Math.log(wordLogLikelihood);
		}

		return docLogLikelihood;
	}
	
	protected double calculate_log_likelihood4Perplexity(_Doc d) {
		double likelihood = 0;
		
		_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA)d;
		
		for (_Word w : DCMDoc.getWords()) {
			int wid = w.getIndex();
			double wordLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				wordLikelihood += DCMDoc.m_topics[k]*DCMDoc.m_wordTopic_prob[k][wid];
			}
			likelihood += Math.log(wordLikelihood);
		}
		
		return likelihood;
	}


}
