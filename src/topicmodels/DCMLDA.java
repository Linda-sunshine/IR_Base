package topicmodels;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

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

			for (int j = 0; j < number_of_iteration; j++) {
				init();
				for (_Doc d : m_trainSet)
					calculate_E_step(d);
			}

			calculate_M_step(i, weightFolder);

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

	@Override
	protected int sampleTopic4Word(_Word w, _Doc d) {
		double p;
		int tid = 0;
		int wid = 0;

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
		return (d.m_sstat[tid] + m_alpha[tid])
				/ (d.getTotalDocLength() + m_totalAlpha - 1);
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
			m_sstat[k] += topicProb;
			for (int v = 0; v < vocabulary_size; v++) {
				m_docWordTopicProb[docID][k][v] += m_docWordTopicStats[docID][k][v]
						+ m_beta[k][v];
				topic_term_probabilty[k][v] += wordTopicProb(k, v, d);
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
	
		ArrayList<Double> t_tempList = new ArrayList<Double>();
		for (int k = 0; k < number_of_topics; k++) {
			if (m_sstat[k] > 0) {
				t_tempList.add ((m_sstat[k] - m_alphaAuxilary[k])
						/ (m_alphaAuxilary[k] - m_sstat[k] * m_sstat[k]));
			}
		}

		Collections.sort(t_tempList);

		double temp = t_tempList.get(t_tempList.size() / 2);
		for (int k = 0; k < number_of_topics; k++)
			m_alpha[k] = m_sstat[k] * temp;

		for (int k = 0; k < number_of_topics; k++) {
			t_tempList.clear();
			for (int v = 0; v < vocabulary_size; v++) {
				if(word_topic_sstat[k][v]>0)
					t_tempList
							.add((topic_term_probabilty[k][v] - word_topic_sstat[k][v])
									/ (word_topic_sstat[k][v] - topic_term_probabilty[k][v]
											* topic_term_probabilty[k][v]));
			}
			Collections.sort(t_tempList);
			
			int listLen = t_tempList.size();
			temp = 1;
			if (listLen > 0)
				temp = t_tempList.get(listLen / 2);
			for (int v = 0; v < vocabulary_size; v++)
				m_beta[k][v] = topic_term_probabilty[k][v] * temp;
		}

		m_totalAlpha = Utils.sumOfArray(m_alpha);
		for (int k = 0; k < number_of_topics; k++) {
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);
		}

	}

	protected void updateAlpha() {
		int i = 0;
		double alphaSum, diAlphaSum, c;
		double lambda = 0.1;
		double[] alphaG = new double[number_of_topics];
		double[] alphaQ = new double[number_of_topics];

		double deltaAlpha, diff;
		int docSize = m_trainSet.size();
		do {
			double b = 0;
			deltaAlpha = 0;
			diff = 0;
			Arrays.fill(alphaG, 0);
			Arrays.fill(alphaQ, 0);
			alphaSum = Utils.sumOfArray(m_alpha);
			diAlphaSum = Utils.digamma(alphaSum);

			c = docSize * Utils.trigamma(alphaSum);

			lambda /= 10;

			for (_Doc d : m_trainSet) {
				c -= Utils.trigamma(d.getTotalDocLength() + alphaSum);

				for (int k = 0; k < number_of_topics; k++) {
					alphaG[k] += Utils.digamma(d.m_sstat[k] + m_alpha[k])
							- Utils.digamma(m_alpha[k]);
					alphaG[k] += diAlphaSum
							- Utils.digamma(alphaSum + d.getTotalDocLength());
					alphaQ[k] += Utils.trigamma(d.m_sstat[k] + m_alpha[k])
							- Utils.trigamma(m_alpha[k]);
				}
			}
			
			boolean singularFlag = false;
			do {
			
				for (int k = 0; k < number_of_topics; k++) {
					alphaQ[k] -= docSize * lambda;
				}

				double b1 = 0, b2 = 0;
				for (int k = 0; k < number_of_topics; k++) {
					b1 += alphaG[k] / alphaQ[k];
					b2 += 1.0 / alphaQ[k];
				}
				b = b1 / ((1 / c) + b2);

				for (int k = 0; k < number_of_topics; k++) {
					if (m_alpha[k] == 0)
						continue;
					deltaAlpha = (alphaG[k] - b) / alphaQ[k];
					if (deltaAlpha > m_alpha[k]) {
						singularFlag = true;
						break;
					}

					if (k == number_of_topics)
						singularFlag = false;
				}

				lambda *= 10;

				if (lambda > 1e6)
					break;
			} while (singularFlag);
			
			if (lambda > 1e6)
				break;

			for (int k = 0; k < number_of_topics; k++) {
				deltaAlpha = (alphaG[k] - b) / alphaQ[k];
				m_alpha[k] -= deltaAlpha;
				diff += deltaAlpha * deltaAlpha;
			}
			
			diff /= number_of_topics;

		} while (++i < m_newtonIter && diff > m_newtonConverge);

		System.out.println("iteration\t" + i);
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

					deltaBeta = totalBetaNumerator[v] / totalBetaDenominator;
				} else {
					deltaBeta = 0;
				}
				
				double newBeta = m_beta[tid][v] * deltaBeta;
				if ((m_beta[tid][v] - newBeta) > diff)
					diff = m_beta[tid][v] - newBeta;

				m_beta[tid][v] = newBeta;

			}

			iteration++;

			System.out.println("beta iteration\t" + iteration);
		} while (diff > m_newtonConverge);

		System.out.println("iteration\t" + iteration);

		// for (int v = 0; v < vocabulary_size; v++) {
		// if ((m_beta[tid][v] != 0) && (m_beta[tid][v] < smoothingBeta)) {
		// smoothingBeta = m_beta[tid][v];
		// }
		// }



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

	protected void finalEst() {
		for (_Doc d : m_trainSet) {
			int docID = d.getID();
			for (int i = 0; i < number_of_topics; i++)
				Utils.L1Normalization(m_docWordTopicProb[docID][i]);
			estThetaInDoc(d);
		}

	}

	public void printTopWords(int k, String topWordPath) {
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

}
