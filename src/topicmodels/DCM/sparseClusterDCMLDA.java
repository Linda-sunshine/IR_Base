package topicmodels.DCM;

import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._Doc4SparseDCMLDA;
import structures._Word;
import utils.Utils;

public class sparseClusterDCMLDA extends sparseDCMLDA {
	public double[][][] m_clusterTopicWordStats;
	public double[][][] m_clusterTopicWordProb;

	public double[][] m_clusterTopicStats;
	public double[][] m_clusterTopicProb;

	public int m_clusterNum;
	public double m_gamma;
	public double[] m_clusterStats;
	public double[] m_clusterProb;

	public double[] m_clusterSamplingCache;

	public sparseClusterDCMLDA(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, int newtonIter, double newtonConverge,
			double tParam, double sParam, int clusterNum, double gammaParam) {
		// TODO Auto-generated constructor stub
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, newtonIter,
				newtonConverge, tParam, sParam);
		m_clusterNum = clusterNum;
		m_gamma = gammaParam;
		m_mu = 1;
	}

	protected void initialize_probability(Collection<_Doc> collection) {
		m_clusterStats = new double[m_clusterNum];
		m_clusterProb = new double[m_clusterNum];
		m_clusterSamplingCache = new double[m_clusterNum];

		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];

		m_totalAlpha = 0;
		m_totalBeta = new double[number_of_topics];

		m_alphaAuxilary = new double[number_of_topics];

		m_clusterTopicWordProb = new double[m_clusterNum][number_of_topics][vocabulary_size];
		m_clusterTopicWordStats = new double[m_clusterNum][number_of_topics][vocabulary_size];

		Arrays.fill(m_clusterStats, 0);
		Arrays.fill(m_clusterProb, 0);
		Arrays.fill(m_clusterSamplingCache, 0);

		m_clusterTopicProb = new double[m_clusterNum][number_of_topics];
		m_clusterTopicStats = new double[m_clusterNum][number_of_topics];

		for (int c = 0; c < m_clusterNum; c++) {
			Arrays.fill(m_clusterTopicStats[c], 0);
			Arrays.fill(m_clusterTopicProb[c], 0);
			for (int k = 0; k < number_of_topics; k++) {
				Arrays.fill(m_clusterTopicWordStats[c][k], 0);
				Arrays.fill(m_clusterTopicWordProb[c][k], 0);
			}
		}

		initialAlphaBeta();

		for (_Doc d : collection) {
			((_Doc4SparseDCMLDA) d).setTopics4GibbsCluster(number_of_topics, m_alpha, m_clusterNum, vocabulary_size);
			int cID = ((_Doc4SparseDCMLDA) d).m_clusterIndicator;
			m_clusterStats[cID]++;
			for (_Word w : d.getWords()) {
				int tid = w.getTopic();
				int wid = w.getIndex();
				m_clusterTopicWordStats[cID][tid][wid]++;
				m_clusterTopicStats[cID][tid]++;
				word_topic_sstat[tid][wid]++;
			}
		}

		imposePrior();
	}

	public double calculate_E_step(_Doc d) {
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;

		DCMDoc.permutation();

		sampleTopicAssignment(DCMDoc);
		sampleOnOffIndicator(DCMDoc);
		sampleClusterIndex(DCMDoc);

		return 0;
	}

	public void sampleClusterIndex(_Doc d) {
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;
		double p = 0;
		Arrays.fill(m_clusterSamplingCache, 0);
		int clusterIndex = DCMDoc.m_clusterIndicator;
		m_clusterStats[clusterIndex]--;
		for (_Word w : DCMDoc.getWords()) {
			int wid = w.getIndex();
			int tid = w.getTopic();
			m_clusterTopicWordStats[clusterIndex][tid][wid]--;
			m_clusterTopicStats[clusterIndex][tid]--;
		}

		// m_clusterSamplingCache[0] = 1;
		p += 1;
		
		for (int c = 0; c < m_clusterNum; c++) {
			double term1 = wordByClusterProb(d, c);
			term1 = clusterProb(c);

			m_clusterSamplingCache[c] = wordByClusterProb(d, c) * clusterProb(c);
			p += m_clusterSamplingCache[c];
		}

		p *= m_rand.nextDouble();
		for (clusterIndex = 0; clusterIndex < m_clusterNum; clusterIndex++) {
			p -= m_clusterSamplingCache[clusterIndex];
			if (p <= 0) {
				break;
			}
		}

		if (clusterIndex >= m_clusterNum)
			System.out.println("p\t" + p);
		DCMDoc.m_clusterIndicator = clusterIndex;
		m_clusterStats[clusterIndex]++;
		for (_Word w : DCMDoc.getWords()) {
			int wid = w.getIndex();
			int tid = w.getTopic();
			m_clusterTopicWordStats[clusterIndex][tid][wid]++;
			m_clusterTopicStats[clusterIndex][tid]++;
		}
	}

	//
	// protected double wordByClusterProb(_Doc d, int clusterIndex) {
	// _Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;
	//
	// double wordClusterProb = 1;
	// double term0 = 0;
	// double term00 = 0;
	// double termc = 0;
	// double termc0 = 0;
	//
	// double product = 1;
	// int iter = 0;
	// for (int k = 0; k < number_of_topics; k++) {
	// term0 = 0;
	// product = 1;
	//
	// for (int v = 0; v < vocabulary_size; v++) {
	// if (DCMDoc.m_wordTopic_stat[k][v] == 0)
	// continue;
	// term00 = m_beta[k][v] + m_clusterTopicWordStats[0][k][v];
	// termc0 = m_beta[k][v] + m_clusterTopicWordStats[clusterIndex][k][v];
	// if (term00 == termc0) {
	// continue;
	// }
	// iter += 1;
	// product *= gammaRatio(termc0, term00,
	// DCMDoc.m_wordTopic_stat[k][v]);
	// }
	// if (DCMDoc.m_sstat[k] == 0)
	// continue;
	// term0 = m_totalBeta[k] + m_clusterTopicStats[0][k];
	// termc = m_totalBeta[k] + m_clusterTopicStats[clusterIndex][k];
	// if (term0 == termc) {
	// continue;
	// }
	// wordClusterProb *= (gammaRatio(term0, termc, DCMDoc.m_sstat[k]) *
	// product);
	// }
	//
	// System.out.println("iter\t" + iter);
	// return wordClusterProb;
	// }

	protected double wordByClusterProb(_Doc d, int clusterIndex) {
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;

		double wordClusterProb = 0;
		double term0 = 0;
		double term00 = 0;
		double termc = 0;
		double termc0 = 0;

		double product = 0;
		int iter = 0;
		for (int k = 0; k < number_of_topics; k++) {
			term0 = 0;
			product = 0;

			for (int v = 0; v < vocabulary_size; v++) {
				if (DCMDoc.m_wordTopic_stat[k][v] == 0)
					continue;
				// term00 = m_beta[k][v] + m_clusterTopicWordStats[0][k][v];
				termc0 = m_beta[k][v]
						+ m_clusterTopicWordStats[clusterIndex][k][v];

				iter += 1;
				product += Math.log(gammaRatio(termc0,
						DCMDoc.m_wordTopic_stat[k][v]));
			}
			if (DCMDoc.m_sstat[k] == 0)
				continue;
			termc = m_totalBeta[k] + m_clusterTopicStats[clusterIndex][k];

			wordClusterProb += product
					- Math.log(gammaRatio(termc, DCMDoc.m_sstat[k]));
		}

		System.out.println("iter\t" + iter);
		return wordClusterProb;
	}

	protected double gammaRatio(double cluster0Start, double clusterCStart, double iteration) {
		double ratio = 1;
		
		while(iteration>0){
			ratio *= (cluster0Start+iteration-1)/(clusterCStart+iteration-1);
			iteration -= 1;
		}
		
		return ratio;
	}
	
	protected double gammaRatio(double cluster0Start, double iteration) {
		double ratio = 1;

		while (iteration > 0) {
			ratio *= (cluster0Start + iteration - 1);
			iteration -= 1;
		}

		return ratio;
	}

	protected double clusterProb(int clusterIndex) {
		double clusterProb = 0;
		clusterProb = (m_gamma + m_clusterStats[clusterIndex])/(m_gamma + m_clusterStats[0]);
		return clusterProb;
	}

	protected void sampleTopicAssignment(_Doc4SparseDCMLDA DCMDoc) {
		int wid, tid;
		double p;

		int clusterIndex = DCMDoc.m_clusterIndicator;
		for (_Word w : DCMDoc.getWords()) {
			wid = w.getIndex();
			tid = w.getTopic();

			DCMDoc.m_sstat[tid]--;
			DCMDoc.m_wordTopic_stat[tid][wid]--;

			if (m_collectCorpusStats) {
				word_topic_sstat[tid][wid]--;
				m_clusterTopicStats[clusterIndex][tid]--;
				m_clusterTopicWordStats[clusterIndex][tid][wid]--;
			}

			p = 0;

			double denominator = 0;
			denominator += DCMDoc.m_alphaDoc;
			denominator += Utils.sumOfArray(DCMDoc.m_sstat);

			for (tid = 0; tid < number_of_topics; tid++) {
				m_topicProbCache[tid] = 0;
				if (DCMDoc.m_topicIndicator[tid] == false)
					continue;
				double term1 = 0;
				term1 = topicInDocProb(tid, denominator, DCMDoc);
				if (term1 < 0) {
					System.out.println("negative1\t" + term1);
				}
				term1 = wordTopicProb(tid, wid, clusterIndex);
				if (term1 < 0) {
					System.out.println("negative2\t" + term1);
				}

				m_topicProbCache[tid] = topicInDocProb(tid, denominator, DCMDoc)
						* wordTopicProb(tid, wid, clusterIndex);

				p += m_topicProbCache[tid];
			}

			p *= m_rand.nextDouble();
			tid = -1;
			if (p <= 0) {
				// for(int k=0; k<number_of_topics; k++)
				// System.out.println(m_alpha[k]+"\t"+m_totalBeta[k]);
				System.out.println(p + "\t" + DCMDoc.getName() + "\t" + DCMDoc.m_indicatorTrue_stat);
			}
			while (p > 0 && tid < number_of_topics - 1) {
				tid++;
				p -= m_topicProbCache[tid];
			}

			w.setTopic(tid);
			DCMDoc.m_sstat[tid]++;
			DCMDoc.m_wordTopic_stat[tid][wid]++;

			if (m_collectCorpusStats) {
				word_topic_sstat[tid][wid]++;
				m_clusterTopicStats[clusterIndex][tid]++;
				m_clusterTopicWordStats[clusterIndex][tid][wid]++;
			}
		}

	}

	protected double wordTopicProb(int tid, int wid, int clusterID) {
		return (m_clusterTopicWordStats[clusterID][tid][wid] + m_mu * m_beta[tid][wid])
				/ (m_clusterTopicStats[clusterID][tid] + m_mu * m_totalBeta[tid]);
	}

	protected void updateBeta(int tid) {
		double diff = 0;
		int iteration = 0;

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

			for (int c = 0; c < m_clusterNum; c++) {
				for (int i = 0; i < m_clusterTopicStats[c][tid]; i++) {
					totalBetaDenominator += 1.0 / (i + m_totalBeta[tid]);
				}

				for (int v = 0; v < vocabulary_size; v++) {
					wordNum4Tid += m_clusterTopicWordStats[c][tid][v];
					wordNum4Tid4V[v] += m_clusterTopicWordStats[c][tid][v];
					for (int i = 0; i < m_clusterTopicWordStats[c][tid][v]; i++)
						totalBetaNumerator[v] += 1.0 / (i + m_beta[tid][v]);
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

	protected void estGlobalParameter() {
		Utils.L1Normalization(m_clusterProb);

		for (int k = 0; k < number_of_topics; k++) {
			Utils.L1Normalization(topic_term_probabilty[k]);
			for (int c = 0; c < m_clusterNum; c++) {
				Utils.L1Normalization(m_clusterTopicWordProb[c][k]);
			}
		}

		for (int c = 0; c < m_clusterNum; c++) {
			Utils.L1Normalization(m_clusterTopicProb[c]);
		}
	}

	protected void collectStats() {
		for (int k = 0; k < number_of_topics; k++)
			for (int v = 0; v < vocabulary_size; v++) {
				topic_term_probabilty[k][v] = word_topic_sstat[k][v] + m_mu * m_beta[k][v];
				for (int c = 0; c < m_clusterNum; c++) {
					m_clusterTopicProb[c][k] = m_clusterTopicStats[c][k] + m_beta[k][v];
					m_clusterTopicWordProb[c][k][v] = m_clusterTopicWordStats[c][k][v] + m_beta[k][v];
				}
			}

		for (_Doc d : m_trainSet) {
			_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;

			for (int k = 0; k < this.number_of_topics; k++) {
				if (DCMDoc.m_topicIndicator[k] == false)
					continue;
				DCMDoc.m_topics[k] = DCMDoc.m_sstat[k] + m_alpha[k];
			}
		}

		for (int c = 0; c < m_clusterNum; c++) {
			m_clusterProb[c] = m_clusterStats[c] + m_gamma;
		}

	}

	protected void estThetaInDoc(_Doc d) {

		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;
		Utils.L1Normalization(d.m_topics);

		DCMDoc.m_topicIndicator_distribution /= DCMDoc.m_MStepIter * number_of_topics;
		for (int k = 0; k < number_of_topics; k++)
			DCMDoc.m_topicIndicator_prob[k] /= DCMDoc.m_MStepIter;
	}
	
	protected double calculate_log_likelihood() {
		double logLikelihood = 0.0;
		for (_Doc d : m_trainSet) {
			logLikelihood += calculate_log_likelihood(d);
		}

		for (int c = 0; c < m_clusterNum; c++) {
			for (int k = 0; k < number_of_topics; k++) {
				for (int v = 0; v < vocabulary_size; v++) {
					double term = Utils.lgamma(m_clusterTopicWordStats[c][k][v]
							+ m_mu * m_beta[k][v]);
					logLikelihood += term;

					term = Utils.lgamma(m_mu * m_beta[k][v]);
					logLikelihood -= term;

				}
				logLikelihood += Utils.lgamma(m_mu * m_totalBeta[k]);
				logLikelihood -= Utils.lgamma(m_clusterTopicStats[c][k]
						+ m_mu*m_totalBeta[k]);
			}
		}

		return logLikelihood;
	}

	protected double calculate_log_likelihood(_Doc d) {
		double docLogLikelihood = 0.0;
		
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;

		for (int k = 0; k < number_of_topics; k++) {
			if(DCMDoc.m_topicIndicator[k]==false)
				continue;
			double term = Utils.lgamma(DCMDoc.m_sstat[k] + m_alpha[k]);
			docLogLikelihood += term;

			term = Utils.lgamma(m_alpha[k]);
			docLogLikelihood -= term;

		}

		docLogLikelihood += Utils.lgamma(DCMDoc.m_alphaDoc);
		docLogLikelihood -= Utils.lgamma(DCMDoc.getTotalDocLength() + DCMDoc.m_alphaDoc);

		docLogLikelihood += Utils.lgamma(m_t+m_s)-Utils.lgamma(m_t)-Utils.lgamma(m_s);
		docLogLikelihood += Utils.lgamma(DCMDoc.m_indicatorTrue_stat+m_s)+Utils.lgamma(m_t+number_of_topics-DCMDoc.m_indicatorTrue_stat)-Utils.lgamma(m_t+m_s+number_of_topics);
		
		return docLogLikelihood;
	}
	
}
