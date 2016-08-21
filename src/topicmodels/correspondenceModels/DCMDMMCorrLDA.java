package topicmodels.correspondenceModels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures._ChildDoc;
import structures._ChildDoc4DCMDMMCorrLDA;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4DCM;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

/****
 * comments of an article share a topic proportion
 * each comment has a single topic. all words in a comment are assigned the same topic
 * ****/

public class DCMDMMCorrLDA extends DCMCorrLDA{
	double d_alpha_c;
	
	public DCMDMMCorrLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha_a,
			double alpha_c, double burnIn, double ksi, double tau, int lag,
			int newtonIter, double newtonConverge){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha_a, alpha_c, burnIn, ksi, tau, lag, newtonIter, newtonConverge);

		d_alpha_c = alpha_c;
	}
	
	public String toString(){
		return String.format("DCMDMMCorrLDA[k:%d, alphaA:%.2f, beta:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta);
	}
	
	protected void initialize_probability(Collection<_Doc> collection) {
		m_alpha_c = new double[number_of_topics];
		m_alphaAuxilary = new double[number_of_topics];

		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];

		m_totalAlpha = 0;
		m_totalAlpha_c = 0;
		m_totalBeta = new double[number_of_topics];

		m_topic_word_prob = new double[number_of_topics][vocabulary_size];

		for (_Doc d : collection) {
			if (d instanceof _ParentDoc4DCM) {
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
				pDoc.setTopics4Gibbs(number_of_topics, 0, vocabulary_size);

				for (_Stn stnObj : d.getSentences()) {
					stnObj.setTopicsVct(number_of_topics);
				}

				for (_ChildDoc cDoc : pDoc.m_childDocs) {

					((_ChildDoc4DCMDMMCorrLDA) cDoc).setTopics4Gibbs(
							number_of_topics, 0);
					((_ChildDoc4DCMDMMCorrLDA) cDoc).createSparseVct4Infer();

					for (_Word w : cDoc.getWords()) {
						int wid = w.getIndex();
						int tid = w.getTopic();

						pDoc.m_wordTopic_stat[tid][wid]++;
						pDoc.m_topic_stat[tid]++;
					}
					computeMu4Doc(cDoc);

					// cDoc.createSparseVct4Infer();
				}
			}

		}

		initialAlphaBeta();
		imposePrior();
	}

	protected void initialAlphaBeta() {

		double parentDocNum = 0;
		double childDocNum = 0;

		Arrays.fill(m_sstat, 0);
		Arrays.fill(m_alphaAuxilary, 0);
		for (int k = 0; k < number_of_topics; k++) {
			Arrays.fill(topic_term_probabilty[k], 0);
			Arrays.fill(word_topic_sstat[k], 0);
		}

		for (_Doc d : m_trainSet) {
			if (d instanceof _ParentDoc4DCM) {
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
				for (int k = 0; k < number_of_topics; k++) {
					double tempProb = pDoc.m_sstat[k]
							/ pDoc.getTotalDocLength();
					m_sstat[k] += tempProb;

					if (pDoc.m_sstat[k] == 0)
						continue;
					for (int v = 0; v < vocabulary_size; v++) {
						tempProb = pDoc.m_wordTopic_stat[k][v]
								/ pDoc.m_topic_stat[k];
						topic_term_probabilty[k][v] += tempProb;
					}
				}
				parentDocNum += 1;
				childDocNum = pDoc.m_childDocs.size();
				
				for (_ChildDoc cDoc : pDoc.m_childDocs) {
					int tid = ((_ChildDoc4DCMDMMCorrLDA) cDoc).m_topic;
				
					m_alphaAuxilary[tid] += 1.0 / childDocNum;
					
				}
			}
		}

		for (int k = 0; k < number_of_topics; k++) {
			m_sstat[k] /= parentDocNum;
			m_alphaAuxilary[k] /= parentDocNum;
			for (int v = 0; v < vocabulary_size; v++) {
				topic_term_probabilty[k][v] /= (parentDocNum);
			}
		}

		for (int k = 0; k < number_of_topics; k++) {
			m_alpha[k] = m_sstat[k];
			m_alpha_c[k] = m_alphaAuxilary[k] + d_alpha_c;
			for (int v = 0; v < vocabulary_size; v++)
				m_beta[k][v] = topic_term_probabilty[k][v] + d_beta;
		}

		m_totalAlpha = Utils.sumOfArray(m_alpha);
		m_totalAlpha_c = Utils.sumOfArray(m_alpha_c);
		for (int k = 0; k < number_of_topics; k++) {
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);
		}

	}

	protected double parentChildInfluenceProb(int tid, _ParentDoc4DCM d){
		double term = 1.0;
		
		if(tid==0)
			return term;
		
		int tidNum = 0;
		int zeroNum = 0;
		
		double mu = 0;
		for(_ChildDoc cDoc:d.m_childDocs){
			mu = cDoc.getMu();
			if (((_ChildDoc4DCMDMMCorrLDA)cDoc).m_topic == tid)
				tidNum ++;
			if(((_ChildDoc4DCMDMMCorrLDA)cDoc).m_topic == 0)
				zeroNum ++;
		}
		
		double muDp = mu/d.getDocInferLength();
		term *= gammaFuncRatio(tidNum, muDp, m_alpha_c[tid] + d.m_sstat[tid]
				* muDp)
				/ gammaFuncRatio(zeroNum, muDp, m_alpha_c[0] + d.m_sstat[0]
						* muDp);
		
		return term;
	}

	protected void sampleInChildDoc(_ChildDoc d) {
		int wid, tid = 0;
		double normalizedProb;
		
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d.m_parentDoc;

		for (_Word w : d.getWords()) {
			tid = w.getTopic();
			wid = w.getIndex();
			
			pDoc.m_wordTopic_stat[tid][wid]--;
			pDoc.m_topic_stat[tid]--;

		}
		d.m_sstat[tid]--;
		
		normalizedProb = 0;
		
		for (tid = 0; tid < number_of_topics; tid++) {
			double pWordTopic = childWordByTopicProb(tid, d, pDoc);
			double pTopic = childTopicInDocProb(tid, d, pDoc);
			
			m_topicProbCache[tid] = pWordTopic * pTopic;
			normalizedProb += m_topicProbCache[tid];
		}

		normalizedProb *= m_rand.nextDouble();
		for (tid = 0; tid < m_topicProbCache.length; tid++) {
			normalizedProb -= m_topicProbCache[tid];
			if (normalizedProb <= 0)
				break;
		}

		if (tid == m_topicProbCache.length)
			tid--;
		
		for (_Word w : d.getWords()) {
			wid = w.getIndex();
			w.setTopic(tid);

			pDoc.m_wordTopic_stat[tid][wid]++;
			pDoc.m_topic_stat[tid]++;

		}

		d.m_sstat[tid]++;
		((_ChildDoc4DCMDMMCorrLDA) d).m_topic = tid;
	}
	
	protected double childWordByTopicProb(int tid, _ChildDoc d,
			_ParentDoc4DCM pDoc) {
		double prob = 0;
		
		double numerator = 1;

		for (_SparseFeature sf : d.getSparseVct4Infer()) {
			int wid = sf.getIndex();
			double value = sf.getValue();
			
			for (int v = 1; v <= value; v++)
				numerator *= (m_beta[tid][wid]
						+ pDoc.m_wordTopic_stat[tid][wid] + v - 1);
		}
		
		double denominator = 1;
		for (int i = 1; i <= d.getDocInferLength(); i++) {
			denominator *= (m_totalBeta[tid] + pDoc.m_topic_stat[tid] + i - 1);
		}

		prob = numerator * 1.0 / denominator;

		return prob;
	}
	
	protected double childTopicInDocProb(int tid, _ChildDoc d,
			_ParentDoc4DCM pDoc) {
		double prob = 0;
		double parentTopicSum = Utils.sumOfArray(pDoc.m_sstat);
		
		double tidNum = 0;
		double totalTidNum = 0;

		int oldTid = ((_ChildDoc4DCMDMMCorrLDA)d).m_topic;
		
		for (_ChildDoc cDoc : pDoc.m_childDocs) {
			if(((_ChildDoc4DCMDMMCorrLDA)cDoc).m_topic == tid)
				tidNum++;
			totalTidNum++;
		}
		
		if(oldTid==tid)
			tidNum --;
		totalTidNum --;
		
		double muDp = d.getMu() / parentTopicSum;
		prob = (m_alpha_c[tid] + muDp * pDoc.m_sstat[tid] + tidNum)
				/ (m_totalAlpha_c + muDp * parentTopicSum + totalTidNum);

		return prob;
	}
	
	public void updateAlphaC() {
		double diff = 0;
		int iteration = 0;
		
		do{
			diff = 0;
			double totalAlphaDenominator = 0;
			double[] tidNum = new double[number_of_topics];
			double[] totalAlphaNumerator = new double[number_of_topics];

			Arrays.fill(totalAlphaNumerator, 0);
			m_totalAlpha_c = Utils.sumOfArray(m_alpha_c);

			double deltaAlpha = 0;
			for (_Doc d : m_trainSet) {
				if (d instanceof _ParentDoc) {
					_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;

					Arrays.fill(tidNum, 0);
					double topicSum = 0;
					double mu = 0;
					for (_ChildDoc cDoc : pDoc.m_childDocs) {
						mu = cDoc.getMu();
						topicSum++;
						int tid = ((_ChildDoc4DCMDMMCorrLDA) cDoc).m_topic;
						tidNum[tid]++;
					}

					double pDocLen = pDoc.getDocInferLength();
					double muDp = mu / pDocLen;
					double t_totalAlpha_c = m_totalAlpha_c + mu;
					double digAlpha = Utils.digamma(t_totalAlpha_c);
					totalAlphaDenominator = Utils.digamma(topicSum
							+ t_totalAlpha_c)
							- digAlpha;

					for (int k = 0; k < number_of_topics; k++)
						totalAlphaNumerator[k] = Utils.digamma(m_alpha_c[k]
								+ muDp * pDoc.m_sstat[k] + tidNum[k])
								- Utils.digamma(m_alpha_c[k] + muDp
										* pDoc.m_sstat[k]);

				}
			}

			for (int k = 0; k < number_of_topics; k++) {
				deltaAlpha = totalAlphaNumerator[k] * 1.0
						/ totalAlphaDenominator;

				double newAlpha = m_alpha_c[k] * deltaAlpha + d_alpha_c;
				double t_diff = Math.abs(m_alpha_c[k] - newAlpha);
				if (t_diff > diff)
					diff = t_diff;

				m_alpha_c[k] = newAlpha;
			}

			iteration++;
			// System.out.println("alpha iteration\t" + iteration);
			
			if (iteration > m_newtonIter)
				break;
		} while (diff > m_newtonConverge);

		// System.out.println("iteration\t" + iteration);
		m_totalAlpha_c = 0;
		for (int k = 0; k < number_of_topics; k++) {
			m_totalAlpha_c += m_alpha_c[k];
		}
	}

	protected double calculate_log_likelihood(_ParentDoc4DCM d) {
		double docLogLikelihood = 0;
		int docID = d.getID();

		double parentDocLength = d.getTotalDocLength();

		for (int k = 0; k < number_of_topics; k++) {
			double term = Utils.lgamma(d.m_sstat[k] + m_alpha[k]);
			docLogLikelihood += term;

			term = Utils.lgamma(m_alpha[k]);
			docLogLikelihood -= term;
		}

		docLogLikelihood += Utils.lgamma(m_totalAlpha);
		docLogLikelihood -= Utils.lgamma(parentDocLength + m_totalAlpha);
		for (int k = 0; k < number_of_topics; k++) {
			for (int v = 0; v < vocabulary_size; v++) {
				double term = Utils.lgamma(d.m_wordTopic_stat[k][v]
						+ m_beta[k][v]);
				docLogLikelihood += term;

				term = Utils.lgamma(m_beta[k][v]);
				docLogLikelihood -= term;
			}

			docLogLikelihood += Utils.lgamma(m_totalBeta[k]);
			docLogLikelihood -= Utils
					.lgamma(d.m_topic_stat[k] + m_totalBeta[k]);
		}

		double[] tidNum = new double[number_of_topics];
		Arrays.fill(tidNum, 0);

		double topicSum = 0;
		double mu = 0;

		for (_ChildDoc cDoc : d.m_childDocs) {
			mu = cDoc.getMu();
			topicSum++;
			int tid = ((_ChildDoc4DCMDMMCorrLDA) cDoc).m_topic;
			tidNum[tid]++;
		}

		double muDp = mu / parentDocLength;
		docLogLikelihood += Utils.digamma(m_totalAlpha_c + mu);

		docLogLikelihood -= Utils.digamma(m_totalAlpha_c + mu + topicSum);

		for (int k = 0; k < number_of_topics; k++) {
			double term = m_alpha_c[k] + muDp * d.m_sstat[k] + tidNum[k];
			term = Utils
					.digamma(m_alpha_c[k] + muDp * d.m_sstat[k] + tidNum[k]);

			double term2 = m_alpha_c[k] + muDp * d.m_sstat[k];
			term2 = Utils.digamma(m_alpha_c[k] + muDp * d.m_sstat[k]);
			docLogLikelihood += term - term2;
		}

		return docLogLikelihood;
	}

	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d) {
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
		for (_Stn stnObj : pDoc.getSentences()) {
			stnObj.setTopicsVct(number_of_topics);
		}

		int testLength = 0;
		pDoc.setTopics4GibbsTest(number_of_topics, 0, testLength,
				vocabulary_size);
		pDoc.createSparseVct4Infer();

		sampleTestSet.add(pDoc);
		for (_ChildDoc cDoc : pDoc.m_childDocs) {
			testLength = (int) (m_testWord4PerplexityProportion * cDoc
					.getTotalDocLength());
			// testLength = 0;
			((_ChildDoc4DCMDMMCorrLDA) cDoc).setTopics4GibbsTest(
					number_of_topics, 0, testLength);

			for (_Word w : cDoc.getWords()) {
				int wid = w.getIndex();
				int tid = w.getTopic();

				pDoc.m_wordTopic_stat[tid][wid]++;
				pDoc.m_topic_stat[tid]++;
			}

			sampleTestSet.add(cDoc);
			cDoc.createSparseVct4Infer();
			computeTestMu4Doc(cDoc);
		}

	}

}
