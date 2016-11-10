package topicmodels.DCM;

import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._Doc4DCMLDA;
import structures._Doc4SparseDCMLDA;
import structures._Word;
import utils.Utils;

public class sparseDCMLDA extends DCMLDA{
	
	public double m_t, m_s, m_mu;
	
	public sparseDCMLDA(int number_of_iteration, double converge, double beta, _Corpus c, 
			double lambda, int number_of_topics, double alpha, double burnIn, int lag, int newtonIter, double newtonConverge, double tParam, double sParam){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, newtonIter, newtonConverge);
		
		m_t = tParam;
		m_s = sParam;
		m_mu = 1;
		
		m_corpusSize = c.getSize();
		m_newtonIter = newtonIter;
		m_newtonConverge = newtonConverge;
	}
	
	protected void initialize_probability(Collection<_Doc> collection) {

		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];

		m_totalAlpha = 0;
		m_totalBeta = new double[number_of_topics];

		m_alphaAuxilary = new double[number_of_topics];
		
		initialAlphaBeta();

		for (_Doc d : collection) {
			((_Doc4SparseDCMLDA)d).setTopics4Gibbs(number_of_topics, m_alpha, vocabulary_size);
		}

		imposePrior();
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

		for (int k = 0; k < number_of_topics; k++) {
			for (int v = 0; v < vocabulary_size; v++) {
				double term = Utils.lgamma(DCMDoc.m_wordTopic_stat[k][v]
 + m_mu
						* m_beta[k][v]);
				docLogLikelihood += term;

				term = Utils.lgamma(m_mu * m_beta[k][v]);
				docLogLikelihood -= term;

			}
			docLogLikelihood += Utils.lgamma(m_mu * m_totalBeta[k]);
			docLogLikelihood -= Utils.lgamma(DCMDoc.m_sstat[k] + m_mu
					* m_totalBeta[k]);
		}

		docLogLikelihood += Utils.lgamma(m_t+m_s)-Utils.lgamma(m_t)-Utils.lgamma(m_s);
		docLogLikelihood += Utils.lgamma(DCMDoc.m_indicatorTrue_stat+m_s)+Utils.lgamma(m_t+number_of_topics-DCMDoc.m_indicatorTrue_stat)-Utils.lgamma(m_t+m_s+number_of_topics);
		
		return docLogLikelihood;
	}
	
	protected void initialAlphaBeta() {

		Arrays.fill(m_sstat, 0);
		Arrays.fill(m_alphaAuxilary, 0);
		for (int k = 0; k < number_of_topics; k++) {
			Arrays.fill(topic_term_probabilty[k], 0);
		}

		 for (int k = 0; k < number_of_topics; k++){
			m_alpha[k] = m_rand.nextDouble()+d_alpha;
			for (int v = 0; v < vocabulary_size; v++) {
				m_beta[k][v] = m_rand.nextDouble() + d_beta;
			}
		}
		
		m_totalAlpha = Utils.sumOfArray(m_alpha);
		for (int k = 0; k < number_of_topics; k++) {
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);
		}

	}
	
	public double calculate_E_step(_Doc d){
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;

		DCMDoc.permutation();
		
		sampleTopicAssignment(DCMDoc);
		sampleOnOffIndicator(DCMDoc);
		
		return 0;
	}
	
	protected void sampleTopicAssignment(_Doc4SparseDCMLDA DCMDoc){
		int wid, tid;
		double p;
		
		
		for(_Word w:DCMDoc.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			
			DCMDoc.m_sstat[tid] --;
			DCMDoc.m_wordTopic_stat[tid][wid]--;
			
			if(m_collectCorpusStats)
				word_topic_sstat[tid][wid] --;
			
			p = 0;
			
//			double totalAlphaDoc = 0;
//			for(int i=0; i<number_of_topics; i++){
//				if(DCMDoc.m_topicIndicator[i]==true)
//					totalAlphaDoc += m_alpha[i];
//			}
			
			
			double denominator = 0;
			denominator += DCMDoc.m_alphaDoc;
			denominator += Utils.sumOfArray(DCMDoc.m_sstat);
			
			for(tid=0; tid<number_of_topics; tid++){
				m_topicProbCache[tid] = 0;
				if(DCMDoc.m_topicIndicator[tid]==false)
					continue;
				double term1 = 0;
				term1 = topicInDocProb(tid, denominator, DCMDoc);
				term1 = wordTopicProb(tid, wid, DCMDoc);

				m_topicProbCache[tid] = topicInDocProb(tid, denominator, DCMDoc)
						* wordTopicProb(tid, wid, DCMDoc);
				if(m_topicProbCache[tid]<0)
					System.out.println("negative\t"+m_topicProbCache[tid]);
				p += m_topicProbCache[tid];
			}
			
			p *= m_rand.nextDouble();
			tid = -1;
			while(p>0 && tid<number_of_topics-1){
				tid ++;
				p -= m_topicProbCache[tid];
			}
			
			w.setTopic(tid);
			DCMDoc.m_sstat[tid] ++;
			DCMDoc.m_wordTopic_stat[tid][wid] ++;
			
			if(m_collectCorpusStats)
				word_topic_sstat[tid][wid] ++;
		}
		
	}
	
	protected void sampleOnOffIndicator(_Doc4SparseDCMLDA DCMDoc){
		for(int k=0; k<number_of_topics; k++){
			boolean xk = DCMDoc.m_topicIndicator[k];
			if(xk==true){
				DCMDoc.m_indicatorTrue_stat --;
				DCMDoc.m_alphaDoc -= m_alpha[k];
			}
			
			if(DCMDoc.m_sstat[k]>0){
				xk = true;
			}else{
				double prob = 0;
				
				double trueProb = 0;
				double falseProb = 0;
				double term1 = DCMDoc.m_alphaDoc;
				double term2 = m_alpha[k];
				double term3 = m_s + DCMDoc.m_indicatorTrue_stat;
				double term4 = m_t + number_of_topics-1
						- DCMDoc.m_indicatorTrue_stat;
				//double term1 = DCMDoc.m_alphaDoc+m_alpha[k], DCMDoc.m_alphaDoc+m_alpha[k]+DCMDoc.getTotalDocLength());
				//double term2 = (m_s+DCMDoc.m_indicatorTrue_stat);
				double Q = term3 / term4;
				for (int i = 0; i < DCMDoc.getTotalDocLength(); i++) {
					double QTemp = (term1 + i) / (term1 + term2 + i);
					Q *= QTemp;
				}

				falseProb = 1.0/(Q+1);
				trueProb = 1-falseProb;

				prob = m_rand.nextDouble()*(trueProb+falseProb);
				if(prob<trueProb)
					xk = true;
				else
					xk = false;
				
			}
			
			DCMDoc.m_topicIndicator[k] = xk;
			if(xk==true){
				DCMDoc.m_indicatorTrue_stat++;	
				DCMDoc.m_alphaDoc += m_alpha[k];
			}
					
		}
	}
	
	protected double gammaRatio(double nominator, double denominator){
		double ratio = 1;
		
		double initialVal = denominator;
		while(initialVal>nominator){
			ratio *= (initialVal-1);
			initialVal -= 1;
		}
		
		return ratio;
	}
	
	protected double topicInDocProb(int tid, double denominator, _Doc4SparseDCMLDA d){
		double term1 = d.m_sstat[tid];
		term1 += m_alpha[tid];
		
		return term1/denominator;
	}
	
	protected double wordTopicProb(int tid, int wid, _Doc d) {
		_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA) d;

		return (DCMDoc.m_wordTopic_stat[tid][wid] + m_mu * m_beta[tid][wid])
				/ (DCMDoc.m_sstat[tid] + m_mu * m_totalBeta[tid]);
	}

	protected void updateAlpha(){
		double diff = 0;
		double smallAlpha = 0.1;
		
		int iteration = 0;
		do {

			diff = 0;
			double[] wordNum4Tid = new double[number_of_topics];

			double totalAlphaDenominator = 0;
			
			double deltaAlpha = 0;

			for (int k = 0; k < number_of_topics; k++) {
				wordNum4Tid[k] = 0;
				double totalAlphaNumerator = 0;
				totalAlphaDenominator = 0;
				for (_Doc d : m_trainSet) {
					_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;
					if(DCMDoc.m_topicIndicator[k]==false)
						continue;
					
					wordNum4Tid[k] += DCMDoc.m_sstat[k];

					totalAlphaDenominator += Utils.digamma(DCMDoc.getTotalDocLength()+DCMDoc.m_alphaDoc)-Utils.digamma(DCMDoc.m_alphaDoc);
					totalAlphaNumerator += Utils.digamma(m_alpha[k]
							+ d.m_sstat[k])
							- Utils.digamma(m_alpha[k]);
				}
				
				if(wordNum4Tid[k]==0){
					deltaAlpha = 0;
				}else{
					deltaAlpha = totalAlphaNumerator*1.0/totalAlphaDenominator;
				}

				double newAlpha = m_alpha[k] * deltaAlpha+d_alpha;
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
		
		for(_Doc d:m_trainSet){
			_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;
			DCMDoc.m_alphaDoc = 0;
			for(int k=0; k<number_of_topics; k++){
				if(DCMDoc.m_topicIndicator[k]==true)
					DCMDoc.m_alphaDoc += m_alpha[k];
			}
				
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
			double digBeta4Tid = Utils.digamma(m_mu * m_totalBeta[tid]);

			for (_Doc d : m_trainSet) {
				_Doc4DCMLDA DCMDoc = (_Doc4DCMLDA) d;
				totalBetaDenominator += Utils.digamma(m_mu * m_totalBeta[tid]
						+ DCMDoc.m_sstat[tid])
						- digBeta4Tid;
				for (int v = 0; v < vocabulary_size; v++) {
					wordNum4Tid += DCMDoc.m_wordTopic_stat[tid][v];
					wordNum4Tid4V[v] += DCMDoc.m_wordTopic_stat[tid][v];
					totalBetaNumerator[v] += Utils.digamma(m_mu
							* m_beta[tid][v]
							+ DCMDoc.m_wordTopic_stat[tid][v]);
					totalBetaNumerator[v] -= Utils.digamma(m_mu
							* m_beta[tid][v]);
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

	protected void finalEst(){
		runLastEM();
		for (_Doc d : m_trainSet) {
			estThetaInDoc(d);
		}
		estGlobalParameter();
	}
	
	protected void runLastEM(){

		for (int j = 0; j < number_of_iteration; j++) {
			init();
			for (_Doc d : m_trainSet) {
				_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;

				calculate_E_step(DCMDoc);
				if (j % 20 == 0) {
					DCMDoc.m_MStepIter += 1;

					for (int k = 0; k < number_of_topics; k++)
						if (DCMDoc.m_topicIndicator[k] == true) {
							DCMDoc.m_topicIndicator_prob[k] += 1; // miss m_s
						}

					DCMDoc.m_topicIndicator_distribution += DCMDoc.m_indicatorTrue_stat;
				}
			}
				
		}
		
		collectStats();
		
	}
	
	protected void collectStats(){
		for(int k=0; k<number_of_topics; k++)
			for(int v=0; v<vocabulary_size; v++)
				topic_term_probabilty[k][v] = word_topic_sstat[k][v] + m_mu
						* m_beta[k][v];

		for(_Doc d:m_trainSet){
			_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;
			
			for (int k = 0; k < this.number_of_topics; k++) {
				if(DCMDoc.m_topicIndicator[k]==false)
					continue;
				DCMDoc.m_topics[k] = DCMDoc.m_sstat[k] + m_alpha[k];
	
				for (int v = 0; v < vocabulary_size; v++){
					DCMDoc.m_wordTopic_prob[k][v] = DCMDoc.m_wordTopic_stat[k][v]
							+ m_mu * m_beta[k][v];
				}
			}
		}
	}
	
	protected void estThetaInDoc(_Doc d){
		
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;
		for (int i = 0; i < number_of_topics; i++)
			Utils.L1Normalization(DCMDoc.m_wordTopic_prob[i]);
		Utils.L1Normalization(d.m_topics);

		DCMDoc.m_topicIndicator_distribution /= DCMDoc.m_MStepIter
				* number_of_topics;
		for(int k=0; k<number_of_topics; k++)
			DCMDoc.m_topicIndicator_prob[k] /= DCMDoc.m_MStepIter;
	
	}

	protected void estGlobalParameter(){
		for(int i=0; i<number_of_topics; i++)
			Utils.L1Normalization(topic_term_probabilty[i]);
	}
}
