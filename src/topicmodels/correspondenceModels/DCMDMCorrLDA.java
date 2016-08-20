package topicmodels.correspondenceModels;

import java.util.Arrays;

import jdk.nashorn.internal.ir.Terminal;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4DCM;
import utils.Utils;

public class DCMDMCorrLDA extends DCMCorrLDA {
	double d_alpha_c;
	
	public DCMDMCorrLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha_a,
			double alpha_c, double burnIn, double ksi, double tau, int lag,
			int newtonIter, double newtonConverge){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha_a, alpha_c, burnIn, ksi, tau, lag, newtonIter, newtonConverge);
		d_alpha_c = alpha_c;
	}
	
	public String toString(){
		return String.format("DCMDMCorrLDA[k:%d, alphaA:%.2f, beta:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta);
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc4DCM d) {
		double term = 1.0;

		if (tid == 0)
			return term;
		
		int tidNum = 0;
		int zeroNum = 0;
		
		double mu = 0;
		for(_ChildDoc cDoc:d.m_childDocs){
			mu = cDoc.getMu();
			tidNum += (int)cDoc.m_sstat[tid];
			zeroNum += (int)cDoc.m_sstat[0];
		}
		
		double muDp =  mu/ d.getDocInferLength();
		term *= gammaFuncRatio(tidNum, muDp, m_alpha_c[tid]+ d.m_sstat[tid] * muDp)
					/ gammaFuncRatio(zeroNum, muDp, m_alpha_c[0]
							+ d.m_sstat[0] * muDp);

		return term;

	}
	
	protected double childTopicInDocProb(int tid, _ChildDoc d, _ParentDoc4DCM pDoc) {
		double prob = 0;
		double parentTopicSum = Utils.sumOfArray(pDoc.m_sstat);
			
		double tidNum = 0;
		double totalTidNum = 0;
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			tidNum += cDoc.m_sstat[tid];
			totalTidNum += cDoc.getDocInferLength();
		}
		
		totalTidNum -= 1;
			
		double muDp = d.getMu() / parentTopicSum;
		prob = (m_alpha_c[tid] + muDp * pDoc.m_sstat[tid] + tidNum)
				/ (m_totalAlpha_c + muDp * parentTopicSum + totalTidNum);
		
		return prob;
	}
	
	protected void collectStats(_Doc d){
		if(d instanceof _ParentDoc4DCM){
			_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
			for(int k=0; k<number_of_topics; k++){
				pDoc.m_topics[k] += pDoc.m_sstat[k]+m_alpha[k];
				for(int v=0; v<vocabulary_size; v++){
					pDoc.m_wordTopic_prob[k][v] += pDoc.m_wordTopic_stat[k][v]+m_beta[k][v];
				}
			}
		}else if(d instanceof _ChildDoc){
			_ChildDoc cDoc = (_ChildDoc)d;
			for(int k=0; k<number_of_topics; k++){
				cDoc.m_topics[k] += cDoc.m_sstat[k];
			}
		}
	}
	
	protected void updateAlphaC(){
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
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc){
					_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
					
					Arrays.fill(tidNum, 0);
					double topicSum = 0;
					double mu = 0;
					for(_ChildDoc cDoc:pDoc.m_childDocs){
						mu = cDoc.getMu();
						topicSum += cDoc.getDocInferLength();
						for(int k=0; k<number_of_topics; k++)
							tidNum[k] += cDoc.m_sstat[k];
					}
					
					double pDocLen = pDoc.getDocInferLength();
					double muDp =mu/pDocLen;
					double t_totalAlpha_c = m_totalAlpha_c+mu;
					double digAlpha = Utils.digamma(t_totalAlpha_c);
					totalAlphaDenominator = Utils.digamma(topicSum+t_totalAlpha_c)-digAlpha;
						
					for(int k=0; k<number_of_topics; k++)
						totalAlphaNumerator[k] = Utils.digamma(m_alpha_c[k]+muDp*pDoc.m_sstat[k]+tidNum[k])-Utils.digamma(m_alpha_c[k]+muDp*pDoc.m_sstat[k]);
					
				}
			}
			
			for(int k=0; k<number_of_topics; k++){
				deltaAlpha = totalAlphaNumerator[k]*1.0/totalAlphaDenominator;
				
				double newAlpha = m_alpha_c[k]*deltaAlpha+d_alpha_c;
				double t_diff = Math.abs(m_alpha_c[k]-newAlpha);
				if(t_diff>diff)
					diff = t_diff;
				
				m_alpha_c[k] = newAlpha;
			}
			
			iteration ++;
			// System.out.println("alpha iteration\t" + iteration);
			
			if(iteration > m_newtonIter)
				break;
			
		}while(diff>m_newtonConverge);
		
		// System.out.println("iteration\t" + iteration);
		m_totalAlpha_c = 0;
		for (int k = 0; k < number_of_topics; k++) {
			m_totalAlpha_c += m_alpha_c[k];
		}
	}
	
	protected void initialAlphaBeta(){
		
		double parentDocNum = 0;
		double childDocNum = 0;
		
		Arrays.fill(m_sstat, 0);
		Arrays.fill(m_alphaAuxilary, 0);
		for(int k=0; k<number_of_topics; k++){
			Arrays.fill(topic_term_probabilty[k], 0);
			Arrays.fill(word_topic_sstat[k], 0);
		}
		
		for(_Doc d:m_trainSet){
			if(d instanceof _ParentDoc4DCM){
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
				for(int k=0; k<number_of_topics; k++){
					double tempProb = pDoc.m_sstat[k]/pDoc.getTotalDocLength();
					m_sstat[k] += tempProb;
					
					if(pDoc.m_sstat[k] == 0)
						continue;
					for(int v=0; v<vocabulary_size; v++){
						tempProb = pDoc.m_wordTopic_stat[k][v]/pDoc.m_topic_stat[k];
						topic_term_probabilty[k][v] += tempProb;
					}
				}
				parentDocNum += 1;
			
				double tempProb = 0;
				double[] tempTopicNum = new double[number_of_topics];
				Arrays.fill(tempTopicNum, 0);
				double topicSum = 0;
				
				for(_ChildDoc cDoc:pDoc.m_childDocs){
					for(int k=0; k<number_of_topics; k++){
						topicSum += cDoc.getDocInferLength();
						tempTopicNum[k] += cDoc.m_sstat[k];
						
					}

				}
				
				childDocNum += 1;
				for(int k=0; k<number_of_topics; k++){
					tempProb = tempTopicNum[k]*1.0/topicSum;
					m_alphaAuxilary[k] += tempProb;
				}
			}
		}
		
		for(int k=0; k<number_of_topics; k++){
			m_sstat[k] /= parentDocNum;
			m_alphaAuxilary[k] /= childDocNum;
			for(int v=0; v<vocabulary_size; v++){
				topic_term_probabilty[k][v] /= (parentDocNum+childDocNum);
			}
		}
		
		for(int k=0; k<number_of_topics; k++){
			m_alpha[k] = m_sstat[k];
			m_alpha_c[k] = m_alphaAuxilary[k]+d_alpha_c;
			for(int v=0; v<vocabulary_size; v++)
				m_beta[k][v] = topic_term_probabilty[k][v]+d_beta;
		}
		
		m_totalAlpha = Utils.sumOfArray(m_alpha);
		m_totalAlpha_c = Utils.sumOfArray(m_alpha_c);
		for(int k=0; k<number_of_topics; k++){
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);
		}
		
	}
	
	protected double calculate_log_likelihood(_ParentDoc4DCM d){
		double docLogLikelihood = 0;
		int docID = d.getID();
		
		double parentDocLength = d.getTotalDocLength();
		
		for(int k=0; k<number_of_topics; k++){
			double term = Utils.lgamma(d.m_sstat[k]+m_alpha[k]);
			docLogLikelihood += term;
			
			term = Utils.lgamma(m_alpha[k]);
			docLogLikelihood -= term;
		}
		
		docLogLikelihood += Utils.lgamma(m_totalAlpha);
		docLogLikelihood -= Utils.lgamma(parentDocLength+m_totalAlpha);
		for(int k=0; k<number_of_topics; k++){
			for(int v=0; v<vocabulary_size; v++){
				double term = Utils.lgamma(d.m_wordTopic_stat[k][v]+m_beta[k][v]);
				docLogLikelihood += term;
				
				term = Utils.lgamma(m_beta[k][v]);
				docLogLikelihood -= term;
			}
			
			docLogLikelihood += Utils.lgamma(m_totalBeta[k]);
			docLogLikelihood -= Utils.lgamma(d.m_topic_stat[k]+m_totalBeta[k]);
		}
				
		double[] tidNum = new double[number_of_topics];
		Arrays.fill(tidNum, 0);

		double topicSum = 0;
		double mu = 0;
		
		for(_ChildDoc cDoc:d.m_childDocs){
			mu = cDoc.getMu();
			topicSum += cDoc.getDocInferLength();
			for(int k=0; k<number_of_topics; k++)
				tidNum[k] += cDoc.m_sstat[k];
		}
		
		double muDp = mu/parentDocLength;
		docLogLikelihood += Utils.digamma(m_totalAlpha_c+mu);

		docLogLikelihood += Utils.digamma(m_totalAlpha_c+mu+topicSum);

		for(int k=0; k<number_of_topics; k++){
			double term = m_alpha_c[k]+muDp*d.m_sstat[k]+tidNum[k];
			term = Utils.digamma(m_alpha_c[k]+muDp*d.m_sstat[k]+tidNum[k]);

			double term2 = m_alpha_c[k]+muDp*d.m_sstat[k];
			term2 = Utils.digamma(m_alpha_c[k]+muDp*d.m_sstat[k]);
			docLogLikelihood += term-term2;
		}

		return docLogLikelihood;
	}
	
}
