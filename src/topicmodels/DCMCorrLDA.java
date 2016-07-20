package topicmodels;

import org.netlib.util.doubleW;

import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4DCM;
import structures._Word;

public class DCMCorrLDA extends DCMLDA{
	
	protected double m_alpha_c;
	
	public DCMCorrLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, 
			double alpha_a, double alpha_c, double burnIn, int lag, int newtonIter, double newtonConverge){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha_a, burnIn, lag, newtonIter, newtonConverge);
		
		m_alpha_c = alpha_c;
	}
	
	public String toString(){
		return String.format("DCMCorrLDA[k:%d, alpha^a:%.2f, alpha^c:%.2f, beta:%.2f, training proportion:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, m_alpha_c, d_beta, m_testWord4PerplexityProportion);
	}
	
	public double calculate_E_step(_Doc d){
		d.permutation();
		
		if(d instanceof _ParentDoc)
			sampleInParentDoc((_ParentDoc)d);
		else if(d instanceof _ChildDoc)
			sampleInChildDoc((_ChildDoc)d);
		
		return 0;
	}
	
	protected void sampleInParentDoc(_Doc d){
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
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = parentWordByTopicProb(tid, wid, pDoc);
				double pTopicPDoc = parentTopicInDocProb(tid, pDoc);
				double pTopicCDoc = parentChildInfluenceProb(tid, pDoc);
				
				m_topicProbCache[tid] = pWordTopic*pTopicPDoc*pTopicCDoc;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}
			
			if(tid==number_of_topics)
				tid --;
			
			w.setTopic(tid);
			pDoc.m_sstat[tid]++;
			pDoc.m_topic_stat[tid]++;
			pDoc.m_wordTopic_stat[tid][wid]++;
		}
		
	}
	
	protected void sampleInChildDoc(_ChildDoc d){
		int wid, tid;
		double normalizedProb;
		
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d.m_parentDoc;

		for(_Word w:d.getWords()){
			tid = w.getTopic();
			wid = w.getIndex();
			
			pDoc.m_wordTopic_stat[tid][wid]--;
			pDoc.m_topic_stat[tid] --;
			d.m_sstat[tid] --;

			normalizedProb = 0;
			for (tid = 0; tid < number_of_topics; tid++) {
				double pWordTopic = childWordByTopicProb(tid, wid, pDoc);
				double pTopic = childTopicInDocProb(tid, d, pDoc);
				
				m_topicProbCache[tid] = pWordTopic * pTopic;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for (tid = 0; tid < m_topicProbCache.length; tid++) {
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}
			
			if(tid==m_topicProbCache.length)
				tid--;
			
			w.setTopic(tid);
			d.m_sstat[tid]++;
			pDoc.m_topic_stat[tid]++;
			pDoc.m_wordTopic_stat[tid][wid]++;
		}
	}

	protected double parentWordByTopicProb(int tid, int wid, _ParentDoc4DCM d) {
		double prob = 0;
		prob = (d.m_wordTopic_stat[tid][wid] + m_beta[tid][wid])
				/ (d.m_topic_stat[tid] + m_totalBeta[tid]);
		
		return prob;
	}
	
	protected double parentTopicInDocProb(int tid, _ParentDoc4DCM d) {
		double prob = 0;
		
		prob = (d.m_sstat[tid] + m_alpha[tid])
				/ (d.getTotalDocLength() + m_totalAlpha);

		return prob;
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc4DCM d) {
		double term = 1.0;

		if (tid == 0)
			return term;
		
		for (_ChildDoc cDoc : d.m_childDocs) {
			double muDp = cDoc.getMu();
			term *= gammaFuncRatio((int) cDoc.m_sstat[tid], muDp, d_alpha
					+ d.m_sstat[tid] * muDp)
					/ gammaFuncRatio((int) cDoc.m_sstat[0], muDp, d_alpha
							+ d.m_sstat[0] * muDp);
		}

		return term;

	}

	protected double gammaFuncRatio(int nc, double muDp, double alphaMuDp) {
		if (nc == 0)
			return 1.0;

		double result = 1.0;
		for (int n = 1; n <= nc; n++) {
			result *= 1 + muDp / (alphaMuDp + n - 1);
		}

		return result;
	}
	
	protected double childWordByTopicProb(int tid, int wid, _ParentDoc4DCM d) {
		double prob = 0;
		prob = (d.m_wordTopic_stat[tid][wid] + m_beta[tid][wid])
				/ (d.m_topic_stat[tid] + m_totalBeta[tid]);
		return prob;
	}
	
	protected double childTopicInDocProb(int tid, _ChildDoc d, _ParentDoc4DCM pDoc) {
		double prob = 0;
		double parentDocLength = d.m_parentDoc.getDocInferLength();
		double childDocLength = d.getDocInferLength();
				
		prob = (m_alpha_c[tid]+d.getMu()*pDoc.m_sstat[tid]+d.m_sstat[tid])/
				(m_totalAlpha_c+d.getMu()*parentDocLength+childDocLength)
		
		return prob;
	}
	
	
}

