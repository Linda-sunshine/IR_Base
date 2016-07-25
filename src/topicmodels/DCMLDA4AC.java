package topicmodels;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;

import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4DCM;
import structures._Word;
import utils.Utils;

/**
 * 
 * per article and its comments sharing a topicWordDistribution
 * 
 * */

public class DCMLDA4AC extends LDA_Gibbs_Debug{
	protected double[] m_alpha;
	protected double[][] m_beta;
	
	protected double m_totalAlpha;
	protected double[] m_totalBeta;
	
	protected int m_newtonIter;
	protected double m_newtonConverge;
	
	public DCMLDA4AC(int number_of_iteration, double converge, double beta, _Corpus c, double lambda, int number_of_topics, double alpha, double  burnIn, int lag, double ksi, double tau, int newtonIter, double newtonConverge){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, ksi, tau);
	
		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];
		
		m_totalAlpha = 0;
		m_totalBeta = new double[number_of_topics];
		
		m_newtonIter = newtonIter;
		m_newtonConverge = newtonConverge;
	}
	
	public String toString(){
		return String.format("DCMLDA4AC[k:%d, alphaA:%.2f, beta:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta);
	}
	
	public void EM(){
		System.out.format("Starting %s ... \n", toString());
		
		long startTime = System.currentTimeMillis();
		
		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);
		
		String filePrefix = "./data/results/DCM_LDA";
		File weightFolder = new File(filePrefix+"");
		if(!weightFolder.exists()){
			weightFolder.mkdir();
		}
		
		double delta = 0, last = 0, current = 0;
		
		int i=0, displayCount = 0;
		do{
			long eStartTime = System.currentTimeMillis();
			for(int j=0; j<number_of_iteration; j++){
				init();
				for(_Doc d:m_trainSet){
					calculate_E_step(d);
				}
			}
			
			long eEndTime = System.currentTimeMillis();
			
			System.out.println("per iteration e step time\t"+(eEndTime-eStartTime));
			
			long mStartTime = System.currentTimeMillis();
			calculate_M_step(i, weightFolder);
			long mEndTime = System.currentTimeMillis();
			
			System.out.println("per iteration m step time\t"+(mEndTime-mStartTime));
			
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
			
		}while(++i<number_of_iteration);
		
		finalEst();
		
		long endTime = System.currentTimeMillis() - startTime;
		
		System.out.format("likelihood %.3f after step %s converge to %f after %d seconds ...\n", current, i, delta, endTime/1000);
	}
	
	protected void initialize_probability(Collection<_Doc>collection){
		for(_Doc d:collection){
			if(d instanceof _ParentDoc4DCM){
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
				pDoc.setTopics4Gibbs(number_of_topics, 0, vocabulary_size);
				
				for(_ChildDoc cDoc:pDoc.m_childDocs){
					cDoc.setTopics4Gibbs_LDA(number_of_topics, 0);
					
					for(_Word w:cDoc.getWords()){
						int wid = w.getIndex();
						int tid = w.getTopic();
						
						pDoc.m_wordTopic_stat[tid][wid] ++;
						pDoc.m_topic_stat[tid] ++;
					}
					computeMu4Doc(cDoc);
				}
			}
			
		}
		
		initialAlphaBeta();
		imposePrior();
	}
	
	protected void computeMu4Doc(_ChildDoc d){
		_ParentDoc tempParent = d.m_parentDoc;
		double mu = Utils.cosine(tempParent.getSparse(), d.getSparse());
		mu = 0.5;
		d.setMu(mu);
	}
	
	protected void computeTestMu4Doc(_ChildDoc d){
		_ParentDoc pDoc = d.m_parentDoc;
		
		double mu = Utils.cosine(d.getSparseVct4Infer(), pDoc.getSparseVct4Infer());
		mu = 0.05;
		d.setMu(mu);
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
				double pWordTopic = wordTopicProb(tid, wid, pDoc);
				double pTopicPDoc = topicInDocProb(tid, pDoc);
				
				m_topicProbCache[tid] = pWordTopic*pTopicPDoc;
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
				double pWordTopic = wordTopicProb(tid, wid, pDoc);
				double pTopic = topicInDocProb(tid, d);
				
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
	
	protected double topicInDocProb(int tid, _Doc d){
		double term1 = d.m_sstat[tid];
		
		return (d.m_sstat[tid]+m_alpha[tid])/(d.getDocInferLength()+m_totalAlpha-1);
	}
	
	protected double wordTopicProb(int tid, int wid, _ParentDoc4DCM d){
		double term1 = d.m_wordTopic_stat[tid][wid];
		
		return (term1+m_beta[tid][wid])/(d.m_sstat[tid]+m_totalBeta[tid]);
	}
	
	public void calculate_M_step(int iter, File weightFolder) {

		for (_Doc d : m_trainSet){
			if(d instanceof _ParentDoc4DCM)
				collectParentStats((_ParentDoc4DCM)d);
			else
				collectChildStats((_ChildDoc)d);
		}
			
			collectStats(d);

		File weightIterFolder = new File(weightFolder, "_" + iter);
		if (!weightIterFolder.exists()) {
			weightIterFolder.mkdir();
		}

		updateParameter(iter, weightIterFolder);

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
	
	protected void collectParentStats(_ParentDoc4DCM d) {
		int docID = d.getID();

		for (int k = 0; k < this.number_of_topics; k++) {
			d.m_topics[k] += d.m_sstat[k] + m_alpha[k];

			for (int v = 0; v < vocabulary_size; v++){
				d.m_wordTopic_prob[k][v] += d.m_wordTopic_stat[k][v] + m_beta[k][v];
			}
		}

	}
	
	
	
}
