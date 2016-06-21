package topicmodels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class LDAonArticles extends LDA_Gibbs_Debug{
	public LDAonArticles(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, 
			int number_of_topics, double alpha, double burnIn, int lag, double ksi, double tau) {
		super( number_of_iteration,  converge,  beta,
				 c,  lambda, number_of_topics,  alpha,  burnIn,  lag, ksi, tau);
		
		// TODO Auto-generated constructor stub
	}

	public void EMonCorpus(){

		ArrayList<_Doc> docs = m_corpus.getCollection();
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		for(_Doc d:docs){
			if(d instanceof _ParentDoc){
				m_trainSet.add(d);
			}else{
				m_testSet.add(d);
			}
		}
		EM();
		m_collectCorpusStats = false;
		for(_Doc d:m_testSet)
			inference(d);
	} 
	
	public void crossValidation(int k){
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		
		double[] perf = null;
		
		_Corpus parentCorpus = new _Corpus();
		
		ArrayList<_Doc> docs = m_corpus.getCollection();
		ArrayList<_ParentDoc> parentDocs = new ArrayList<_ParentDoc>();
		
		for(_Doc d:docs){
			if(d instanceof _ParentDoc){
				parentCorpus.addDoc(d);
				parentDocs.add((_ParentDoc)d);
			}
		}
		
		System.out.println("size of parent docs\t"+parentDocs.size());
		
		parentCorpus.setMasks();
		if(m_randomFold==true){
			perf = new double[k];
			parentCorpus.shuffle(k);
			int[] masks = parentCorpus.getMasks();
			
			for(int i=0; i<k; i++){
				for(int j=0; j<masks.length; j++){
					if(masks[j] == i){
						m_testSet.add(parentDocs.get(j));
						for(_ChildDoc d:parentDocs.get(j).m_childDocs){
							m_testSet.add(d);
						}
					}else {
						m_trainSet.add(parentDocs.get(j));
						
					}
					
				}
				
//				writeFile(i, m_trainSet, m_testSet);
				System.out.println("Fold number "+i);
				infoWriter.println("Fold number "+i);
				
				System.out.println("Train Set Size "+m_trainSet.size());
				infoWriter.println("Train Set Size "+m_trainSet.size());
				
				System.out.println("Test Set Size "+m_testSet.size());
				infoWriter.println("Test Set Size "+m_testSet.size());

				long start = System.currentTimeMillis();
				EM();
				perf[i] = Evaluation(i);
				
				System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
				infoWriter.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
				
				if(i<k-1){
					m_trainSet.clear();
					m_testSet.clear();	
				}
			}
			
		}
		
		double mean = Utils.sumOfArray(perf)/k, var = 0;
		for(int i=0; i<perf.length; i++)
			var += (perf[i]-mean) * (perf[i]-mean);
		var = Math.sqrt(var/k);
		System.out.format("Perplexity %.3f+/-%.3f\n", mean, var);
		infoWriter.format("Perplexity %.3f+/-%.3f\n", mean, var);
	}
	
	public double Evaluation(int i){
		m_collectCorpusStats = false;
		double perplexity = 0, loglikelihood, totalWords=0, sumLikelihood = 0;
		
		System.out.println("In Normal");
		
		for(_Doc d:m_testSet) {		
//			if(d instanceof _ChildDoc){
//				inference(d);
//				continue;
//			}
			loglikelihood = inference(d);
			sumLikelihood += loglikelihood;
			perplexity += loglikelihood;
			totalWords += d.getDocTestLength();	
		}
		System.out.println("total Words\t"+totalWords+"perplexity\t"+perplexity);
		infoWriter.println("total Words\t"+totalWords+"perplexity\t"+perplexity);
		perplexity /= totalWords;
		perplexity = Math.exp(-perplexity);
		sumLikelihood /= m_testSet.size();

		System.out.format("Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, sumLikelihood);
		infoWriter.format("Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, sumLikelihood);
		return perplexity;	
	}
	
	public double inference(_Doc d) {
		initTest(d);//this is not a corpus level estimation
		
		double likelihood = Double.NEGATIVE_INFINITY, count = 0;
		int  i = 0;
		do {
			calculate_E_step(d);
			if (i>m_burnIn && i%m_lag==0){
				collectStats(d);
			}
		} while (++i<this.number_of_iteration);
		
		estThetaInDoc(d);			
		likelihood = calculate_test_log_likelihood(d);
		return likelihood;
	
	}
	
	protected double testLogLikelihoodByIntegrateTopics(_ParentDoc d){
		double docLogLikelihood = 0.0;
		
		double docInferLen = d.getDocInferLength();
		
		for(_Word w:d.getTestWords()){
			int wid = w.getIndex();
	
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = wordByTopicProb(k, wid)
						* topicInDocProb(k, d)
						/ (docInferLen + number_of_topics * d_alpha);
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}
		
		return docLogLikelihood;
	}
	
	protected void initTest(_Doc d){
		if(d instanceof _ParentDoc){
			_ParentDoc pDoc = (_ParentDoc)d;
			for(_Stn stnObj: pDoc.getSentences()){
				stnObj.setTopicsVct(number_of_topics);
			}
			
			int testLength = (int)(m_testWord4PerplexityProportion*d.getTotalDocLength());
			pDoc.setTopics4GibbsTest(number_of_topics, d_alpha, testLength);
			
			pDoc.createSparseVct4Infer();
		}else{
			_ChildDoc cDoc = (_ChildDoc)d;
				
			int testLength = (int)(m_testWord4PerplexityProportion*d.getTotalDocLength());
			cDoc.setTopics4GibbsTest(number_of_topics, d_alpha, testLength);
			
			cDoc.createSparseVct4Infer();
		}
	}
	
//	protected void discoverSpecificComments(String similarityFile) {
//		return;
//	}
//	
//	protected void printTopKChild4Parent(String filePrefix, int topK){
//		return;
//	}
//	
//	protected void printTopKStn4Child(String filePrefix, int topK){
//		return;
//	}
	
}
