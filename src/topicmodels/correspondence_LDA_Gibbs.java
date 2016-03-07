package topicmodels;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class correspondence_LDA_Gibbs extends LDA_Gibbs_Debug{
	boolean m_statisticsNormalized = false;//a warning sign of normalizing statistics before collecting new ones
	double[] m_topicProbCache;
	
	public correspondence_LDA_Gibbs(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag);
	
		m_topicProbCache = new double[number_of_topics];
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection){
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		
		for(_Doc d: collection){
			if(d instanceof _ParentDoc){
				for(_Stn stnObj: d.getSentences()){
					stnObj.setTopicsVct(number_of_topics);	
				}
				d.setTopics4Gibbs(number_of_topics, 0);

			}
			else if(d instanceof _ChildDoc){
				((_ChildDoc) d).setTopics4Gibbs_LDA(number_of_topics, 0);
			}
			
			for(_Word w:d.getWords()){
				word_topic_sstat[w.getTopic()][w.getIndex()] ++;
				m_sstat[w.getTopic()] ++;
			}
		
		}
		
		imposePrior();
		
		m_statisticsNormalized = false;
	}
	
	public String toString(){
		return String.format("correspondence LDA [k:%d, alpha:%.2f, beta:%.2f, Gibbs Sampling]",
				number_of_topics, d_alpha, d_beta);
	}
	
	public double calculate_E_step(_Doc d){
		d.permutation();
		
		if(d instanceof _ParentDoc)
			sampleInParentDoc((_ParentDoc)d);
		else if(d instanceof _ChildDoc)
			sampleInChildDoc((_ChildDoc)d);
		
		return 0;
	}
	
	public void sampleInParentDoc(_ParentDoc d){
		int wid, tid;
		double normalizedProb;
		
		for(_Word w:d.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			
			d.m_sstat[tid] --;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
			normalizedProb = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = parentWordByTopicProb(tid, wid);
				double pTopicPDoc = parentTopicInDocProb(tid, d);
				double pTopicCDoc = parentChildInfluenceProb(tid, d);
				
				m_topicProbCache[tid] = pWordTopic*pTopicPDoc*pTopicCDoc;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<0)
					break;
			}
			
			if(tid == number_of_topics)
				tid --;
			
			w.setTopic(tid);
			d.m_sstat[tid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
			
		}
	}
	
	protected double parentWordByTopicProb(int tid, int wid){
		return word_topic_sstat[tid][wid]/m_sstat[tid];
	}
	
	protected double parentTopicInDocProb(int tid, _ParentDoc d){
		return (d_alpha+d.m_sstat[tid]);
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc d){
		double term = 1;
		
		if(tid==0)
			return term;
		
		for(_ChildDoc cDoc: d.m_childDocs){
			for(_Word w:cDoc.getWords()){
				int tempTid = w.getTopic();
				if(tempTid == tid)
					term *= (d.m_sstat[tempTid]+1)/(d.m_sstat[tempTid]+1e-10);
				else 
					if(tempTid == 0)
						term *= (d.m_sstat[tempTid]+1e-10)/(d.m_sstat[tempTid]+1);
				
			}
		}
		
		if(term==0){
			term += 1e-10;
		}
		
		return term;
	}
	
	protected void sampleInChildDoc(_ChildDoc d){
		int wid, tid;
		double normalizedProb = 0;
		
		for(_Word w: d.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			
			d.m_sstat[tid]--;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
			normalizedProb = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = childWordByTopicProb(tid, wid);
				double pTopicDoc = childTopicInDoc(tid, d);
				
				m_topicProbCache[tid] = pWordTopic*pTopicDoc;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<0)
					break;
			}
			
			if(tid == number_of_topics)
				tid --;
			
			w.setTopic(tid);
			d.m_sstat[tid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
		}
	}
	
	protected double childWordByTopicProb(int tid, int wid){
		return word_topic_sstat[tid][wid]/m_sstat[tid];
	}
	
	protected double childTopicInDoc(int tid, _ChildDoc d){
		_ParentDoc tempParentDoc = d.m_parentDoc;
		double term = tempParentDoc.m_sstat[tid];
		if (term==0)
			term += 1e-10;
		return term;
	}
	
	public void calculate_M_step(int iter){
		if(iter>m_burnIn && iter%m_lag==0){
			if(m_statisticsNormalized){
				System.err.println("The statistics collector has been normlaized before, cannot further accumulate the samples!");
				System.exit(-1);
			}
			
			for(int i=0; i<number_of_topics; i++){
				for(int v=0; v<vocabulary_size; v++){
					topic_term_probabilty[i][v] += word_topic_sstat[i][v];
				}
			}
			
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc)
					collectParentStats((_ParentDoc)d);
				else if(d instanceof _ChildDoc)
					collectChildStats((_ChildDoc)d);
					
			}
		}
	}
	
	public void collectParentStats(_ParentDoc d){
		for(int k=0; k<number_of_topics; k++)
			d.m_topics[k] += d.m_sstat[k] + d_alpha;
		d.collectTopicWordStat();		
	}
	
	public void collectChildStats(_ChildDoc d){
		for(int k=0; k<number_of_topics; k++)
			d.m_topics[k] += d.m_sstat[k];
		
		_ParentDoc pDoc = d.m_parentDoc;
//		rankStn4Child(d, pDoc);
	}
	
	protected void estThetaInDoc(_Doc d){
		super.estThetaInDoc(d);
		
		if(d instanceof _ParentDoc)
			estParentStnTopicProportion((_ParentDoc)d);
	}
	
	public void estParentStnTopicProportion(_ParentDoc pDoc){
		for(_Stn stnObj : pDoc.getSentences() ){
			estStn(stnObj, pDoc);
		}
	}
	
	public void estStn(_Stn stnObj,  _ParentDoc d){
		int i=0;
		initStn(stnObj);
		do{
			calculateStn_E_step(stnObj, d);
			if(i>m_burnIn && i%m_lag == 0){
				collectStnStats(stnObj, d);
			}
			
		}while(++i<number_of_iteration);
		
		Utils.L1Normalization(stnObj.m_topics);
	}
	
	public void initStn(_Stn stnObj){
		stnObj.setTopicsVct(number_of_topics);
	}
	
	public void calculateStn_E_step( _Stn stnObj, _ParentDoc d){
		stnObj.permuteStn();
		
		double normalizedProb = 0;
		int wid, tid;
		for(_Word w: stnObj.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
	
			stnObj.m_topicSstat[tid] --;
		
			normalizedProb = 0;
			
			double pWordTopic = 0;
			
			for(tid=0; tid<number_of_topics; tid++){
				pWordTopic = wordByTopicProb(tid, wid);
				double pTopic = parentTopicInStnProb(tid, stnObj, d);
				
				m_topicProbCache[tid] = pWordTopic*pTopic;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<m_topicProbCache.length; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0)
					break;
			}
			
			if(tid==m_topicProbCache.length)
				tid --;
			
			w.setTopic(tid);
			stnObj.m_topicSstat[tid] ++;
			
		}
		
	}
	
	public void crossValidation(int k) {
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		
		double[] perf = null;
		
		_Corpus parentCorpus = new _Corpus();
		ArrayList<_Doc> docs = m_corpus.getCollection();
		ArrayList<_ParentDoc> parentDocs = new ArrayList<_ParentDoc>();
		for(_Doc d: docs){
			if(d instanceof _ParentDoc){
				parentCorpus.addDoc(d);
				parentDocs.add((_ParentDoc) d);
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
					}else {
						m_trainSet.add(parentDocs.get(j));
						for(_ChildDoc d: parentDocs.get(j).m_childDocs){
							m_trainSet.add(d);
						}
					}
					
				}
				
//				writeFile(i, m_trainSet, m_testSet);
				System.out.println("Fold number "+i);
				System.out.println("Train Set Size "+m_trainSet.size());
				System.out.println("Test Set Size "+m_testSet.size());

				long start = System.currentTimeMillis();
				EM();
				perf[i] = Evaluation(i);
				
				System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
				m_trainSet.clear();
				m_testSet.clear();			
			}
			
		}
		double mean = Utils.sumOfArray(perf)/k, var = 0;
		for(int i=0; i<perf.length; i++)
			var += (perf[i]-mean) * (perf[i]-mean);
		var = Math.sqrt(var/k);
		System.out.format("Perplexity %.3f+/-%.3f\n", mean, var);
	}
	
	public double Evaluation(int i) {
		m_collectCorpusStats = false;
		double perplexity = 0, loglikelihood, totalWords=0, sumLikelihood = 0;
		
		System.out.println("In Normal");
		
		for(_Doc d:m_testSet) {				
			loglikelihood = inference(d);
			sumLikelihood += loglikelihood;
			perplexity += loglikelihood;
			totalWords += d.getTotalDocLength();
			for(_ChildDoc cDoc: ((_ParentDoc)d).m_childDocs){
				totalWords += cDoc.getTotalDocLength();
			}
		}
		System.out.println("total Words\t"+totalWords+"perplexity\t"+perplexity);
		perplexity /= totalWords;
		perplexity = Math.exp(-perplexity);
		sumLikelihood /= m_testSet.size();

		System.out.format("Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, sumLikelihood);
		
		return perplexity;		
	}
	
	@Override
	public double inference(_Doc pDoc){
		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();
		
		initTest(sampleTestSet, pDoc);
	
		double logLikelihood = 0.0, count = 0;
		int  iter = 0;
		do {
			int t;
			_Doc tmpDoc;
			for(int i=sampleTestSet.size()-1; i>1; i--) {
				t = m_rand.nextInt(i);
				
				tmpDoc = sampleTestSet.get(i);
				sampleTestSet.set(i, sampleTestSet.get(t));
				sampleTestSet.set(t, tmpDoc);			
			}
			
			for(_Doc doc: sampleTestSet)
				calculate_E_step(doc);
			
			if (iter>m_burnIn && iter%m_lag==0){
				double tempLogLikelihood = 0;
				for(_Doc doc: sampleTestSet){
					if(doc instanceof _ParentDoc){
						collectParentStats((_ParentDoc) doc);
						tempLogLikelihood += calculate_log_likelihood((_ParentDoc) doc);
					}
					else if(doc instanceof _ChildDoc){
						collectChildStats((_ChildDoc) doc);
						tempLogLikelihood += calculate_log_likelihood((_ChildDoc) doc);
					}
					
				}
				count ++;
				if(logLikelihood == 0)
					logLikelihood = tempLogLikelihood;
				else{

					logLikelihood = Utils.logSum(logLikelihood, tempLogLikelihood);
				}
			}
		} while (++iter<this.number_of_iteration);

		for(_Doc doc: sampleTestSet){
			estThetaInDoc(doc);
		}
		
		return logLikelihood - Math.log(count); 	
	}

	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc pDoc = (_ParentDoc)d;
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		pDoc.setTopics4Gibbs(number_of_topics, 0);		
		sampleTestSet.add(pDoc);
		
		for(_ChildDoc cDoc: pDoc.m_childDocs){
			cDoc.setTopics4Gibbs_LDA(number_of_topics, 0);
			sampleTestSet.add(cDoc);
		}
	}

	public double calculate_log_likelihood(_Doc d){
		if (d instanceof _ParentDoc)
			return logLikelihoodByIntegrateTopics((_ParentDoc)d);
		else
			return logLikelihoodByIntegrateTopics((_ChildDoc)d);
	}
	
	public double logLikelihoodByIntegrateTopics(_ParentDoc d){
		double docLogLikelihood = 0;
		
		_SparseFeature[] fv = d.getSparse();
		
		for(int j=0; j<fv.length; j++){
			int wid = fv[j].getIndex();
			double value = fv[j].getValue();
			
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)*parentTopicInDocProb(k, d)/(d_alpha*number_of_topics+d.getTotalDocLength());
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			if(Math.abs(wordLogLikelihood)<1e-10){
				System.out.println("wordLogLikelihood\t"+wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}
			
			wordLogLikelihood = Math.log(wordLogLikelihood);
			
			docLogLikelihood += value*wordLogLikelihood;
		}
		
		return docLogLikelihood;
	}
	
	public double logLikelihoodByIntegrateTopics(_ChildDoc d){
		double docLogLikelihood = 0;
		
		_SparseFeature[] fv = d.getSparse();
		for(int i=0; i<fv.length; i++){
			int wid = fv[i].getIndex();
			double value = fv[i].getValue();
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*d.m_sstat[k]/d.getTotalDocLength();
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			if(wordLogLikelihood< 1e-10){
				wordLogLikelihood += 1e-10;
				System.out.println("small likelihood in child");
			}
			
			wordLogLikelihood = Math.log(wordLogLikelihood);
			
			docLogLikelihood += value*wordLogLikelihood;
		}
		
		return docLogLikelihood;
	}
	
	//stn is a query, retrieve comment by likelihood
	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj, _ParentDoc pDoc){
	
		HashMap<String, Double>childLikelihoodMap = new HashMap<String, Double>();

		for(_ChildDoc cDoc:pDoc.m_childDocs){
			int cDocLen = cDoc.getTotalDocLength();
			
			double stnLogLikelihood = 0;
			for(_Word w: stnObj.getWords()){
				double wordLikelihood = 0;
				int wid = w.getIndex();
			
				for(int k=0; k<number_of_topics; k++){
					wordLikelihood += (word_topic_sstat[k][wid]/m_sstat[k])*((cDoc.m_sstat[k]+d_alpha)/(d_alpha*number_of_topics+cDocLen));
				}
				
				stnLogLikelihood += Math.log(wordLikelihood);
			}
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
//			if(cDoc.m_stnLikelihoodMap.containsKey(stnObj.getIndex()))
//				stnLogLikelihood += cDoc.m_stnLikelihoodMap.get(stnObj.getIndex());
//			cDoc.m_stnLikelihoodMap.put(stnObj.getIndex(), stnLogLikelihood);
//		}	
	}
	
//	public void rankStn4Child(_ChildDoc cDoc, _ParentDoc pDoc){
//		int pDocLen = pDoc.getTotalDocLength();
//		for(_Stn stnObj:pDoc.getSentences()){
//			double stnSim = computeSimilarity(cDoc.m_topics, stnObj.m_topics);
//			cDoc.m_stnLikelihoodMap.put(stnObj.getIndex(), stnSim);
//			
//			double stnLogLikelihood = 0;
//			for(_Word w: stnObj.getWords()){
//				double wordLikelihood = 0;
//				int wid = w.getIndex();
//			
//				for(int k=0; k<number_of_topics; k++){
//					wordLikelihood += childWordByTopicProb(k, wid)*childTopicInDoc(k, cDoc)/(d_alpha*number_of_topics+pDocLen);
//				}
//				
//				stnLogLikelihood += Math.log(wordLikelihood);
//			}
//			
//			if(cDoc.m_stnLikelihoodMap.containsKey(stnObj.getIndex()))
//				stnLogLikelihood += cDoc.m_stnLikelihoodMap.get(stnObj.getIndex());
//			cDoc.m_stnLikelihoodMap.put(stnObj.getIndex(), stnLogLikelihood);
//		}
//	}
	
}


