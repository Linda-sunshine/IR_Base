package topicmodels.correspondenceModels;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import structures.MyPriorityQueue;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._RankItem;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import topicmodels.LDA.LDA_Gibbs;
import utils.Utils;

public class ACCTM extends LDA_Gibbs {
	
	protected double[] m_topicProbCache;
	protected double m_kAlpha;	
	protected boolean m_statisticsNormalized = false;//a warning sign of normalizing statistics before collecting new ones
	
	public ACCTM(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag);
		
		m_topicProbCache = new double[number_of_topics];
		m_kAlpha = d_alpha * number_of_topics;
	}
	
	@Override
	public String toString(){
		return String.format("Parent Child Base topic model [k:%d, alpha:%.2f, beta:%.2f, training proportion:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_testWord4PerplexityProportion);
	}
	
	protected void initialize_probability(Collection<_Doc> collection){
		createSpace();
		
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size); // avoid adding such prior later on
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				d.setTopics4Gibbs(number_of_topics, 0);
				for(_Stn stnObj: d.getSentences())
					stnObj.setTopicsVct(number_of_topics);				
			} else if(d instanceof _ChildDoc){
				((_ChildDoc) d).setTopics4Gibbs_LDA(number_of_topics, 0);
				computeMu4Doc((_ChildDoc) d);
			}
			
			for (_Word w:d.getWords()) {
				word_topic_sstat[w.getTopic()][w.getIndex()]++;
				m_sstat[w.getTopic()]++;
			}			

		}
	
		imposePrior();
		
		m_statisticsNormalized = false;
	}
	
	protected void computeMu4Doc(_ChildDoc d){
		_ParentDoc tempParent = d.m_parentDoc;
		double mu = Utils.cosine(tempParent.getSparse(), d.getSparse());
		d.setMu(mu);
	}
	
	protected void computeTestMu4Doc(_ChildDoc d){
		_ParentDoc pDoc = d.m_parentDoc;
		
		double mu = Utils.cosine(d.getSparseVct4Infer(), pDoc.getSparseVct4Infer());
		mu = 1e32;
		d.setMu(mu);
	}
	
	@Override
	public double calculate_E_step(_Doc d){
		d.permutation();
		
		if(d instanceof _ParentDoc)
			sampleInParentDoc((_ParentDoc)d);
		else if(d instanceof _ChildDoc)
			sampleInChildDoc((_ChildDoc)d);
		
		return 0;
	}
	
	protected void sampleInParentDoc(_ParentDoc d){
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
				if(normalizedProb <= 0)
					break;
			}
			
			if(tid==number_of_topics)
				tid --;
			
			w.setTopic(tid);
			d.m_sstat[tid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
		}
	}
	
	//probability of word given topic p(w|z, phi^p, beta)
	protected double parentWordByTopicProb(int tid, int wid){
		return word_topic_sstat[tid][wid] / m_sstat[tid];
	}

	//probability of topic given doc p(z|d, alpha)
	protected double parentTopicInDocProb(int tid, _ParentDoc d){
		return d_alpha + d.m_sstat[tid];
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc pDoc){
		double term = 1.0;
		
		if(tid==0)
			return term;
		
		for(_ChildDoc cDoc: pDoc.m_childDocs){
			double muDp = cDoc.getMu()/pDoc.getDocInferLength();
			term *= gammaFuncRatio((int)cDoc.m_sstat[tid], muDp, d_alpha+pDoc.m_sstat[tid]*muDp)
					/ gammaFuncRatio((int)cDoc.m_sstat[0], muDp, d_alpha+pDoc.m_sstat[0]*muDp);
		}
		
		return term;
	}
	
	protected double gammaFuncRatio(int nc, double muDp, double alphaMuNp) {
		if (nc==0)
			return 1.0;
		
		double result = 1.0;
		for (int n = 1; n <= nc; n++)
			result *= 1 + muDp / (alphaMuNp + n - 1);
		return result;
	}
	
	protected void sampleInChildDoc(_ChildDoc d){
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
				double pWordTopic = childWordByTopicProb(tid, wid);
				double pTopicCDoc = childTopicInDocProb(tid, d);
				
				m_topicProbCache[tid] = pWordTopic*pTopicCDoc;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0)
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
	
	//probability of word given topic p(w|z, phi^c, beta)
	protected double childWordByTopicProb(int tid, int wid){
		return word_topic_sstat[tid][wid] / m_sstat[tid];
	}

	protected double childTopicInDocProb(int tid, _ChildDoc d){
		double parentDocLength = d.m_parentDoc.getDocInferLength();
		double childDocLength = d.getDocInferLength();
		
		return (d_alpha + d.getMu()*d.m_parentDoc.m_sstat[tid]/parentDocLength + d.m_sstat[tid])
					/(m_kAlpha + d.getMu() + childDocLength);
	
	}
	
	@Override
	public void calculate_M_step(int iter){

		if(iter>m_burnIn && iter%m_lag==0){
			if (m_statisticsNormalized) {
				System.err.println("The statistics collector has been normlaized before, cannot further accumulate the samples!");
				System.exit(-1);
			}
			
			for(int i=0; i<this.number_of_topics; i++){
				for(int v=0; v<this.vocabulary_size; v++){
					topic_term_probabilty[i][v] += word_topic_sstat[i][v];//collect the current sample
				}
			}
			
			// used to estimate final theta for each document
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc)
					collectParentStats((_ParentDoc)d);
				else if(d instanceof _ChildDoc)
					collectChildStats((_ChildDoc)d);
			}
		}
	}	
	
	protected void collectStats(_Doc d){
		if(d instanceof _ParentDoc){
			collectParentStats((_ParentDoc)d);
		}else if(d instanceof _ChildDoc){
			collectChildStats((_ChildDoc)d);
		}
	}
	
	//such statistic collection mechanism makes us unable to normalize the corresponding structure for efficient likelihood computation
	protected void collectParentStats(_ParentDoc d) {
		for (int k = 0; k < this.number_of_topics; k++) 
			d.m_topics[k] += d.m_sstat[k] + d_alpha;
		d.collectTopicWordStat();
	}
	
	protected void collectChildStats(_ChildDoc d) {
		_ParentDoc pDoc = d.m_parentDoc;
		double parentDocLength = pDoc.getDocInferLength();
		for (int k = 0; k < this.number_of_topics; k++) 
			d.m_topics[k] += d.m_sstat[k] + d_alpha
		+d.getMu()*pDoc.m_sstat[k] / parentDocLength;

//			d.m_topics[k] += d.m_sstat[k] + d_alpha+d.getMu()*pDoc.m_sstat[k] / parentDocLength;
	}
	
	protected void estThetaInDoc(_Doc d){
		Utils.L1Normalization(d.m_topics);
		if(d instanceof _ParentDoc){
			((_ParentDoc)d).estStnTheta();
//			estParentStnTopicProportion((_ParentDoc)d);
		}
	}
	
	double computeSimilarity(double[] topic1, double[] topic2) {
		return Utils.cosine(topic1, topic2);
	}
	
	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc pDoc = (_ParentDoc)d;
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		
		int testLength = 0;
//		int testLength = (int)(m_testWord4PerplexityProportion*d.getTotalDocLength());
		pDoc.setTopics4GibbsTest(number_of_topics, 0, testLength);
		sampleTestSet.add(pDoc);
		pDoc.createSparseVct4Infer();

		for(_ChildDoc cDoc: pDoc.m_childDocs){
			testLength = (int)(m_testWord4PerplexityProportion*cDoc.getTotalDocLength());
			cDoc.setTopics4GibbsTest(number_of_topics, 0, testLength);
			sampleTestSet.add(cDoc);
			cDoc.createSparseVct4Infer();

			computeTestMu4Doc(cDoc);
		}
	}
	
	protected double inference4Doc(ArrayList<_Doc> sampleTestSet){
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
				for(_Doc doc: sampleTestSet){
					collectStats(doc);
				}
			}
		} while (++iter<this.number_of_iteration);

		for(_Doc doc: sampleTestSet){
			estThetaInDoc(doc);
		}
		
		return logLikelihood;
	}
	
	@Override
	public double inference(_Doc pDoc) {
		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();
		
		initTest(sampleTestSet, pDoc);
		
		double logLikelihood = 0.0, count = 0;
		inferenceParentDoc((_ParentDoc)pDoc);
		logLikelihood = inferenceChildDoc((_ParentDoc)pDoc);
		
		return logLikelihood;
	}
	
	protected double inferenceParentDoc(_ParentDoc pDoc){
		double likelihood = 0;
		int iter = 0;
		do{
			
			calculate_E_step(pDoc);
			
			if(iter>m_burnIn && iter%m_lag==0){
				collectParentStats(pDoc);
			}
			
		}while(++iter<number_of_iteration);
		
		return likelihood;
	}
	
	protected double inferenceChildDoc(_ParentDoc pDoc){
		double likelihood = 0;
		int iter = 0;
		
		do{
			int t;
			_ChildDoc tmpDoc;
			for(int i=pDoc.m_childDocs.size()-1; i>1; i--){
				t = m_rand.nextInt(i);
				
				tmpDoc = pDoc.m_childDocs.get(i);
				pDoc.m_childDocs.set(i, pDoc.m_childDocs.get(t));
				pDoc.m_childDocs.set(t, tmpDoc);
			}
			
			for(_Doc doc:pDoc.m_childDocs){
				calculate_E_step(doc);
			}
			
			if(iter>m_burnIn && iter%m_lag==0){
				for(_Doc doc:pDoc.m_childDocs){
					collectChildStats((_ChildDoc)doc);
				}
			}
		}while(++iter<number_of_iteration);
		
		for(_Doc doc:pDoc.m_childDocs)
			likelihood += calculate_test_log_likelihood(doc);
		
		return likelihood;
	}

	protected double calculate_test_log_likelihood(_Doc d){
		if(d instanceof _ParentDoc)
			return testLogLikelihoodByIntegrateTopics((_ParentDoc)d);
		else
			return testLogLikelihoodByIntegrateTopics((_ChildDoc)d);
	}	
	
	@Override
	public void printTopWords(int k, String betaFile) {
		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < m_sstat.length; i++) {
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];	
				if (Double.isNaN(d.m_topics[i]))
					System.out.println("nan name\t" + d.getName());
			}
		}
		
		Utils.L1Normalization(m_sstat);
		
		try {
			System.out.println("beta file");
			PrintWriter betaOut = new PrintWriter(new File(betaFile));
			for (int i = 0; i < topic_term_probabilty.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j), topic_term_probabilty[i][j]));

				betaOut.format("Topic %d(%.3f):\t", i, m_sstat[i]);
				for (_RankItem it : fVector) {
					betaOut.format("%s(%.3f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
					System.out.format("%s(%.3f)\t", it.m_name,
						m_logSpace ? Math.exp(it.m_value) : it.m_value);
				}
				betaOut.println();
				System.out.println();
			}
			betaOut.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}

	@Override
	protected double calculate_log_likelihood(){
		double corpusLogLikelihood = 0;//how could we get the corpus-level likelihood?
		
		for(_Doc d: m_trainSet)
			corpusLogLikelihood += calculate_log_likelihood(d);
		
		return corpusLogLikelihood;
	}
	
	@Override
	public double calculate_log_likelihood(_Doc d){
		if (d instanceof _ParentDoc)
			return logLikelihoodByIntegrateTopics((_ParentDoc)d);
		else
			return logLikelihoodByIntegrateTopics((_ChildDoc)d);
	}
	
	protected double logLikelihoodByIntegrateTopics(_ParentDoc d) {
		double docLogLikelihood = 0.0;
		_SparseFeature[] fv = d.getSparse();

		for (int j = 0; j < fv.length; j++) {
			int wid = fv[j].getIndex();
			double value = fv[j].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)*parentTopicInDocProb(k, d)/(d.getDocInferLength()+number_of_topics*d_alpha);
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			if(Math.abs(wordLogLikelihood) < 1e-10){
				System.out.println("wordLoglikelihood\t"+wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}
			
			wordLogLikelihood = Math.log(wordLogLikelihood);

			docLogLikelihood += value * wordLogLikelihood;
		}

		return docLogLikelihood;
	}
	
	protected double logLikelihoodByIntegrateTopics(_ChildDoc d) {
		double docLogLikelihood = 0.0;

		// prepare compute the normalizers
		_SparseFeature[] fv = d.getSparse();
		
		for (int i=0; i<fv.length; i++) {
			int wid = fv[i].getIndex();
			double value = fv[i].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, d);		
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			if(Math.abs(wordLogLikelihood) < 1e-10){
				System.out.println("wordLoglikelihood\t"+wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}
			
			wordLogLikelihood = Math.log(wordLogLikelihood);
			docLogLikelihood += value * wordLogLikelihood;
		}
		
		return docLogLikelihood;
	}
		
	protected double testLogLikelihoodByIntegrateTopics(_ParentDoc d){
		double docLogLikelihood = 0.0;
		double docInferLen = d.getWords().length;
		
		for(_Word w:d.getTestWords()){
			int wid = w.getIndex();
	
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)*parentTopicInDocProb(k, d)/(docInferLen+number_of_topics*d_alpha);

				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}
		
		return docLogLikelihood;
	}
	
	protected double testLogLikelihoodByIntegrateTopics(_ChildDoc d){
		double docLogLikelihood = 0.0;
		double docInferLen = d.getWords().length;
		
		for(_Word w:d.getTestWords()){
			int wid = w.getIndex();
	
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, d);

				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}
		
		return docLogLikelihood;
	}
}
