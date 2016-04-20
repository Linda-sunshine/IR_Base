package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import structures.MyPriorityQueue;
import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._ChildDoc4ThreePhi;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._RankItem;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class ParentChildBase_Gibbs extends LDA_Gibbs_Debug{
	
	protected double[] m_topicProbCache;
	protected double m_kAlpha;
	
	protected boolean m_statisticsNormalized = false;//a warning sign of normalizing statistics before collecting new ones
	
	public ParentChildBase_Gibbs(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double ksi, double tau) {
		// TODO Auto-generated constructor stub
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, ksi, tau);
		
		m_topicProbCache = new double[number_of_topics];
		
		m_kAlpha = d_alpha * number_of_topics;
		
		m_topicProbCache = new double[number_of_topics];
	
	}
	
	@Override
	public String toString(){
		return String.format("Parent Child Base topic model [k:%d, alpha:%.2f, beta:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta);
	}
	
	protected void initialize_probability(Collection<_Doc> collection){
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size); // avoid adding such prior later on
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				d.setTopics4Gibbs(number_of_topics, 0);
				for(_Stn stnObj: d.getSentences())
					stnObj.setTopic(number_of_topics);				
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
		double mu = Utils.cosine_values(tempParent.getSparse(), d.getSparse());
//		mu = 0.001;
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
			double muDp = cDoc.getMu()/pDoc.getTotalDocLength();
			term *= gammaFuncRatio((int)cDoc.m_sstat[tid], muDp, d_alpha+pDoc.m_sstat[tid]*muDp)
					/ gammaFuncRatio((int)cDoc.m_sstat[0], muDp, d_alpha+pDoc.m_sstat[0]*muDp);
		}
		
		return term;
	}
	
	protected double gammaFuncRatio(int nc, double muDp, double alphaMuNp) {
		if (nc==0)
			return 1.0;
		
		double result = 1.0;
		for(int n=1; n<=nc; n++) 
			result *= 1 + muDp / (alphaMuNp + n);
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
		double docLength = d.m_parentDoc.getTotalDocLength();
		
		return (d_alpha + d.getMu()*d.m_parentDoc.m_sstat[tid]/docLength + d.m_sstat[tid])
					/(m_kAlpha + d.getMu() + d.getTotalDocLength());
	
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
	
	//such statistic collection mechanism makes us unable to normalize the corresponding structure for efficient likelihood computation
	protected void collectParentStats(_ParentDoc d) {
		for (int k = 0; k < this.number_of_topics; k++) 
			d.m_topics[k] += d.m_sstat[k] + d_alpha;
		d.collectTopicWordStat();
	}
	
	protected void collectChildStats(_ChildDoc d) {
		_ParentDoc pDoc = d.m_parentDoc;
		double parentDocLength = pDoc.getTotalDocLength();
		for (int k = 0; k < this.number_of_topics; k++) 
			d.m_topics[k] += d.m_sstat[k] + d_alpha;
//		+d.getMu()*pDoc.m_sstat[k] / parentDocLength;

//			d.m_topics[k] += d.m_sstat[k] + d_alpha+d.getMu()*pDoc.m_sstat[k] / parentDocLength;
	}
	
	protected void estThetaInDoc(_Doc d){
		Utils.L1Normalization(d.m_topics);
		if(d instanceof _ParentDoc){
			estParentStnTopicProportion((_ParentDoc)d);
		}
	}
	
	double computeSimilarity(double[] topic1, double[] topic2) {
		return Utils.cosine(topic1, topic2);
	}
	
	
//	
//	@Override
//	public double inference(_Doc pDoc){
//		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();
//		
//		initTest(sampleTestSet, pDoc);
//	
//		double logLikelihood = 0.0, count = 0;
//		int  iter = 0;
//		do {
//			int t;
//			_Doc tmpDoc;
//			for(int i=sampleTestSet.size()-1; i>1; i--) {
//				t = m_rand.nextInt(i);
//				
//				tmpDoc = sampleTestSet.get(i);
//				sampleTestSet.set(i, sampleTestSet.get(t));
//				sampleTestSet.set(t, tmpDoc);			
//			}
//			
//			for(_Doc doc: sampleTestSet)
//				calculate_E_step(doc);
//			
//			if (iter>m_burnIn && iter%m_lag==0){
//				double tempLogLikelihood = 0;
//				for(_Doc doc: sampleTestSet){
//					if(doc instanceof _ParentDoc){
//						collectParentStats((_ParentDoc) doc);
//						tempLogLikelihood += calculate_log_likelihood((_ParentDoc) doc);
//					}
//					else if(doc instanceof _ChildDoc){
//						collectChildStats((_ChildDoc) doc);
//						tempLogLikelihood += calculate_log_likelihood((_ChildDoc) doc);
//					}
//					
//				}
//				count ++;
//				if(logLikelihood == 0)
//					logLikelihood = tempLogLikelihood;
//				else{
//
//					logLikelihood = Utils.logSum(logLikelihood, tempLogLikelihood);
//				}
//			}
//		} while (++iter<this.number_of_iteration);
//
//		for(_Doc doc: sampleTestSet){
//			estThetaInDoc(doc);
//		}
//		
//		return logLikelihood - Math.log(count); 	
//	}

	//to make it consistent, we will not assume the statistic collector has been normalized before calling this function
	@Override
	protected double calculate_log_likelihood(){
		double corpusLogLikelihood = 0;//how could we get the corpus-level likelihood?
		
		for(_Doc d: m_trainSet)
			corpusLogLikelihood += calculate_log_likelihood(d);
		
		return corpusLogLikelihood;
	}
	
//	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
//		_ParentDoc pDoc = (_ParentDoc)d;
//		for(_Stn stnObj: pDoc.getSentences()){
//			stnObj.setTopicsVct(number_of_topics);
//		}
//		pDoc.setTopics4Gibbs(number_of_topics, 0);		
//		sampleTestSet.add(pDoc);
//		
//		for(_ChildDoc cDoc: pDoc.m_childDocs){
//			cDoc.setTopics4Gibbs_LDA(number_of_topics, 0);
//			sampleTestSet.add(cDoc);
//		}
//	}
	
	public void debugOutput(String filePrefix){

		File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
		File childTopicFolder = new File(filePrefix + "childTopicAssignment");
		if (!parentTopicFolder.exists()) {
			System.out.println("creating directory" + parentTopicFolder);
			parentTopicFolder.mkdir();
		}
		if (!childTopicFolder.exists()) {
			System.out.println("creating directory" + childTopicFolder);
			childTopicFolder.mkdir();
		}
		
		File parentPhiFolder = new File(filePrefix + "parentPhi");
		File childPhiFolder = new File(filePrefix + "childPhi");
		if (!parentPhiFolder.exists()) {
			System.out.println("creating directory" + parentPhiFolder);
			parentPhiFolder.mkdir();
		}
		if (!childPhiFolder.exists()) {
			System.out.println("creating directory" + childPhiFolder);
			childPhiFolder.mkdir();
		}
		
		File childXFolder = new File(filePrefix+"xValue");
		if(!childXFolder.exists()){
			System.out.println("creating x Value directory" + childXFolder);
			childXFolder.mkdir();
		}

		for (_Doc d : m_corpus.getCollection()) {
		if (d instanceof _ParentDoc) {
				printTopicAssignment((_ParentDoc)d, parentTopicFolder);
				printParentPhi((_ParentDoc)d, parentPhiFolder);
			} else if (d instanceof _ChildDoc) {
				printTopicAssignment(d, childTopicFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParameter(parentParameterFile, childParameterFile);

		String similarityFile = filePrefix+"topicSimilarity.txt";
		
		printEntropy(filePrefix);
		
		int topKStn = 10;
		int topKChild = 10;
		printTopKChild4StnWithHybrid(filePrefix, topKChild);
		printTopKChild4Stn(filePrefix, topKChild);
		
		printTopKStn4Child(filePrefix, topKStn);
		
		printTopKChild4Parent(filePrefix, topKChild);
		
		String childMuFile = filePrefix+"childMu.txt";
		printMu(childMuFile);
	}
	
	//used to print test parameter
	public void printTestParameter(String parentParameterFile, String childParameterFile){
		System.out.println("printing parameter");
		try{
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			for(_Doc d: m_testSet){
				
				parentParaOut.print(d.getName()+"\t");
				parentParaOut.print("topicProportion\t");
				for(int k=0; k<number_of_topics; k++){
					parentParaOut.print(d.m_topics[k]+"\t");
				}
				
				for(_Stn stnObj:d.getSentences()){							
					parentParaOut.print("sentence"+(stnObj.getIndex()+1)+"\t");
					for(int k=0; k<number_of_topics;k++){
						parentParaOut.print(stnObj.m_topics[k]+"\t");
					}
				}
				
				parentParaOut.println();
				
				for(_ChildDoc cDoc: ((_ParentDoc)d).m_childDocs){
					childParaOut.print(cDoc.getName()+"\t");

					childParaOut.print("topicProportion\t");
					for (int k = 0; k < number_of_topics; k++) {
						childParaOut.print(cDoc.m_topics[k] + "\t");
					}
					
					childParaOut.println();
					
				}
			}
			
			parentParaOut.flush();
			parentParaOut.close();
			
			childParaOut.flush();
			childParaOut.close();
		}
		catch (Exception e) {
			e.printStackTrace();
//				e.printStackTrace();
//				System.err.print("para File Not Found");
		}
	}
	
		@Override
	public void printTopWords(int k, String betaFile) {
		
		String filePrefix = betaFile.replace("topWords.txt", "");
		debugOutput(filePrefix);
		
		double loglikelihood = 0.0;
		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		for (_Doc d : m_trainSet) {
//				loglikelihood += calculate_log_likelihood(d);
			for (int i = 0; i < m_sstat.length; i++) {
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];	
				if (Double.isNaN(d.m_topics[i]))
					System.out.println("nan name\t" + d.getName());
			}
		}
		
		Utils.L1Normalization(m_sstat);

		System.out.format("Final Log Likelihood %.3f\t", loglikelihood);
		
		try {
			System.out.println("beta file");
			PrintWriter betaOut = new PrintWriter(new File(betaFile));
			for (int i = 0; i < topic_term_probabilty.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							topic_term_probabilty[i][j]));

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
	
			betaOut.flush();
			betaOut.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	
	}
	
	public void printTopicAssignment(_ParentDoc d, File parentFolder) {
	//	System.out.println("printing topic assignment parent documents");
		
		String topicAssignmentFile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(parentFolder,
					topicAssignmentFile));
			
			for(_Stn stnObj: d.getSentences()){
				for(_Word w: stnObj.getWords()){
					int index = w.getIndex();
					int topic = w.getTopic();
					String featureName = m_corpus.getFeature(index);
					pw.print(featureName + ":" + topic + "\t");
				}
				pw.println();
			}
				
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}

	public void printParentPhi(_ParentDoc d, File phiFolder){
		String parentPhiFileName = d.getName()+".txt";
		_SparseFeature[] fv = d.getSparse();
		
		try{
			PrintWriter parentPW = new PrintWriter(new File(phiFolder, parentPhiFileName));
		
			for(int n=0; n<fv.length; n++){
				int index = fv[n].getIndex();
				String featureName = m_corpus.getFeature(index);
				parentPW.print(featureName + ":\t");
				for(int k=0; k<number_of_topics; k++)
					parentPW.print(d.m_phi[n][k]+"\t");
				parentPW.println();
			}
			parentPW.flush();
			parentPW.close();
		}catch(Exception ex){
			ex.printStackTrace();
		}
	}
	
	protected void printEntropy(String filePrefix){
		String entropyFile = filePrefix+"entropy.txt";
		boolean logScale = true;
		
		try{
			PrintWriter entropyPW = new PrintWriter(new File(entropyFile));
			
			for(_Doc d: m_corpus.getCollection()){
				double entropyValue = 0.0;
				entropyValue = Utils.entropy(d.m_topics, logScale);
				entropyPW.print(d.getName()+"\t"+entropyValue);
				entropyPW.println();
			}
			entropyPW.flush();
			entropyPW.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}	
	
	protected void printTopKStn4Child(String filePrefix, int topK){
		String topKStn4ChildFile = filePrefix+"topStn4Child.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKStn4ChildFile));
			
			for(_Doc d: m_corpus.getCollection()){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.println(pDoc.getName()+"\t"+pDoc.m_childDocs.size());
					
					for(_ChildDoc childDoc: pDoc.m_childDocs){
//							_ChildDoc4ThreePhi cDoc = (_ChildDoc4ThreePhi)childDoc;
						HashMap<Integer, Double>stnSimMap = rankStn4ChildBySim(pDoc, childDoc);
						int i = 0;
						
						pw.print(childDoc.getName()+"\t");
						for(Map.Entry<Integer, Double> e: sortHashMap4Integer(stnSimMap, true)){
//								if(i==topK)
//									break;
							pw.print(e.getKey());
							pw.print(":"+e.getValue());
							pw.print("\t");
							
							i++;
						}
						pw.println();
					}
				}
			}
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	protected void printTopKChild4Parent(String filePrefix, int topK) {
		String topKChild4StnFile = filePrefix+"topChild4Parent.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));
			
			for(_Doc d: m_corpus.getCollection()){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.print(pDoc.getName()+"\t");
					
					for(_ChildDoc childDoc:pDoc.m_childDocs){
//							_ChildDoc4ThreePhi cDoc = (_ChildDoc4ThreePhi) childDoc;
						double docLogLikelihood = rankChild4ParentByLikelihood(childDoc, pDoc);
				
						pw.print(childDoc.getName()+":"+docLogLikelihood+"\t");	
					}
					
					pw.println();
				}
			}
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}
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
				double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)*parentTopicInDocProb(k, d)/(d.getTotalDocLength()+number_of_topics*d_alpha);
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
		_SparseFeature[] fv = d.getSparse();
		
		for(_Word w:d.getTestWords()){
			int wid = w.getIndex();
	
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)*parentTopicInDocProb(k, d)/(d.getTotalDocLength()+number_of_topics*d_alpha);

				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}
		
		return docLogLikelihood;
	}
	
	protected double testLogLikelihoodByIntegrateTopics(_ChildDoc d){
		double docLogLikelihood = 0.0;
		_SparseFeature[] fv = d.getSparse();
		
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
	
	protected HashMap<Integer, Double> rankStn4ChildBySim( _ParentDoc pDoc, _ChildDoc cDoc){

		HashMap<Integer, Double> stnSimMap = new HashMap<Integer, Double>();
		
		for(_Stn stnObj:pDoc.getSentences()){
//			double stnSim = computeSimilarity(cDoc.m_topics, stnObj.m_topics);
//			stnSimMap.put(stnObj.getIndex()+1, stnSim);

			double stnKL = Utils.klDivergence(cDoc.m_topics, stnObj.m_topics);
//			double stnKL = Utils.KLsymmetric(cDoc.m_topics, stnObj.m_topics);
//			double stnKL = Utils.klDivergence(stnObj.m_topics, cDoc.m_topics);

			stnSimMap.put(stnObj.getIndex()+1, -stnKL);

		}
		
		return stnSimMap;
	}
	
	protected HashMap<String, Double> rankChild4StnByHybrid(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		
		double smoothingMu = m_LM.m_smoothingMu;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			double cDocLen = cDoc.getTotalDocLength();
			_SparseFeature[] fv = cDoc.getSparse();
			
			double stnLogLikelihood = 0;
			double alphaDoc = smoothingMu/(smoothingMu+cDocLen);
			
			_SparseFeature[] sv = stnObj.getFv();
			for(_SparseFeature svWord:sv){
				double featureLikelihood = 0;
				
				int wid = svWord.getIndex();
				double stnVal = svWord.getValue();
				
				int featureIndex = Utils.indexOf(fv, wid);
				double docVal = 0;
				if(featureIndex!=-1){
					docVal = fv[featureIndex].getValue();
				}
				
				double LMLikelihood = (1-alphaDoc)*docVal/(cDocLen);
				
				LMLikelihood += alphaDoc*m_LM.getReferenceProb(wid);
				
				double TMLikelihood = 0;
				for(int k=0; k<number_of_topics; k++){
					TMLikelihood += (wordByTopicProb(k, wid))*(childTopicInDocProb(k, cDoc));
				}
				
				featureLikelihood = m_tau*LMLikelihood+(1-m_tau)*TMLikelihood;
//				featureLikelihood = TMLikelihood;
				featureLikelihood = Math.log(featureLikelihood);
				stnLogLikelihood += stnVal*featureLikelihood;
				
			}
			
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj, _ParentDoc pDoc){
		
		HashMap<String, Double>childLikelihoodMap = new HashMap<String, Double>();
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			
			double stnLogLikelihood = 0;
			for(_Word w: stnObj.getWords()){
				int wid = w.getIndex();
			
				double wordLogLikelihood = 0;
				
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
				
				stnLogLikelihood += Math.log(wordLogLikelihood);
			}
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}

	public void printMu(String childMuFile){
		System.out.println("print mu");
		try{
			PrintWriter muPW = new PrintWriter(new File(childMuFile));
			
			for(_Doc d:m_corpus.getCollection()){
				if(d instanceof _ChildDoc){
					muPW.println(d.getName()+"\t"+((_ChildDoc)d).getMu());
				}
				
			}
			muPW.flush();
			muPW.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void printParameter(String parentParameterFile, String childParameterFile){
		System.out.println("printing parameter");
		try{
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			for(_Doc d: m_corpus.getCollection()){
				if(d instanceof _ParentDoc){
					parentParaOut.print(d.getName()+"\t");
					parentParaOut.print("topicProportion\t");
					for(int k=0; k<number_of_topics; k++){
						parentParaOut.print(d.m_topics[k]+"\t");
					}
					
					for(_Stn stnObj:d.getSentences()){							
						parentParaOut.print("sentence"+(stnObj.getIndex()+1)+"\t");
						for(int k=0; k<number_of_topics;k++){
							parentParaOut.print(stnObj.m_topics[k]+"\t");
						}
					}
					
					parentParaOut.println();
					
				}else{
					if(d instanceof _ChildDoc){
						childParaOut.print(d.getName()+"\t");

						childParaOut.print("topicProportion\t");
						for (int k = 0; k < number_of_topics; k++) {
							childParaOut.print(d.m_topics[k] + "\t");
						}
						
						childParaOut.println();
					}
				}
			}
			
			parentParaOut.flush();
			parentParaOut.close();
			
			childParaOut.flush();
			childParaOut.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}
	
}
