//package topicmodels;
//
//import java.io.File;
//import java.io.PrintWriter;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.Collection;
//
//import structures._ChildDoc;
//import structures._Corpus;
//import structures._Doc;
//import structures._ParentDoc;
//import structures._SparseFeature;
//import structures._Stn;
//import structures._Word;
//import topicmodels.ParentChild_Gibbs.MatchPair;
//import utils.Utils;
//
//public class ParentChildWith1TopicProportionTwoPhi extends ParentChild_Gibbs{
//	public ParentChildWith1TopicProportionTwoPhi(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
//			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double mu) {
//		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, mu);
//		// TODO Auto-generated constructor stub
//	}
//	
//	public String toString(){
//		return String.format("ParentChildWith1TopicProportionTwoPhi [k:%d, alpha:%.2f, beta:%.4f, Gibbs Sampling]", 
//				number_of_topics, d_alpha, d_beta);
//	}
//	
//	protected void initialize_probability(Collection<_Doc> collection){
//		for(int i=0; i<number_of_topics; i++)
//			Arrays.fill(word_topic_sstat[i], d_beta);
//		Arrays.fill(m_sstat, d_beta*vocabulary_size); // avoid adding such prior later on
//		
//		for(_Doc d:collection){
//			if(d instanceof _ParentDoc){
//				for(_Stn stnObj: d.getSentences())
//					stnObj.setTopicsVct(number_of_topics);				
//			} else if(d instanceof _ChildDoc4OneTopicProportion){
//				((_ChildDoc4OneTopicProportion) d).createXSpace(number_of_topics, m_gamma.length);
//				((_ChildDoc4OneTopicProportion) d).createLocalWordTopicDistribution(this.vocabulary_size, d_beta);
//				computeMu4Doc((_ChildDoc4OneTopicProportion)d);
//			}
//			
//			d.setTopics4Gibbs(number_of_topics, 0);
//			if(d instanceof _ParentDoc){
//				for (_Word w:d.getWords()) {
//					word_topic_sstat[w.getTopic()][w.getIndex()]++;
//					m_sstat[w.getTopic()]++;
//				}
//			}else if(d instanceof _ChildDoc4OneTopicProportion){
//				for (_Word w:d.getWords()) {
//					if(w.getX()==0){
//						word_topic_sstat[w.getTopic()][w.getIndex()]++;
//						m_sstat[w.getTopic()]++;
//					}
//				}
//			}			
//		}
//		
//		imposePrior();
//		
//		m_statisticsNormalized = false;
//	
//	}
//	
//	protected void computeMu4Doc(_ChildDoc4OneTopicProportion d){
//		_ParentDoc tempParent =  d.m_parentDoc;
//		double mu = Utils.cosine_values(tempParent.getSparse(), d.getSparse());
//		d.setMu(mu);
//	}
//	
//	protected double parentChildInfluenceProb(int tid, _ParentDoc pDoc){
////		return super.parentChildInfluenceProb(tid, pDoc);
//		double term = 1.0;
//		
//		if (tid==0)
//			return term;//reference point
//		for (_ChildDoc cDoc : pDoc.m_childDocs) {
//			double muDp = cDoc.getMu() / pDoc.getTotalDocLength();
//			term *= gammaFuncRatio(cDoc.m_xTopicSstat[0][tid], muDp, d_alpha+pDoc.m_sstat[tid]*muDp) 
//					/ gammaFuncRatio(cDoc.m_xTopicSstat[0][0], muDp, d_alpha+pDoc.m_sstat[0]*muDp);		
//		} 
//		return term;
//	}
//	
//	void sampleInChildDoc(_ChildDoc d){
////		super.sampleInChildDoc(d);
//		int wid, tid, xid;
//		double normalizedProb;
//		
//		if(d.m_xSstat[1]!=d.m_xTopicSstat[1][0]){
//			System.out.println("not aligned to each other");
//		}
//		
//		for(_Word w: d.getWords()){
//			wid = w.getIndex();
//			tid = w.getTopic();
//			xid = w.getX();	
//			
//			d.m_xTopicSstat[xid][tid] --;
//			d.m_xSstat[xid] --;
//			
//			if(xid == 1){
//				((_ChildDoc4OneTopicProportion)d).m_localWordSstat[wid] --;
//				((_ChildDoc4OneTopicProportion)d).m_localWord --;
//			}
//			
//			if(m_collectCorpusStats){
//				if(xid == 0){
//					word_topic_sstat[tid][wid] --;
//					m_sstat[tid] --;
//				}
//			}
//			
//			normalizedProb = 0;
//			
//			double pLambdaOne = childXInDocProb(1, d);
//			double pLambdaZero = childXInDocProb(0, d);
//			
//			for(tid=0; tid<number_of_topics; tid++){
//				double pWordTopicGlobal = childWordByTopicProb(tid, wid);
//				
//				double pTopicGlobal = childTopicInDocProb(tid, 0, d);
//				m_xTopicProbCache[0][tid] = pWordTopicGlobal*pTopicGlobal*pLambdaZero;
//				normalizedProb += m_xTopicProbCache[0][tid];
//			}
//			
//			double pWordTopicLocal = localChildWordByTopicProb(wid, (_ChildDoc4OneTopicProportion)d);
//			m_xTopicProbCache[1][0] = pWordTopicLocal*pLambdaOne;
//			normalizedProb += m_xTopicProbCache[1][0];
//			
//			normalizedProb *= m_rand.nextDouble();
//			xid = 0;
//			for(tid=0; tid<number_of_topics; tid++){
//				normalizedProb -= m_xTopicProbCache[0][tid];
//				if(normalizedProb <= 0)
//					break;
//			}
//			
//			if(normalizedProb >0){
//				normalizedProb -= m_xTopicProbCache[1][0];
//				tid = 0; 
//				xid = 1;
//			}
//			
//			if(tid == number_of_topics)
//				tid --;
//			
//			w.setTopic(tid);
//			w.setX(xid);
//			
//			d.m_xTopicSstat[xid][tid] ++;
//			d.m_xSstat[xid] ++;
//			
//			if(xid == 1){
//				((_ChildDoc4OneTopicProportion)d).m_localWordSstat[wid] ++;
//				((_ChildDoc4OneTopicProportion)d).m_localWord ++;
//			}
//			
//			if(m_collectCorpusStats){
//				if(xid ==0){
//					word_topic_sstat[tid][wid] ++;
//					m_sstat[tid] ++;
//				}
//			}
//		}
//		
//	}
//	
//	protected double localChildWordByTopicProb(int wid, _ChildDoc d){
////		return d.m_localWordSstat[wid]/(d.m_xSstat[1]+vocabulary_size*d_beta*0.1);
//		return ((_ChildDoc4OneTopicProportion)d).m_localWordSstat[wid]/((_ChildDoc4OneTopicProportion)d).m_localWord;
//	}
//		
//	protected double childTopicInDocProb(int tid, int xid, _ChildDoc d){
////		return super.childTopicInDocProb(tid, xid, d);
//		double docLength = d.m_parentDoc.getTotalDocLength();
//
//		if(xid == 1){//local topics
//			return (d_alpha + d.m_xTopicSstat[1][tid])
//					/(m_kAlpha + d.m_xSstat[1]);
//		} else if(xid == 0){//global topics
//			return (d_alpha + d.getMu()*d.m_parentDoc.m_sstat[tid]/docLength + d.m_xTopicSstat[0][tid])
//					/(m_kAlpha + d.getMu() + d.m_xSstat[0]);
//		} else
//			return Double.NaN;//this branch is impossible
//	}
//	
//	protected void collectChildStats(_ChildDoc d) {
//		for (int j = 0; j < m_gamma.length; j++)
//			d.m_xProportion[j] += d.m_xSstat[j] + m_gamma[j];
//		
//		double parentDocLength = d.m_parentDoc.getTotalDocLength()/d.getMu(), gTopic, lTopic;
//		// used to output the topK words and parameters
//		for (int k = 0; k < number_of_topics; k++) {		
//			lTopic = d.m_xTopicSstat[1][k] + d_alpha;
//			gTopic = d.m_xTopicSstat[0][k] + d_alpha + d.m_parentDoc.m_sstat[k] / parentDocLength;
//			d.m_xTopics[1][k] += lTopic;
//			d.m_xTopics[0][k] += gTopic;
//			d.m_topics[k] += gTopic + lTopic; // this is just an approximation
//		}
//		
//		((_ChildDoc4OneTopicProportion)d).collectLocalWordSstat();
//		
//		for(_Word w:d.getWords())
//			w.collectXStats();
//	}
//	
//	public double inference(_Doc pDoc){
//		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();
//		
//		for(_Stn stnObj: pDoc.getSentences()){
//			stnObj.setTopicsVct(number_of_topics);
//		}
//		pDoc.setTopics4Gibbs(number_of_topics, 0);		
//		sampleTestSet.add(pDoc);
//		
//		for(_ChildDoc cDoc: ((_ParentDoc)pDoc).m_childDocs){
//			((_ChildDoc4OneTopicProportion) cDoc).createXSpace(number_of_topics, m_gamma.length);
//			((_ChildDoc4OneTopicProportion) cDoc).createLocalWordTopicDistribution(this.vocabulary_size, d_beta);
//			((_ChildDoc4OneTopicProportion) cDoc).setTopics4Gibbs(number_of_topics, 0);
//			computeMu4Doc((_ChildDoc4OneTopicProportion) cDoc);
//			sampleTestSet.add(cDoc);
//		}
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
////					double likelihood1 = Math.exp(tempLogLikelihood);
////					double likelihood2 = Math.exp(logLikelihood);
////					logLikelihood = Math.log(likelihood1+likelihood2);
//					logLikelihood = Utils.logSum(logLikelihood, tempLogLikelihood);
//				}
////					logLikelihood = Utils.logSum(logLikelihood, tempLogLikelihood);
//			}
//		} while (++iter<this.number_of_iteration);
//
//		for(_Doc doc: sampleTestSet){
//			if(doc instanceof _ParentDoc)
//				estThetaInDoc((_ParentDoc)doc);
//			else if(doc instanceof _ChildDoc)
//				estThetaInDoc((_ChildDoc)doc);
//		}
//		
//		return logLikelihood - Math.log(count); 	
//	}
//
//	public void debugOutput(String filePrefix){
//
//		File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
//		File childTopicFolder = new File(filePrefix + "childTopicAssignment");
//		File childLocalWordTopicFolder = new File(filePrefix+ "childLocalTopic");
//		
//		if (!parentTopicFolder.exists()) {
//			System.out.println("creating directory" + parentTopicFolder);
//			parentTopicFolder.mkdir();
//		}
//		if (!childTopicFolder.exists()) {
//			System.out.println("creating directory" + childTopicFolder);
//			childTopicFolder.mkdir();
//		}
//		if (!childLocalWordTopicFolder.exists()) {
//			System.out.println("creating directory" + childLocalWordTopicFolder);
//			childLocalWordTopicFolder.mkdir();
//		}
//		
//		File parentPhiFolder = new File(filePrefix + "parentPhi");
//		File childPhiFolder = new File(filePrefix + "childPhi");
//		if (!parentPhiFolder.exists()) {
//			System.out.println("creating directory" + parentPhiFolder);
//			parentPhiFolder.mkdir();
//		}
//		if (!childPhiFolder.exists()) {
//			System.out.println("creating directory" + childPhiFolder);
//			childPhiFolder.mkdir();
//		}
//		
//		File childXFolder = new File(filePrefix+"xValue");
//		if(!childXFolder.exists()){
//			System.out.println("creating x Value directory" + childXFolder);
//			childXFolder.mkdir();
//		}
//
//		for (_Doc d : m_corpus.getCollection()) {
//		if (d instanceof _ParentDoc) {
//				printTopicAssignment(d, parentTopicFolder);
//				printParentPhi((_ParentDoc)d, parentPhiFolder);
//			} else if (d instanceof _ChildDoc) {
//				printTopicAssignment(d, childTopicFolder);
//				printChildXValue(d, childXFolder);
//				printChildLocalWordTopicDistribution((_ChildDoc4OneTopicProportion)d, childLocalWordTopicFolder);
//			}
//
//		}
//
//		String parentParameterFile = filePrefix + "parentParameter.txt";
//		String childParameterFile = filePrefix + "childParameter.txt";
//		printParameter(parentParameterFile, childParameterFile);
//
//		String similarityFile = filePrefix+"topicSimilarity.txt";
//		discoverSpecificComments(MatchPair.MP_ChildDoc, similarityFile);
//		
//		printEntropy(filePrefix);
//	}
//	
//	public void printChildLocalWordTopicDistribution(_ChildDoc4OneTopicProportion d, File childLocalTopicDistrifolder){
////		System.out.println("printing local word topic distribution");
//		
//		String childLocalTopicDistriFile = d.getName() + ".txt";
//		try{
////			System.out.println(childLocalTopicDistrifolder);
//			
//			PrintWriter childOut = new PrintWriter(new File(childLocalTopicDistrifolder, childLocalTopicDistriFile));
//			
//			for(int wid=0; wid<this.vocabulary_size; wid++){
//				String featureName = m_corpus.getFeature(wid);
//				double wordTopicProb = d.m_localWordProb[wid];
//				if(wordTopicProb > 0.001)
//					childOut.format("%s:%.3f\t", featureName, wordTopicProb);
//			}
//			childOut.flush();
//			childOut.close();
//			
//		}catch (Exception e) {
//			e.printStackTrace();
//		}
//		
//	}
//	
//	
//	protected double logLikelihoodByIntegrateTopics(_ParentDoc d) {
//		double docLogLikelihood = 0.0;
//		_SparseFeature[] fv = d.getSparse();
//
//		for (int j = 0; j < fv.length; j++) {
//			int index = fv[j].getIndex();
//			double value = fv[j].getValue();
//
//			double wordLogLikelihood = 0;
//			for (int k = 0; k < number_of_topics; k++) {
//				double wordPerTopicLikelihood = parentWordByTopicProb(k, index)*parentTopicInDocProb(k, d)/(d.getTotalDocLength()+number_of_topics*d_alpha);
//				wordLogLikelihood += wordPerTopicLikelihood;
//				
////				double wordPerTopicLikelihood = Math.log(parentWordByTopicProb(k, index))+Math.log(parentTopicInDocProb(k, d)/(d.getTotalDocLength()+number_of_topics*d_alpha));
////				
////				if(wordLogLikelihood == 0)
////					wordLogLikelihood = wordPerTopicLikelihood;
////				else
////					wordLogLikelihood = Utils.logSum(wordLogLikelihood, wordPerTopicLikelihood);
//			}
//			
//			if(Math.abs(wordLogLikelihood) < 1e-10){
//				System.out.println("wordLoglikelihood\t"+wordLogLikelihood);
//				wordLogLikelihood += 1e-10;
//			}
//
//			wordLogLikelihood = Math.log(wordLogLikelihood);
//			docLogLikelihood += value * wordLogLikelihood;
//		}
//
//		return docLogLikelihood;
//	}
//	
//	protected double logLikelihoodByIntegrateTopics(_ChildDoc d) {
//		double docLogLikelihood = 0.0;
//
//		// prepare compute the normalizers
//		_SparseFeature[] fv = d.getSparse();
//		
//		for (int i=0; i<fv.length; i++) {
//			int wid = fv[i].getIndex();
//			double value = fv[i].getValue();
//
//			double wordLogLikelihood = 0;
//			
//			for (int k = 0; k < number_of_topics; k++) {
//				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, 0, d)*childXInDocProb(0, d)/(d.getTotalDocLength()+m_gamma[0]+m_gamma[1]);
//				wordLogLikelihood += wordPerTopicLikelihood;
////				double wordPerTopicLikelihood = Math.log(childWordByTopicProb(k, wid))+Math.log(childTopicInDocProb(k, 0, d))+Math.log(childXInDocProb(0, d))-Math.log(d.getTotalDocLength()+m_gamma[0]+m_gamma[1]);
//		
////				if(wordLogLikelihood == 0)
////					wordLogLikelihood = wordPerTopicLikelihood;
////				else
////					wordLogLikelihood = Utils.logSum(wordLogLikelihood, wordPerTopicLikelihood);
//			}
//			
////			double localWordLikelihood = Math.log(localChildWordByTopicProb(wid, d))+Math.log(childXInDocProb(1, d))-Math.log(d.getTotalDocLength()+m_gamma[0]+m_gamma[1]);
//			
////			wordLogLikelihood = Utils.logSum(wordLogLikelihood, localWordLikelihood);
//			
//			wordLogLikelihood += localChildWordByTopicProb(wid, d)*childXInDocProb(1, d)/(d.getTotalDocLength()+m_gamma[0]+m_gamma[1]);
//			
//			if(Math.abs(wordLogLikelihood) < 1e-10){
//				System.out.println("wordLoglikelihood\t"+wordLogLikelihood);
//				wordLogLikelihood += 1e-10;
//			}
//			
//			wordLogLikelihood = Math.log(wordLogLikelihood);
//	
//			docLogLikelihood += value * wordLogLikelihood;
//		}
//		
//		return docLogLikelihood;
//	}
//	
//	
//}
