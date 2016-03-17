//package topicmodels;
//
//import java.io.File;
//import java.io.PrintWriter;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.Collection;
//import structures._ChildDoc;
//import structures._Corpus;
//import structures._Doc;
//import structures._ParentDoc;
//import structures._Stn;
//import structures._Word;
//import topicmodels.ParentChild_Gibbs.MatchPair;
//import utils.Utils;
//
//public class ParentChildWith1TopicProportion_Gibbs extends ParentChild_Gibbs{
//	public ParentChildWith1TopicProportion_Gibbs(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
//			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double mu) {
//		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, mu);
//		// TODO Auto-generated constructor stub
//	}
//	
//	public String toString(){
//		return String.format("Parent Child topic model with 1 topic proportion [k:%d, alpha:%.2f, beta:%.4f, Gibbs Sampling]", 
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
//		return super.parentChildInfluenceProb(tid, pDoc);
////		double term = 1.0;
////		
////		if (tid==0)
////			return term;//reference point
////		for (_ChildDoc cDoc : pDoc.m_childDocs) {
////			double muDp = cDoc.getMu() / pDoc.getTotalDocLength();
////			term *= gammaFuncRatio(cDoc.m_xTopicSstat[0][tid], muDp, d_alpha+pDoc.m_sstat[tid]*muDp) 
////					/ gammaFuncRatio(cDoc.m_xTopicSstat[0][0], muDp, d_alpha+pDoc.m_sstat[0]*muDp);		
////		} 
////		return term;
//	}
//	
//	void sampleInChildDoc(_ChildDoc4OneTopicProportion d){
//		super.sampleInChildDoc(d);
////		int wid, tid, xid;
////		double normalizedProb;
////		
////		if(d.m_xSstat[1]!=d.m_xTopicSstat[1][0]){
////			System.out.println("not aligned to each other");
////		}
////		
////		for(_Word w: d.getWords()){
////			wid = w.getIndex();
////			tid = w.getTopic();
////			xid = w.getX();	
////			
////			d.m_xTopicSstat[xid][tid] --;
////			d.m_xSstat[xid] --;
////			
////			if(xid == 1){
////				d.m_localWordSstat[wid] --;
////				d.m_localWord --;
////			}
////			
////			if(m_collectCorpusStats){
////				if(xid == 0){
////					word_topic_sstat[tid][wid] --;
////					m_sstat[tid] --;
////				}
////			}
////			
////			normalizedProb = 0;
////			
////			double pLambdaOne = childXInDocProb(1, d);
////			double pLambdaZero = childXInDocProb(0, d);
////			
////			for(tid=0; tid<number_of_topics; tid++){
////				double pWordTopicGlobal = childWordByTopicProb(tid, wid);
////				
////				double pTopicGlobal = childTopicInDocProb(tid, 0, d);
////				m_xTopicProbCache[0][tid] = pWordTopicGlobal*pTopicGlobal*pLambdaZero;
////				normalizedProb += m_xTopicProbCache[0][tid];
////			}
////			
////			double pWordTopicLocal = localChildWordByTopicProb(wid, (_ChildDoc4OneTopicProportion)d);
////			m_xTopicProbCache[1][0] = pWordTopicLocal*pLambdaOne;
////			normalizedProb += m_xTopicProbCache[1][0];
////			
////			normalizedProb *= m_rand.nextDouble();
////			xid = 0;
////			for(tid=0; tid<number_of_topics; tid++){
////				normalizedProb -= m_xTopicProbCache[0][tid];
////				if(normalizedProb <= 0)
////					break;
////			}
////			
////			if(normalizedProb >0){
////				normalizedProb -= m_xTopicProbCache[1][0];
////				tid = 0; 
////				xid = 1;
////			}
////			
////			if(tid == number_of_topics)
////				tid --;
////			
////			w.setTopic(tid);
////			w.setX(xid);
////			
////			d.m_xTopicSstat[xid][tid] ++;
////			d.m_xSstat[xid] ++;
////			
////			if(xid == 1){
////				d.m_localWordSstat[wid] ++;
////				d.m_localWord ++;
////			}
////			
////			if(m_collectCorpusStats){
////				if(xid ==0){
////					word_topic_sstat[tid][wid] ++;
////					m_sstat[tid] ++;
////				}
////			}
////		}
////		
//	}
//	
////	protected double localChildWordByTopicProb(int wid, _ChildDoc4OneTopicProportion d){
////		return d.m_localWordSstat[wid]/(d.m_xSstat[1]+vocabulary_size*d_beta*0.1);
//////		return d.m_localWordSstat[wid]/d.m_localWord;
////	}
////		
////	protected double childTopicInDocProb(int tid, int xid, _ChildDoc d){
//////		return super.childTopicInDocProb(tid, xid, d);
////		double docLength = d.m_parentDoc.getTotalDocLength();
////
////		if(xid == 1){//local topics
////			return (d_alpha + d.m_xTopicSstat[1][tid])
////					/(m_kAlpha + d.m_xSstat[1]);
////		} else if(xid == 0){//global topics
////			return (d_alpha + d.getMu()*d.m_parentDoc.m_sstat[tid]/docLength + d.m_xTopicSstat[0][tid])
////					/(m_kAlpha + d.getMu() + d.m_xSstat[0]);
////		} else
////			return Double.NaN;//this branch is impossible
////	}
////	
//	protected void collectChildStats(_ChildDoc4OneTopicProportion d) {
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
////		d.collectLocalWordSstat();
//		
//		for(_Word w:d.getWords())
//			w.collectXStats();
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
//}
