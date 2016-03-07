package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import cern.jet.random.Normal;
import cern.jet.random.engine.DRand;
import cern.jet.random.engine.RandomEngine;
import structures._ChildDoc;
import structures._ChildDoc4ProbitModel;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._Stn;
import topicmodels.ParentChild_Gibbs.MatchPair;

//public class computePerplexity extends ParentChild_Gibbs{
//	public Normal m_Normal;
//	public computePerplexity(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
//			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double mu){
//		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, mu);
//		RandomEngine engine = new DRand();
//		m_Normal = new Normal(0, 1, engine);
//		
//		m_mu = mu;
//		m_kAlpha = d_alpha * number_of_topics;
//
//		m_gamma = new double[gamma.length];
//		System.arraycopy(gamma, 0, m_gamma, 0, gamma.length);
//		m_topicProbCache = new double[number_of_topics];
//		m_xTopicProbCache = new double[gamma.length][number_of_topics];
//	}
//	
//	public String toString(){
//		return String.format("Parent Child topic model with probit model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]",
//				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
//	}
//	
//	protected void initialize_probability(Collection<_Doc> collection){
//		for(int i=0; i<number_of_topics; i++)
//			Arrays.fill(word_topic_sstat[i], d_beta);
//		Arrays.fill(m_sstat, d_beta*vocabulary_size); // avoid adding such prior later on
//		
//		for(_Doc d:collection){
//			if(d instanceof _ParentDoc){
//				for(_Stn stnObj: d.getSentences()){
//					stnObj.setTopicsVct(number_of_topics);
//				}
//				d.setTopics4Gibbs(number_of_topics, 0);
//			}else if(d instanceof _ChildDoc4ProbitModel){
//				((_ChildDoc4ProbitModel)d).createXSpace(number_of_topics, m_gamma.length);
////				((_ChildDoc4ProbitModel)d).setTopics4Gibbs(number_of_topics, 0, m_corpus);
//			}
//			
//			
//			for(int i=0; i<d.m_words.length; i++){
//				word_topic_sstat[d.m_topicAssignment[i]][d.m_words[i]] ++;
//				m_sstat[d.m_topicAssignment[i]] ++;
//			}
//		}
//		
//		imposePrior();
//	}
//
//	public double calculate_E_step(_Doc d){
//		d.permutation();
//		
//		if(d instanceof _ParentDoc)
//			sampleInParentDoc((_ParentDoc)d);
//		else if(d instanceof _ChildDoc)
//			sampleInChildDoc((_ChildDocProbitModel)d);
//		
//		return 0;
//	}
//	
//	void sampleInChildDoc(_ChildDocProbitModel d){
//		int wid, tid, xid;
//		double xVal = 0.0;
//		double normalizedProb;
//		
//		for(int i=0; i<d.m_words.length; i++){
//			xVal = d.m_xIndicator[i];
//			tid = d.m_topicAssignment[i];
//					
//			if(xVal > 0)
//				xid = 1;	
//			else
//				xid = 0;
//			
//			d.m_xSstat[xid] --;
//			d.m_xTopicSstat[xid][tid] --;
//			
////			xPredictiveProb(d, i);
//			xVal = xPredictiveProb(d, i);
//			d.m_xIndicator[i] = xVal;
//			
//			if(xVal > 0)
//				xid = 1;			
//			else 
//				xid = 0;
//			
//			d.m_xSstat[xid] ++;
//			d.m_xTopicSstat[xid][tid] ++;
//			
//		}
//		
//		for(int i=0; i<d.m_words.length; i++){
//			wid = d.m_words[i];
//			tid = d.m_topicAssignment[i];
//			xVal = d.m_xIndicator[i];
//			
//			if(xVal > 0)
//				xid = 1;
//			else
//				xid = 0;
//			
//			d.m_xTopicSstat[xid][tid] --;
//			d.m_xSstat[xid] --;
//			
//			if(m_collectCorpusStats){
//				word_topic_sstat[tid][wid] --;
//				m_sstat[tid] --;
//			}
//			
//			normalizedProb = 0.0;
//			
//			for(tid=0; tid<number_of_topics; tid++){
//				
//				double pWordTopic = childWordByTopicProb(tid, wid);
//				if(xid == 1){
//					//p(z=tid,x=1) from specific
//					double pTopicLocal = childTopicInDocProb(tid, 1, d);
//					m_topicProbCache[tid] = pWordTopic*pTopicLocal;
//					
//				}
//				else{
//					//p(z=tid,x=0) from background
//					double pTopicGlobal = childTopicInDocProb(tid, 0, d);	
//					m_topicProbCache[tid] = pWordTopic*pTopicGlobal;
//					
//				}
//				
//				normalizedProb += m_topicProbCache[tid];
//			}
//			
//			normalizedProb *= m_rand.nextDouble();
//			
//			for(tid=0; tid<number_of_topics; tid++){
//				normalizedProb -= m_topicProbCache[tid];
//				if(normalizedProb<=0){
//					break;
//				}
//			}
//
//			if (tid == number_of_topics)
//				tid--;
//			
//			d.m_topicAssignment[i] = tid;
//		
//			d.m_xTopicSstat[xid][tid] ++;
//			d.m_xSstat[xid] ++;
//			if(m_collectCorpusStats){
//				word_topic_sstat[tid][wid]++;
//				m_sstat[tid]++;
//			}	
//			
//		}
//	}
//	
//	public double xPredictiveProb(_ChildDocProbitModel d, int i){
//		double mu = 0.0;
//		double sigma = 0.0;
//		double gaussianVal = 0.0;
//		
//		int wid = d.m_words[i];
//		for(int n=0; n<d.m_words.length; n++){
//			if(n==i)
//				continue;
//			if(n<i)
//				mu += d.m_xIndicator[n]*d.m_fixedMuPartMap.get(wid)[n];
//			else if(n>i)
//				mu += d.m_xIndicator[n]*d.m_fixedMuPartMap.get(wid)[n-1];
//		}
//		
//		sigma = d.m_fixedSigmaPartMap.get(wid);
//		
////		Random m_rand = new Random();
////		double n1 = m_rand.nextDouble();
////		double n2 = m_rand.nextDouble();
////		
////		double gaussianVal = Math.sqrt(-2*Math.log(n2))*Math.cos(2*Math.PI*n1);
////		gaussianVal = gaussianVal*sigma + mu;
//
//		gaussianVal = m_Normal.nextDouble();
//		gaussianVal = gaussianVal*Math.sqrt(sigma) + mu;
//		return gaussianVal;
//		
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
//			cDoc.createXSpace(number_of_topics, m_gamma.length);
//			((_ChildDocProbitModel)cDoc).setTopics4Gibbs(number_of_topics, 0, m_corpus);
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
//			if (iter>m_burnIn && iter%m_lag==0){
//				for(_Doc doc: sampleTestSet){
//					if(doc instanceof _ParentDoc){
//						collectParentStats((_ParentDoc) doc);
//					}
//					else if(doc instanceof _ChildDoc){
//						collectChildStats((_ChildDoc) doc);
//					}
//				
////					System.out.println("logLikelihood\t"+logLikelihood);
//				}
//				count ++;
//			}
//		} while (++iter<this.number_of_iteration);
//		
//		for(_Doc doc:sampleTestSet){
//			estThetaInDoc(doc);
//			logLikelihood += calculate_log_likelihood(doc);
//		}
//
//		return logLikelihood; // this is average joint probability!	
//	}
//	
//	protected void printChildXValue(_ChildDocProbitModel d, File childXFolder){
//		String XValueFile = d.getName() + ".txt";
//		try {
//			PrintWriter pw = new PrintWriter(new File(childXFolder,
//					XValueFile));
//
//			for (int n = 0; n < d.m_words.length; n++) {
//				int index = d.m_words[n];
//				double xValue = d.m_xIndicator[n];
////				int topic = d.m_topicAssignment[n];
//				String featureName = m_corpus.getFeature(index);
//					
//				pw.print(featureName + ":" + xValue + "\t");
//			}
//			pw.flush();
//			pw.close();
//		} catch (FileNotFoundException e) {
//			e.printStackTrace();
//		}
//	}
//	
//	public void debugOutput(String filePrefix){
//
//		File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
//		File childTopicFolder = new File(filePrefix + "childTopicAssignment");
//		if (!parentTopicFolder.exists()) {
//			System.out.println("creating directory" + parentTopicFolder);
//			parentTopicFolder.mkdir();
//		}
//		if (!childTopicFolder.exists()) {
//			System.out.println("creating directory" + childTopicFolder);
//			childTopicFolder.mkdir();
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
//			System.out.println("creating directory" + childXFolder);
//			childXFolder.mkdir();
//		}
//
//		for (_Doc d : m_trainSet) {
//		if (d instanceof _ParentDoc) {
//				printParentTopicAssignment((_ParentDoc) d, parentTopicFolder);
//				printParentPhi((_ParentDoc)d, parentPhiFolder);
//			} else if (d instanceof _ChildDocProbitModel) {
//				printChildTopicAssignment((_ChildDocProbitModel) d, childTopicFolder);
//				printChildXValue((_ChildDocProbitModel)d, childXFolder);
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
//}
