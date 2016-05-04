package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures._APPQuery;
import structures._ChildDoc;
import structures._ChildDoc4APP;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class LDA_APP extends LDA_Gibbs_Debug{
	ArrayList<_APPQuery> m_APPQueries;
	
	public LDA_APP(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, 
			int number_of_topics, double alpha, double burnIn, int lag, double ksi, double tau, ArrayList<_APPQuery> queryList){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, ksi, tau);
	
		m_APPQueries = queryList;
	}
	
	protected void initialize_probability(Collection<_Doc> collection) {
		createSpace();
		for(int i=0; i< number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		
		for(_Doc d: collection){
			if(d instanceof _ParentDoc) {	
				d.setTopics4Gibbs(number_of_topics, d_alpha);
			}else if(d instanceof _ChildDoc){
				((_ChildDoc) d).setTopics4Gibbs_LDA(number_of_topics, d_alpha);
			}
				
			for(_Word w:d.getWords()) {
				word_topic_sstat[w.getTopic()][w.getIndex()] ++;
				m_sstat[w.getTopic()] ++;
			}
		}
		
		imposePrior();
	}
	
	protected void estThetaInDoc(_Doc d) {
		Utils.L1Normalization(d.m_topics);
	}
	
	protected void collectStats(_Doc d) {
		for(int k=0; k<this.number_of_topics; k++)
			d.m_topics[k] += d.m_sstat[k];
	}
	
	public void debugOutput(String filePrefix){

		File topicFolder = new File(filePrefix + "topicAssignment");
	
		if (!topicFolder.exists()) {
			System.out.println("creating directory" + topicFolder);
			topicFolder.mkdir();
		}
		
		for (_Doc d : m_trainSet) {
			if(d instanceof _ParentDoc){
				printParentTopicAssignment(d, topicFolder);
			}else if(d instanceof _ChildDoc){
				printChildTopicAssignment(d, topicFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
	
		printParentParameter(parentParameterFile);
		
		printEntropy(filePrefix);
		
		m_LM.generateReferenceModel();
		printAPP4QueryByTopicModel(filePrefix);
		printAPP4QueryByHybrid(filePrefix);
	}
	
	public void printParentTopicAssignment(_Doc d, File topicFolder) {
		//	System.out.println("printing topic assignment parent documents");
			
		String topicAssignmentFile = d.getName() + ".txt";
		try {

			PrintWriter pw = new PrintWriter(new File(topicFolder,
					topicAssignmentFile));

			for (_Word w : d.getWords()) {
				int index = w.getIndex();
				int topic = w.getTopic();
				String featureName = m_corpus.getFeature(index);
//							System.out.println("test\t"+featureName+"\tdocName\t"+d.getName());			
				pw.print(featureName + ":" + topic + "\t");
			}
			pw.println();
			
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	protected void printParentParameter(String parentParameterFile){
		System.out.println("printing parent parameter");
		try{
			System.out.println(parentParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			for(_Doc d: m_corpus.getCollection()){
				if(d instanceof _ParentDoc){
					parentParaOut.print(d.getName()+"\t");
					parentParaOut.print("topicProportion\t");
					for(int k=0; k<number_of_topics; k++){
						parentParaOut.print(d.m_topics[k]+"\t");
					}
					
					parentParaOut.println();
					
				}
			}
			
			parentParaOut.flush();
			parentParaOut.close();
			
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected void printAPP4QueryByHybrid(String filePrefix){
		String topKChild4QueryFile = filePrefix + "topAPP4Query.txt";
		System.out.println("hybrid model rank");
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4QueryFile));
			
			for(_APPQuery appQuery: m_APPQueries){

				pw.print(appQuery.getQueryID()+"\t");
				for(_Doc d:m_trainSet){
					if(d instanceof _ParentDoc){
						double likelihood = rankAPP4QueryByHybrid(appQuery, (_ParentDoc)d);
						
						pw.print(d.getTitle()+":"+likelihood);
						pw.print("\t");
					}
				}
				
				pw.println();
			}
			
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected double rankAPP4QueryByHybrid(_APPQuery appQuery, _ParentDoc pDoc){
		double queryLikelihood = 0.0;
		
		double smoothingMu = m_LM.m_smoothingMu;
		_SparseFeature[] pFV = pDoc.getSparse();
		
		double docLenHybridVal = 0;
		docLenHybridVal += pDoc.getDocInferLength();
		
		double alphaDoc = smoothingMu/(smoothingMu+docLenHybridVal);
		
		for(_Word w:appQuery.getWords()){
			int wid = w.getIndex();
			double wordLoglikelihood = 0;
			double featureHybridVal = 0;
		
			int featureIndex = Utils.indexOf(pFV, wid);
			double pDocVal = 0;
			if(featureIndex != -1){
				pDocVal = pFV[featureIndex].getValue();
			}
			featureHybridVal += pDocVal;
		
			double wordLMLikelihood = (1-alphaDoc)*(featureHybridVal/docLenHybridVal);
			
			wordLMLikelihood += alphaDoc*m_LM.getReferenceProb(wid);
			
			double wordTMLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = wordByTopicProb(k, wid)*topicInDocProb(k,  pDoc)/(pDoc.getDocInferLength()+d_alpha*number_of_topics);
				wordTMLikelihood += wordPerTopicLikelihood;
			}
			
			wordLoglikelihood = (m_tau)*wordLMLikelihood+(1-m_tau)*wordTMLikelihood;
			
			queryLikelihood += Math.log(wordLoglikelihood);
		}
		
		return queryLikelihood;
	}
	
	protected void printAPP4QueryByTopicModel(String filePrefix){
		String topKChild4QueryFile = filePrefix + "topChild4Query.txt";
		System.out.println("topic model rank");

		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4QueryFile));
			
			for(_APPQuery appQuery: m_APPQueries){
				
				pw.print(appQuery.getQueryID()+"\t");
				for(_Doc d:m_trainSet){
					if(d instanceof _ParentDoc){
						double likelihood = rankAPP4QueryByTM(appQuery, d);
//						likelihoodMap.put(d.getTitle(), likelihood);
//						for(_ChildDoc cDoc:((_ParentDoc)d).m_childDocs){
//							likelihood += rankAPP4QueryByTM(appQuery, cDoc);
//						}
						pw.print(d.getTitle()+":"+likelihood);
						pw.print("\t");
					}
				}
				
				pw.println();
			}
			
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected double rankAPP4QueryByTM(_APPQuery query, _Doc d){
		double queryLikelihood = 0;
	
		for(_Word w:query.getWords()){
			int wid = w.getIndex();
				
			double wordLoglikelihood = 0;
				
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = wordByTopicProb(k, wid)*(topicInDocProb(k, d))/(d.getDocInferLength()+d_alpha*number_of_topics);
				wordLoglikelihood += wordPerTopicLikelihood;
			}
			queryLikelihood += Math.log(wordLoglikelihood);
			
		}
		
		return queryLikelihood;
	}
	
}
