//package topicmodels;
//
//import java.io.File;
//import java.io.FileNotFoundException;
//import java.io.PrintWriter;
//import java.util.Arrays;
//import java.util.Collection;
//import java.util.Random;
//
//import com.sun.org.apache.xml.internal.resolver.helpers.PublicId;
//
//import structures.MyPriorityQueue;
//import structures._ChildDoc;
//import structures._Corpus;
//import structures._Doc;
//import structures._ParentDoc;
//import structures._RankItem;
//import structures._SparseFeature;
//import structures._Stn;
//import utils.Utils;
//
//public class LDA_Gibbs_Debug extends LDA_Gibbs{
//	Random m_rand;
//	int m_burnIn; // discard the samples within burn in period
//	int m_lag; // lag in accumulating the samples
//	
//	//all computation here is not in log-space!!!
//	public LDA_Gibbs_Debug(int number_of_iteration, double converge, double beta,
//			_Corpus c, double lambda, 
//			int number_of_topics, double alpha, double burnIn, int lag) {
//		super( number_of_iteration,  converge,  beta,
//			 c,  lambda, number_of_topics,  alpha,  burnIn,  lag);
//		
//		m_rand = new Random();
//		m_burnIn = (int) (burnIn * number_of_iteration);
//		m_lag = lag;
//	}
//	
//	protected void initialize_probability(Collection<_Doc> collection) {
//		for(int i=0; i< number_of_topics; i++)
//			Arrays.fill(word_topic_sstat[i], d_beta);
//		Arrays.fill(m_sstat, d_beta*vocabulary_size);
//		
//		for(_Doc d: collection){
//			if(d instanceof _ParentDoc) {
//				for(_Stn stnObj: d.getSentences()){
//					stnObj.setTopicsVct(number_of_topics);
//				}	
//			}else{
////				System.out.println("debug");
//			}
//			
//			
//			
//			d.setTopics4Gibbs(number_of_topics, d_alpha);
//			for(int i=0; i<d.m_words.length; i++){
//				word_topic_sstat[d.m_topicAssignment[i]][d.m_words[i]] ++;
//				m_sstat[d.m_topicAssignment[i]] ++;
//			}
//		}
//		
//		imposePrior();
//	};
//	
//	protected void finalEst(){
//		super.finalEst();
//	}
//	
//	protected void estThetaInDoc(_Doc d) {
//		super.estThetaInDoc(d);
//		if(d instanceof _ParentDoc){
//			((_ParentDoc)d).estStnTheta();
//		}
//	}
//	
//	protected void collectStats(_Doc d) {
//		for(int k=0; k<this.number_of_topics; k++)
//			d.m_topics[k] += d.m_sstat[k] + d_alpha;
//		if(d instanceof _ParentDoc){
//			((_ParentDoc) d).collectTopicWordStat();
//		}
//	}
//	
//	@Override
//	public double inference(_Doc d) {
//		initTestDoc(d);//this is not a corpus level estimation
//		
//		double likelihood = Double.NEGATIVE_INFINITY, count = 0;
//		int  i = 0;
//		do {
//			calculate_E_step(d);
//			if (i>m_burnIn && i%m_lag==0){
//				collectStats(d);
//				count ++;
//			}
//		} while (++i<this.number_of_iteration);
//		
//		estThetaInDoc(d);
//		likelihood = Utils.logSum(likelihood, calculate_log_likelihood(d));				
//
//		return likelihood; // this is average joint probability!
//	}
//	
//	@Override
//	public double calculate_log_likelihood(_Doc d){
//		double docLogLikelihood = 0.0;
//		_SparseFeature[] fv = d.getSparse();
//		
//		for(int j=0; j<fv.length; j++){
//			int index = fv[j].getIndex();
//			double value = fv[j].getValue();
//			
//			double wordLogLikelihood = 0;
//			for(int k=0; k<number_of_topics; k++){
////				if(topic_term_probabilty[k][index] == 0){
////					topic_term_probabilty[k][index] = 1e-9;
////				}
//				double wordPerTopicLikelihood = Math.log(topic_term_probabilty[k][index]);
////				System.out.println("first part\t"+wordPerTopicLikelihood);
////				if(d.m_topics[k] == 0){
////					d.m_topics[k] = 1e-9;
////				}
//				wordPerTopicLikelihood += Math.log(d.m_topics[k]);
////				System.out.println("second part\t"+wordPerTopicLikelihood);
//				if(wordLogLikelihood == 0){
//					wordLogLikelihood = wordPerTopicLikelihood;
//				}else{
//					wordLogLikelihood = Utils.logSum(wordLogLikelihood, wordPerTopicLikelihood);
//				}
//			}
//			docLogLikelihood += value*wordLogLikelihood;
//		}
//
//		return docLogLikelihood;
//	}
//	
//	
//	@Override
// 	public void printTopWords(int k, String betaFile) {
//		Arrays.fill(m_sstat, 0);
//
//		System.out.println("print top words");
//		for (_Doc d : m_trainSet) {
//			for (int i = 0; i < number_of_topics; i++)
//				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
//						: d.m_topics[i];	
//		}
//
//		Utils.L1Normalization(m_sstat);
//
//		try {
//			System.out.println("beta file");
//			PrintWriter betaOut = new PrintWriter(new File(betaFile));
//			for (int i = 0; i < topic_term_probabilty.length; i++) {
//				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
//						k);
//				for (int j = 0; j < vocabulary_size; j++)
//					fVector.add(new _RankItem(m_corpus.getFeature(j),
//							topic_term_probabilty[i][j]));
//
//				betaOut.format("Topic %d(%.3f):\t", i, m_sstat[i]);
//				for (_RankItem it : fVector) {
//					betaOut.format("%s(%.3f)\t", it.m_name,
//							m_logSpace ? Math.exp(it.m_value) : it.m_value);
//					System.out.format("%s(%.3f)\t", it.m_name,
//						m_logSpace ? Math.exp(it.m_value) : it.m_value);
//				}
//				betaOut.println();
//				System.out.println();
//			}
//	
//			betaOut.flush();
//			betaOut.close();
//		} catch (Exception ex) {
//			System.err.print("File Not Found");
//		}
//
//		double loglikelihood = calLogLikelihoodByIntegrateTopics(0);
//		System.out.format("Final Log Likelihood %.3f\t", loglikelihood);
//		infoWriter.format("Final Log Likelihood %.3f\t", loglikelihood);
//		
//		String filePrefix = betaFile.replace("topWords.txt", "");
//		debugOutput(filePrefix);
//		
//	}
//	
//	public void debugOutput(String filePrefix){
//
//		File topicFolder = new File(filePrefix + "topicAssignment");
//	
//		if (!topicFolder.exists()) {
//			System.out.println("creating directory" + topicFolder);
//			topicFolder.mkdir();
//		}
//
//		for (_Doc d : m_trainSet) {
//			printTopicAssignment(d, topicFolder);
//
//		}
//
//		String parentParameterFile = filePrefix + "parentParameter.txt";
//		String childParameterFile = filePrefix + "childParameter.txt";
//	
//		printParameter(parentParameterFile, childParameterFile);
//
//		printEntropy(filePrefix);
//	}
//
//	public void printTopicAssignment(_Doc d, File topicFolder) {
//	//	System.out.println("printing topic assignment parent documents");
//		
//		String topicAssignmentFile = d.getName() + ".txt";
//		try {
//			PrintWriter pw = new PrintWriter(new File(topicFolder,
//					topicAssignmentFile));
//			
//			for(int n=0; n<d.m_words.length; n++){
//				int index = d.m_words[n];
//				int topic = d.m_topicAssignment[n];
//				String featureName = m_corpus.getFeature(index);
//				pw.print(featureName + ":" + topic + "\t");
//			}
//			
//			pw.flush();
//			pw.close();
//		} catch (FileNotFoundException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//
//	}
//
//	public void printParameter(String parentParameterFile, String childParameterFile){
//		System.out.println("printing parameter");
//		try{
//			System.out.println(parentParameterFile);
//			System.out.println(childParameterFile);
//			
//			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
//			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
//			for(_Doc d: m_trainSet){
//				if(d instanceof _ParentDoc){
//					parentParaOut.print(d.getName()+"\t");
//					parentParaOut.print("topicProportion\t");
//					for(int k=0; k<number_of_topics; k++){
//						parentParaOut.print(d.m_topics[k]+"\t");
//					}
//					
//					for(_Stn stnObj:d.getSentences()){							
//						parentParaOut.print("sentence"+(stnObj.getIndex()+1)+"\t");
//						for(int k=0; k<number_of_topics;k++){
//							parentParaOut.print(stnObj.m_topics[k]+"\t");
//						}
//					}
//					
//					parentParaOut.println();
//					
//				}else{
////					if(d instanceof _ChildDoc){
//						childParaOut.print(d.getName()+"\t");
//
//						childParaOut.print("topicProportion\t");
//						for (int k = 0; k < number_of_topics; k++) {
//							childParaOut.print(d.m_topics[k] + "\t");
//						}
//						
//						
//						childParaOut.println();
////					}
//				}
//			}
//			
//			parentParaOut.flush();
//			parentParaOut.close();
//			
//			childParaOut.flush();
//			childParaOut.close();
//		}
//		catch (Exception e) {
//			e.printStackTrace();
////			e.printStackTrace();
////			System.err.print("para File Not Found");
//		}
//
//	}
//	
//	protected void printEntropy(String filePrefix){
//		String entropyFile = filePrefix+"entropy.txt";
//		boolean logScale = true;
//		
//		try{
//			PrintWriter entropyPW = new PrintWriter(new File(entropyFile));
//			
//			for(_Doc d: m_trainSet){
//				double entropyValue = 0.0;
//				entropyValue = Utils.entropy(d.m_topics, logScale);
//				entropyPW.print(d.getName()+"\t"+entropyValue);
//				entropyPW.println();
//			}
//			entropyPW.flush();
//			entropyPW.close();
//		}catch(Exception e){
//			e.printStackTrace();
//		}
//		
//		
//		
//	} 
//
//	//p(w)= \sum_z p(w|z)p(z)
//	protected double calLogLikelihoodByIntegrateTopics(int iter){
//		double logLikelihood = 0.0;
//		
//		for(_Doc d: m_trainSet){
//			logLikelihood += docLogLikelihoodByIntegrateTopics(d);
//		}
//		
//		return logLikelihood;
//	}
//	
//	protected double docLogLikelihoodByIntegrateTopics(_Doc d){
//		
//		double docLogLikelihood = 0.0;
//		_SparseFeature[] fv = d.getSparse();
//		
//		for(int j=0; j<fv.length; j++){
//			int index = fv[j].getIndex();
//			double value = fv[j].getValue();
//			
//			double wordLogLikelihood = 0;
//			for(int k=0; k<number_of_topics; k++){
//				double wordPerTopicLikelihood = Math.log(topic_term_probabilty[k][index]);
//				wordPerTopicLikelihood += Math.log(d.m_topics[k]);
//				if(wordLogLikelihood == 0){
//					wordLogLikelihood = wordPerTopicLikelihood;
//				}else{
//					wordLogLikelihood = Utils.logSum(wordLogLikelihood, wordPerTopicLikelihood);
//				}
//			}
//			docLogLikelihood += value*wordLogLikelihood;
//		}
//		
//		return docLogLikelihood;
//	}
//	
//}
