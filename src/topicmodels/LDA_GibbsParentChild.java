////////********parent and child documents train LDA together************////////
package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._RankItem;
import structures._SparseFeature;
import structures._Stn;
import utils.Utils;

public class LDA_GibbsParentChild extends LDA_Gibbs {

	public LDA_GibbsParentChild(int number_of_iteration, double converge,
			double beta, _Corpus c, double lambda, int number_of_topics,
			double alpha, double burnIn, int lag) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag);

		// converge = 0

		// TODO Auto-generated constructor stub
	}

	protected void createSpace() {
		super.createSpace();
	}

	@Override
	public String toString() {
		return String
				.format("LDA for whole corpus[k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]",
						number_of_topics, d_alpha, d_beta);
	}

	protected void initialize_probability(Collection<_Doc> collection) {
		for (int i = 0; i < number_of_topics; i++) {
			// Arrays.fill(word_topic_sstat[i], 0);
			Arrays.fill(word_topic_sstat[i], 0);

		}
		Arrays.fill(m_sstat, 0);

		for (_Doc d : collection) {
			if (d instanceof _ParentDoc) {
				((_ParentDoc) d).setTopics4Gibbs(number_of_topics);
				for (int i = 0; i < d.m_words.length; i++) {
					word_topic_sstat[d.m_topicAssignment[i]][d.m_words[i]]++;
					m_sstat[d.m_topicAssignment[i]]++;
				}
			} else if (d instanceof _ChildDoc) {
				((_ChildDoc) d).setTopics4Gibbs(number_of_topics, 0);
				for (int i = 0; i < d.m_words.length; i++) {
					word_topic_sstat[d.m_topicAssignment[i]][d.m_words[i]]++;
					m_sstat[d.m_topicAssignment[i]]++;

				}

			}

		}

		imposePrior();
	}

	public double calculate_E_step(_Doc d) {
		if (d instanceof _ParentDoc) {
			((_ParentDoc) d).permutation();
			// d.permutation();
			sampleParentTopic((_ParentDoc) d);

		} else {
			if (d instanceof _ChildDoc) {
				((_ChildDoc) d).permutation(0);
				sampleChildTopic((_ChildDoc) d);
			}
		}

		return 1;
	}

	public void sampleParentTopic(_ParentDoc d) {
		double p;
		int wid, tid;
		for (int i = 0; i < d.m_words.length; i++) {
			wid = d.m_words[i];
			tid = d.m_topicAssignment[i];

			// remove the word's topic assignment

			d.m_sstat[tid]--;
			if (m_collectCorpusStats) {
				word_topic_sstat[tid][wid]--;
				m_sstat[tid]--;
			}

			// perform random sampling
			p = 0;
			for (tid = 0; tid < number_of_topics; tid++) {
				// p(z|d) * p(w|z)
				p += (d.m_sstat[tid] + d_alpha)
						* (word_topic_sstat[tid][wid] + d_beta)
						/ (m_sstat[tid] + d_beta * vocabulary_size);
			}
			p *= m_rand.nextDouble();

			tid = -1;
			while (p > 0 && tid < number_of_topics - 1) {
				tid++;
				p -= (d.m_sstat[tid] + d_alpha)
						* (word_topic_sstat[tid][wid] + d_beta)
						/ (m_sstat[tid] + d_beta * vocabulary_size);
			}

			// assign the selected topic to word
			d.m_topicAssignment[i] = tid;
			d.m_sstat[tid]++;
			if (m_collectCorpusStats) {
				word_topic_sstat[tid][wid]++;
				m_sstat[tid]++;
			}
		}
		// return samplingTopics;
	}

	public void sampleChildTopic(_ChildDoc d) {
		double p;
		int wid, tid;
		for (int i = 0; i < d.m_words.length; i++) {
			wid = d.m_words[i];
			tid = d.m_topicAssignment[i];

			// remove the word's topic assignment
			if (d.m_sstat == null)
				System.out.println("null pointer");
			d.m_sstat[tid]--;
			if (m_collectCorpusStats) {
				word_topic_sstat[tid][wid]--;
				m_sstat[tid]--;
			}

			// perform random sampling
			p = 0;
			for (tid = 0; tid < number_of_topics; tid++)
				p += (d.m_sstat[tid] + d_alpha)
						* (word_topic_sstat[tid][wid] + d_beta)
						/ (m_sstat[tid] + d_beta * vocabulary_size);
			// p(z|d) * p(w|z)

			p *= m_rand.nextDouble();

			tid = -1;
			while (p > 0 && tid < number_of_topics - 1) {
				tid++;
				p -= (d.m_sstat[tid] + d_alpha)
						* (word_topic_sstat[tid][wid] + d_beta)
						/ (m_sstat[tid] + d_beta * vocabulary_size);
			}

			// assign the selected topic to word
			d.m_topicAssignment[i] = tid;
			d.m_sstat[tid]++;
			if (m_collectCorpusStats) {
				word_topic_sstat[tid][wid]++;
				m_sstat[tid]++;
			}
		}
		// return samplingTopics;
	}


	public void calculate_M_step(int iter) {
		// if (iter % m_lag == 0) {
		// calLogLikelihood(iter);
		// }

		if (iter > m_burnIn && iter % m_lag == 0) {
			for (int i = 0; i < this.number_of_topics; i++) {
				for (int v = 0; v < this.vocabulary_size; v++) {
					topic_term_probabilty[i][v] += word_topic_sstat[i][v]
							+ d_beta;
				}
			}

			// used to estimate final theta for each document
			for (_Doc d : m_trainSet)
				if (d instanceof _ParentDoc)
					collectParentStats((_ParentDoc) d);
				else if (d instanceof _ChildDoc)
					collectChildStats((_ChildDoc) d);
		}

	}

	protected void collectParentStats(_ParentDoc d) {
		for (int k = 0; k < this.number_of_topics; k++) {
			d.m_topics[k] += (d.m_sstat[k] + d_alpha);
		}
	}

	protected void collectChildStats(_ChildDoc d) {
		for (int k = 0; k < this.number_of_topics; k++) {
			d.m_topics[k] += (d.m_sstat[k] + d_alpha);
		}
	}

	protected void finalEst() {
		for (int i = 0; i < this.number_of_topics; i++) {

			Utils.L1Normalization(topic_term_probabilty[i]);
		}

		for (_Doc d : m_trainSet) {
			estThetaInDoc(d);
		}
		discoverSpecificComments();
	}

	protected void estThetaInDoc(_Doc d) {
		if (d instanceof _ParentDoc) {
			Utils.L1Normalization(d.m_topics);
			estStnThetaInParentDoc((_ParentDoc) d);
		} else if (d instanceof _ChildDoc) {
			
			Utils.L1Normalization(d.m_topics);
			
		}

	}

	public void estStnThetaInParentDoc(_ParentDoc d) {
		_SparseFeature[] fv = d.getSparse();
		double[][] phi = new double[fv.length][number_of_topics];
		HashMap<Integer, Integer> indexMap = new HashMap<Integer, Integer>();

		// //computeWordTopicProportionInDoc
		////compute phi
		for (int i = 0; i < fv.length; i++) {
			int index = fv[i].getIndex();
			indexMap.put(index, i);
		}

		for (int n = 0; n < d.m_words.length; n++) {
			int index = d.m_words[n];
			int topic = d.m_topicAssignment[n];
			phi[indexMap.get(index)][topic]++;
		}

		for (int i = 0; i < fv.length; i++) {
			Utils.L1Normalization(phi[i]);
		}

		////compute topic proportion
		//// sentenceMap:HashMap<sentenceID, _stn> in _ParentDoc2
		for (int i = 0; i < d.m_sentenceMap.size(); i++) {
			
			_Stn stnObject = d.m_sentenceMap.get(i);
			if(stnObject==null){
				continue;
			}
			// initial topic proportions (m_topics) of sentences
			_SparseFeature[] sv = stnObject.getFv();
			
			//m_stnLength: the length of sentence
			//m_words: the index in CV of each word in the sentence 
			stnObject.setTopicsVct(number_of_topics);
			for (int j = 0; j < sv.length; j++) {
				int index = sv[j].getIndex();
				double value = sv[j].getValue();
				for (int k = 0; k < number_of_topics; k++) {
					stnObject.m_topics[k] += value*phi[indexMap.get(index)][k];
				}
			}
			Utils.L1Normalization(stnObject.m_topics);
		

		}

	}

	public void discoverSpecificComments() {
		System.out.println("topic similarity");
		String fileName = "topicSimilarity.txt";

		try {
			PrintWriter pw = new PrintWriter(new File(fileName));

			for (_Doc doc : m_trainSet) {
				if (doc instanceof _ParentDoc) {
					pw.print(doc.getName() + "\t");
					double stnTopicSimilarity = 0.0;
					double docTopicSimilarity = 0.0;
					for (_ChildDoc cDoc : ((_ParentDoc) doc).m_childDocs) {
						pw.print(cDoc.getName() + ":");

						docTopicSimilarity = computeSimilarity(
								((_ParentDoc) doc).m_topics, cDoc.m_topics);
						pw.print(docTopicSimilarity);
						for (int i = 0; i < ((_ParentDoc) doc).m_sentenceMap
								.size(); i++) {
							_Stn stnObj = ((_ParentDoc) doc).m_sentenceMap
									.get(i);

							if (stnObj == null) {
								// some sentences are normalized into zero
								//length sentences, the similarity is set to be 0
								pw.print(":0");
								continue;
							}
							double[] stnTopics = stnObj.m_topics;

							stnTopicSimilarity = computeSimilarity(stnTopics,
									cDoc.m_topics);
							pw.print(":" + stnTopicSimilarity);
						}
						pw.print("\t");
					}
					pw.println();
				} else {
					continue;
				}
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public double computeSimilarity(double[] topic1, double[] topic2) {
		double similarity = 0.0;
		double numerator = 0.0;
		double denominator1 = 0.0;
		double denominator2 = 0.0;
		for (int k = 0; k < number_of_topics; k++) {

			numerator += topic1[k] * topic2[k];
			denominator1 += topic1[k] * topic1[k];
			denominator2 += topic2[k] * topic2[k];
		}

		if ((denominator1 == 0) || (denominator2 == 0)) {
			similarity = 0;
			return similarity;
		}

		similarity = Math.log(numerator) - Math.log(Math.sqrt(denominator1))
				- Math.log(Math.sqrt(denominator2));

		similarity = Math.exp(similarity);
		return similarity;
	}
	
	public void printTopWords(int k, String betaFile) {
		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < number_of_topics; i++)
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];	
		}

		Utils.L1Normalization(m_sstat);

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

		String filePrefix = betaFile.replace("topWords.txt", "");
		debugOutput(filePrefix);
		
	}
	
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

		for (_Doc d : m_trainSet) {
		if (d instanceof _ParentDoc) {
				printParentTopicAssignment((_ParentDoc) d, parentTopicFolder);
			} else if (d instanceof _ChildDoc) {
				printChildTopicAssignment((_ChildDoc) d, childTopicFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParameter(parentParameterFile, childParameterFile);

	}

	public void printParentTopicAssignment(_ParentDoc d, File parentFolder) {
		String topicAssignmentFile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(parentFolder,
					topicAssignmentFile));
			
			for(int n=0; n<d.m_words.length; n++){
				int index = d.m_words[n];
				int topic = d.m_topicAssignment[n];
				String featureName = m_corpus.getFeature(index);
				pw.print(featureName + ":" + topic + "\t");
			}
			
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void printChildTopicAssignment(_ChildDoc d, File childFolder) {
		String topicAssignmentfile = d.getName() + "_.txt";
		try {
			PrintWriter pw = new PrintWriter(new File(childFolder,
					topicAssignmentfile));

			for (int n = 0; n < d.m_words.length; n++) {
				int index = d.m_words[n];
				int topic = d.m_topicAssignment[n];
				String featureName = m_corpus.getFeature(index);
					
				pw.print(featureName + ":" + topic + "\t");
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}


	public void printParameter(String parentParameterFile, String childParameterFile){
		try{
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			for(_Doc d: m_trainSet){
				if(d instanceof _ParentDoc){
					parentParaOut.print(d.getName()+"\t");
					parentParaOut.print("topicProportion\t");
					for(int k=0; k<number_of_topics; k++){
						parentParaOut.print(d.m_topics[k]+"\t");
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
		}
		catch (Exception ex) {
			System.err.print("File Not Found");
		}

	}

}
