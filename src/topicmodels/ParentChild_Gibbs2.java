package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures._ChildDoc2;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc2;
import structures._RankItem;
import structures._SparseFeature;
import structures._Stn;
import utils.Utils;

public class ParentChild_Gibbs2 extends ParentChild_Gibbs {
	public double m_mu;

	public ParentChild_Gibbs2(int number_of_iteration, double converge,
			double beta, _Corpus c, double lambda, int number_of_topics,
			double alpha, double burnIn, int lag, double[] gamma, double mu) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag, gamma);
		m_mu = mu;
		// converge = 0
	}

	@Override
	protected void createSpace() {
		super.createSpace();
	}

	protected void initialize_probability(Collection<_Doc> collection) {
		// for(int i=0; i<number_of_topics; i++)
		// Arrays.fill(word_topic_sstat[i], d_beta);
		// Arrays.fill(m_sstat, d_beta*vocabulary_size);

		for (int i = 0; i < number_of_topics; i++) {
			Arrays.fill(m_parentWordTopicSstat[i], 0);
			Arrays.fill(m_childWordTopicSstat[i], 0);
		}
		Arrays.fill(m_parentSstat, 0);
		Arrays.fill(m_childSstat, 0);

		for (_Doc d : collection) {
			if (d instanceof _ParentDoc2) {
				((_ParentDoc2) d).setTopics4Gibbs(number_of_topics);
				for (int i = 0; i < d.m_words.length; i++) {
					m_parentWordTopicSstat[d.m_topicAssignment[i]][d.m_words[i]]++;
					m_parentSstat[d.m_topicAssignment[i]]++;
				}
			} else if (d instanceof _ChildDoc2) {
				((_ChildDoc2) d).setTopics4Gibbs(number_of_topics, m_gamma);
				for (int i = 0; i < d.m_words.length; i++) {
					m_childWordTopicSstat[d.m_topicAssignment[i]][d.m_words[i]]++;
					m_childSstat[d.m_topicAssignment[i]]++;

				}
			}
		}

		imposePrior();
	}

	@Override
	public String toString() {
		return String.format("Parent Child topic model 2 [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, mu:%.2f Gibbs Sampling]",
						number_of_topics, d_alpha, d_beta, m_gamma[1],
						m_gamma[2], m_mu);
	}

	@Override
	public double calculate_E_step(_Doc d) {		
		if (d instanceof _ParentDoc2) {
			((_ParentDoc2) d).permutation();
			sampleParentDocTopic((_ParentDoc2) d);	
		} else {
			if (d instanceof _ChildDoc2) {
				((_ChildDoc2) d).permutation();
				sampleChildDocTopic((_ChildDoc2) d);
			}
		}

		return 1;
	}

	void sampleParentDocTopic(_ParentDoc2 d) {
		int samplingTopic = 0;
		int wid, tid;
		double[] probRatio = new double[number_of_topics];
		double prob;
		double normalizedProb;
		double docLength = d.m_words.length;
		// double docLength = 1;

		for (int i = 0; i < d.m_words.length; i++) {
			normalizedProb = 0;
			prob = 0;
			wid = d.m_words[i];
			tid = d.m_topicAssignment[i];

			d.m_sstat[tid]--;
			if (m_collectCorpusStats) {
				m_parentWordTopicSstat[tid][wid]--;
				m_parentSstat[tid]--;
			}

			probRatio[0] = 1;
			normalizedProb += probRatio[0];
			for (tid = 0; tid < number_of_topics; tid++) {
				probRatio[tid] = 0;

				double term1 = (d_beta + m_parentWordTopicSstat[0][wid] + m_childWordTopicSstat[0][wid])
						/ (d_beta + m_parentWordTopicSstat[tid][wid] + m_childWordTopicSstat[tid][wid]);
				double term2 = (vocabulary_size * d_beta + m_parentSstat[tid] + m_childSstat[tid])
						/ (vocabulary_size * d_beta + m_parentSstat[0] + m_childSstat[0]);
				double term3 = (d_alpha + d.m_sstat[0])
						/ (d_alpha + d.m_sstat[tid]);
				double term4 = 1;
				
				double term5 = 0;
				for (_ChildDoc2 cDoc : d.m_childDocs) {
					
					double term41 = (Utils.lgamma(d_alpha + m_mu
							* (d.m_sstat[0] + 1) / (docLength)
							+ cDoc.m_xTopicSstat[0][0]));
					double term42 = (Utils.lgamma(d_alpha + m_mu
							* (d.m_sstat[0]) / (docLength)
							+ cDoc.m_xTopicSstat[0][0]));
					double term43 = (Utils.lgamma(d_alpha + m_mu
							* (d.m_sstat[tid]) / (docLength)
							+ cDoc.m_xTopicSstat[0][tid]));
					double term44 = (Utils.lgamma(d_alpha + m_mu
							* (d.m_sstat[tid] + 1) / (docLength)
							+ cDoc.m_xTopicSstat[0][tid]));
					//
					double term45 = (Utils.lgamma(d_alpha + m_mu
							* (d.m_sstat[0]) / (docLength)));
					double term46 = (Utils.lgamma(d_alpha + m_mu
							* (d.m_sstat[0] + 1) / (docLength)));
					double term47 = (Utils.lgamma(d_alpha + m_mu
							* (d.m_sstat[tid] + 1) / (docLength)));
					double term48 = (Utils.lgamma(d_alpha + m_mu
							* (d.m_sstat[tid]) / (docLength)));

					term5 = term41 - term42 + term43 - term44 + term45 - term46
							+ term47 - term48;

					term4 *= Math.exp(term5);

					if (invalidValue(term41) || invalidValue(term42)
							|| invalidValue(term43) || invalidValue(term44)
							|| invalidValue(term45) || invalidValue(term46)
							|| invalidValue(term47) || invalidValue(term48)
							|| invalidValue(term5) || invalidValue(term4)) {
						System.out.println("invalid term");
					}
				}

				probRatio[tid] = 1.0 / (term1 * term2 * term3 * term4);

				normalizedProb += probRatio[tid];
				
			}

			for (int k = 0; k < number_of_topics; k++) {
				probRatio[k] = probRatio[k] / normalizedProb;
			}

			prob = m_rand.nextDouble();

			for (tid = 0; tid < number_of_topics; tid++) {
				prob -= probRatio[tid];
				if (prob <= 0)
					break;
			}
			if (tid == number_of_topics)
				tid--;
			samplingTopic = tid;

			d.m_topicAssignment[i] = samplingTopic;
			d.m_sstat[samplingTopic]++;
			if (m_collectCorpusStats) {
				m_parentWordTopicSstat[samplingTopic][wid]++;
				m_parentSstat[samplingTopic]++;
			}
		}
	}

	public boolean invalidValue(double term) {
		if (Math.abs(term) == Double.MAX_VALUE - 1) {
			System.out.println(term);
			return true;
		} else {
			return false;
		}
	}

	void sampleChildDocTopic(_ChildDoc2 d) {
		int wid, tid, xid;

		double[][] xTopicProb = new double[2][number_of_topics];
		double prob;
		double normalizedProb = 0;

		for (int i = 0; i < d.m_words.length; i++) {
			int samplingX = 0;
			int samplingTopic = 0;
			prob = 0;
			normalizedProb = 0;

			wid = d.m_words[i];
			tid = d.m_topicAssignment[i];
			xid = d.m_xIndicator[i];

			d.m_xTopicSstat[xid][tid]--;
			d.m_xSstat[xid]--;
			if (m_collectCorpusStats) {
				m_childWordTopicSstat[tid][wid]--;
				m_childSstat[tid]--;
			}

			// p(z=tid,x=1) from specific
			for (tid = 0; tid < number_of_topics; tid++) {
				double term1 = (d_beta + m_parentWordTopicSstat[tid][wid] + m_childWordTopicSstat[tid][wid])
						/ (d_beta * vocabulary_size + m_parentSstat[tid] + m_childSstat[tid]);
				double term2 = (d.m_xTopicSstat[1][tid] + d_alpha)
						/ (number_of_topics * d_alpha + d.m_xSstat[1]);
				// double term3 =
				// (m_gamma[1]+d.m_xSstat[0])/(m_gamma[1]+m_gamma[2]+d.m_xSstat[0]+d.m_xSstat[1]);
				double term3 = (m_gamma[1] + d.m_xSstat[1]);
				xTopicProb[1][tid] = term1 * term2 * term3;
				normalizedProb += xTopicProb[1][tid];
			}

			if (d.m_parentDoc2 == null) {
				System.out.println("null parent in child doc" + d.getName());
			}

			double parentDocLen = d.m_parentDoc2.getTotalDocLength();
		
			// p(z=tid, x=0) from background
			for (tid = 0; tid < number_of_topics; tid++) {
				double term1 = (d_beta + m_parentWordTopicSstat[tid][wid] + m_childWordTopicSstat[tid][wid])
						/ (d_beta * vocabulary_size + m_parentSstat[tid] + m_childSstat[tid]);
				double term2 = (d_alpha + m_mu * d.m_parentDoc2.m_sstat[tid]
						/ parentDocLen + d.m_xTopicSstat[0][tid])
						/ (number_of_topics * d_alpha + m_mu + d.m_xSstat[0]);
				double term3 = (m_gamma[0] + d.m_xSstat[0]);
				xTopicProb[0][tid] = term1 * term2 * term3;
				normalizedProb += xTopicProb[0][tid];
			}

			boolean finishLoop = false;
			prob = normalizedProb * m_rand.nextDouble();
			for (xid = 0; xid < m_gamma.length; xid++) {
				for (tid = 0; tid < number_of_topics; tid++) {
					prob -= xTopicProb[xid][tid];
					if (prob <= 0) {
						finishLoop = true;
						break;
					}
				}
				if (finishLoop) {
					break;
				}
			}

			if (xid == 2)
				xid--;
			if (tid == number_of_topics)
				tid--;

			samplingX = xid;
			samplingTopic = tid;

			d.m_topicAssignment[i] = samplingTopic;
			d.m_xIndicator[i] = samplingX;

			d.m_xTopicSstat[samplingX][samplingTopic]++;
			d.m_xSstat[samplingX]++;
			if (m_collectCorpusStats) {
				m_childWordTopicSstat[samplingTopic][wid]++;
				m_childSstat[samplingTopic]++;
			}

		}
	}

	public void calculate_M_step(int iter) {
		if (iter % m_lag == 0)
			calLogLikelihood2(iter);

		if (iter > m_burnIn && iter % m_lag == 0) {
			for (int i = 0; i < this.number_of_topics; i++) {
				for (int v = 0; v < this.vocabulary_size; v++) {
					//only one \phi, m_parentWordTopicSstat+m_childWordTopicSstatk, to reuse the code
					m_parentTopicTermProb[i][v] += (m_parentWordTopicSstat[i][v]
							+ m_childWordTopicSstat[i][v] + d_beta);
					m_childTopicTermProb[i][v] += (m_parentWordTopicSstat[i][v]
							+ m_childWordTopicSstat[i][v] + d_beta);
				}
			}

			for (_Doc d : m_trainSet) {
				if (d instanceof _ParentDoc2)
					collectParentStats((_ParentDoc2) d);
				else if (d instanceof _ChildDoc2) {
					collectChildStats((_ChildDoc2) d);

				}

			}
		}
	}

	protected void collectParentStats(_ParentDoc2 d) {
		for (int k = 0; k < this.number_of_topics; k++) {
			d.m_topics[k] += (d.m_sstat[k] + d_alpha);
		}
	}

	protected void collectChildStats(_ChildDoc2 d) {
		for (int j = 0; j < m_gamma.length; j++) {
			d.m_xProportion[j] += d.m_xSstat[j] + m_gamma[j];
		}

		double parentDocLength = d.m_parentDoc2.getTotalDocLength();

		for (int k = 0; k < this.number_of_topics; k++) {
			d.m_xTopics[1][k] += (d.m_xTopicSstat[1][k] + d_alpha);
			d.m_xTopics[0][k] += (d.m_xTopicSstat[0][k] + d_alpha + m_mu
					* d.m_parentDoc2.m_sstat[k] / parentDocLength);
			d.m_topics[k] += d.m_xTopics[1][k] + d.m_xTopics[0][k];
		}
	}

	protected void finalEst() {
		for (int i = 0; i < this.number_of_topics; i++) {
			Utils.L1Normalization(m_parentTopicTermProb[i]);
			Utils.L1Normalization(m_childTopicTermProb[i]);
		}

		for (_Doc d : m_trainSet) {
			estThetaInDoc(d);
		}

		// used to compare similarity between sentences of parent documents and child documents
		discoverSpecificComments();
	}

	protected void estThetaInDoc(_Doc d) {
		if (d instanceof _ParentDoc2) {
			Utils.L1Normalization(d.m_topics);
			// estimate topic proportion of sentences in parent documents
			estStnThetaInParentDoc((_ParentDoc2) d);
		} else if (d instanceof _ChildDoc2) {
			Utils.L1Normalization(((_ChildDoc2) d).m_xProportion);

			Utils.L1Normalization(d.m_topics);
			for (int x = 0; x < m_gamma.length; x++) {
				Utils.L1Normalization(((_ChildDoc2) d).m_xTopics[x]);
			}
		}

	}

	public void estStnThetaInParentDoc(_ParentDoc2 d) {
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
			// initial topic proportions (m_topics) of sentences
			stnObject.setTopicsVct(number_of_topics);
			Arrays.fill(stnObject.m_topics, 0);
			
			//m_stnLength: the length of sentence
			if (stnObject.m_stnLength != 0) {
				
				//m_words: the index in CV of each word in the sentence 
				int[] indexInCV = stnObject.m_words;
				for (int j = 0; j < indexInCV.length; j++) {
					int index = indexInCV[j];
					for (int k = 0; k < number_of_topics; k++) {
						stnObject.m_topics[k] += phi[indexMap.get(index)][k];
					}
				}
				Utils.L1Normalization(stnObject.m_topics);
			}
		}

	}

	public void discoverSpecificComments() {
		System.out.println("topic similarity");
		String fileName = "./data/results/0108_9/topicSimilarity.txt";

		try {
			PrintWriter pw = new PrintWriter(new File(fileName));

			for (_Doc doc : m_trainSet) {
				if (doc instanceof _ParentDoc2) {
					pw.print(doc.getName() + "\t");
					double stnTopicSimilarity = 0.0;
					double docTopicSimilarity = 0.0;
					for (_ChildDoc2 cDoc : ((_ParentDoc2) doc).m_childDocs) {
						pw.print(cDoc.getName() + ":");

						docTopicSimilarity = computeSimilarity(
								((_ParentDoc2) doc).m_topics, cDoc.m_topics);
						pw.print(docTopicSimilarity);
						for (int i = 0; i < ((_ParentDoc2) doc).m_sentenceMap
								.size(); i++) {
							_Stn stnObj = ((_ParentDoc2) doc).m_sentenceMap
									.get(i);

							if (stnObj.m_stnLength == 0) {
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
		return Math.exp(Utils.cosine(topic1, topic2));
	}
	
	public void printParentTopicAssignment(_ParentDoc2 d) {
		String topicAssignmentFile = "./data/results/0108_9/parentTopicAssignment/topicAssignment_"
				+ d.getName() + "_.txt";
		try {
			PrintWriter pw = new PrintWriter(new File(topicAssignmentFile));
			for (int i = 0; i < d.m_sentenceMap.size(); i++) {

				_Stn stnObject = d.m_sentenceMap.get(i);

				if (stnObject.m_stnLength != 0) {
					//m_words: the index in CV of each word in the sentence 
					int[] indexInCV = stnObject.m_words;
					// m_wordPositionInDoc: the position in parent document of each word in the sentence.
					int[] positionInDoc = stnObject.m_wordPositionInDoc;
					for (int j = 0; j < indexInCV.length; j++) {
						int index = indexInCV[j];
						int topic = d.m_topicAssignment[positionInDoc[j]];
						String featureName = m_corpus.getFeature(index);
						pw.print(featureName + ":" + topic + "\t");

					}

				}
				pw.println();
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void printChildTopicAssignment(_ChildDoc2 d) {
		String topicAssignmentfile = "./data/results/0108_9/childTopicAssignment/topicAssignment_"
				+ d.getName() + "_.txt";
		try {
			PrintWriter pw = new PrintWriter(new File(topicAssignmentfile));

			// m_index: the index in CV of each word in child document.
			// m_index not permutated, remain constant
			for (int i = 0; i < d.m_index.length; i++) {
				int index = d.m_index[i];

				String featureName = m_corpus.getFeature(index);

				//m_positionInDoc: the position in child document of each word
				int positionInDoc = d.m_positionInDoc[i];
				int topic = d.m_topicAssignment[positionInDoc];

				//used to verify whether there are bugs.
				//if they are the same, the part of code is correct.
				if (index == d.m_words[positionInDoc])
					pw.print(featureName + ":" + topic + "\t");
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void printTopWords(int k, String betaFile) {
		Arrays.fill(m_parentSstat, 0);
		Arrays.fill(m_childSstat, 0);

		System.out.println("print top words");
		for (_Doc d : m_trainSet) {
			if (d instanceof _ParentDoc2) {
				//print out topic assignments of parent documents
				printParentTopicAssignment((_ParentDoc2)d);

				for (int i = 0; i < number_of_topics; i++)
					m_parentSstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
							: d.m_topics[i];
			} else if (d instanceof _ChildDoc2) {
				//print out topic assignments of child documents
				printChildTopicAssignment((_ChildDoc2)d);
				
				for (int i = 0; i < number_of_topics; i++)
					m_childSstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
							: d.m_topics[i];
			}

		}

		Utils.L1Normalization(m_parentSstat);
		Utils.L1Normalization(m_childSstat);

		String parentBetaFile = betaFile.replace(".txt", "parent.txt");
		String childBetaFile = betaFile.replace(".txt", "child.txt");

		printParentTopWords(k, parentBetaFile);
		printChildTopWords(k, childBetaFile);

		String parentParameterFile = parentBetaFile
				.replace("beta", "parameter");
		String childParameterFile = childBetaFile.replace("beta", "parameter");
		printParameter(parentParameterFile, childParameterFile);

	}

	public void printParentTopWords(int k, String parentBetaFile) {
		try {
			System.out.println("parent beta file");
			PrintWriter parentBetaOut = new PrintWriter(
					new File(parentBetaFile));
			for (int i = 0; i < m_parentTopicTermProb.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							m_parentTopicTermProb[i][j]));

				parentBetaOut.format("Topic %d(%.3f):\t", i, m_parentSstat[i]);
				for (_RankItem it : fVector) {
					parentBetaOut.format("%s(%.3f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
					System.out.format("%s(%.3f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				}
				parentBetaOut.println();
				System.out.println();
			}

			parentBetaOut.flush();
			parentBetaOut.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}

	public void printChildTopWords(int k, String childBetaFile) {
		try {
			System.out.println("child beta file");
			PrintWriter childBetaOut = new PrintWriter(new File(childBetaFile));

			for (int i = 0; i < m_childTopicTermProb.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							m_childTopicTermProb[i][j]));

				childBetaOut.format("Topic %d(%.3f):\t", i, m_childSstat[i]);
				System.out.format("Topic %d(%.3f):\t", i, m_childSstat[i]);
				for (_RankItem it : fVector) {
					childBetaOut.format("%s(%.3f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
					System.out.format("%s(%.3f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				}
				childBetaOut.println();
				System.out.println();
			}

			childBetaOut.flush();
			childBetaOut.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}

	public void printParameter(String parentParameterFile,
			String childParameterFile) {
		try {
			PrintWriter parentParaOut = new PrintWriter(new File(
					parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(
					childParameterFile));
			for (_Doc d : m_trainSet) {
				if (d instanceof _ParentDoc2) {
					parentParaOut.print(d.getName() + "\t");
					parentParaOut.print("topicProportion\t");
					for (int k = 0; k < number_of_topics; k++) {
						parentParaOut.print(d.m_topics[k] + "\t");
					}

					// print topic proportion of the sentence in parent docs
					for (int i = 0; i < ((_ParentDoc2) d).m_sentenceMap.size(); i++) {
						_Stn stnObject = ((_ParentDoc2) d).m_sentenceMap.get(i);
						parentParaOut.print("sentence" + (i + 1) + "\t"
								+ stnObject.m_stnLength + "\t");
						for (int k = 0; k < number_of_topics; k++) {
							parentParaOut.print(stnObject.m_topics[k] + "\t");
						}

					}
					parentParaOut.println();

				} else if (d instanceof _ChildDoc2) {
					childParaOut.print(d.getName() + "\t");
					childParaOut.print("topicProportion\t");
					for (int k = 0; k < number_of_topics; k++) {
						childParaOut.print(d.m_topics[k] + "\t");
					}

					childParaOut.print("general\t");
					for (int k = 0; k < number_of_topics; k++) {
						childParaOut.print(((_ChildDoc2) d).m_xTopics[0][k]
								+ "\t");
					}

					childParaOut.print("specific\t");
					for (int k = 0; k < number_of_topics; k++) {
						childParaOut.print(((_ChildDoc2) d).m_xTopics[1][k]
								+ "\t");
					}

					childParaOut.print("xProportion\t");
					for (int x = 0; x < m_gamma.length; x++) {
						childParaOut.print(((_ChildDoc2) d).m_xProportion[x]
								+ "\t");
					}

					childParaOut.println();
				}
			}

			parentParaOut.flush();
			parentParaOut.close();
			childParaOut.flush();
			childParaOut.close();

		} catch (Exception ex) {
			System.err.print("File Not Found");
		}

	}

	// p(w, z)=p(w|z)p(z)~~multinomial-dirichlet
	public void calLogLikelihood(int iter) {
		double logLikelihood = 0.0;
		double parentLogLikelihood = 0.0;
		double childLogLikelihood = 0.0;

		for (_Doc d : m_trainSet) {
			if (d instanceof _ParentDoc2) {
				parentLogLikelihood += calParentLogLikelihood((_ParentDoc2) d);
			} else {
				if (d instanceof _ChildDoc2) {
					childLogLikelihood += calChildLogLikelihood((_ChildDoc2) d);
				}
			}

		}

		double term1 = 0.0;
		double term2 = 0.0;
		double term3 = 0.0;
		double term4 = 0.0;
		for (int k = 0; k < number_of_topics; k++) {
			for (int n = 0; n < vocabulary_size; n++) {
				term3 += Utils.lgamma(d_beta + m_parentWordTopicSstat[k][n]
						+ m_childWordTopicSstat[k][n]);
			}
			term4 -= Utils.lgamma(vocabulary_size * d_beta + m_parentSstat[k]
					+ m_childSstat[k]);
		}

		term1 = number_of_topics * Utils.lgamma(vocabulary_size * d_beta);
		term2 = -number_of_topics * (vocabulary_size * Utils.lgamma(d_beta));

		parentLogLikelihood += term1 + term2 + term3 + term4;

		term1 = 0.0;
		term2 = 0.0;
		term3 = 0.0;
		term4 = 0.0;
		for (int k = 0; k < number_of_topics; k++) {
			for (int n = 0; n < vocabulary_size; n++) {
				term3 += Utils.lgamma(d_beta + m_parentWordTopicSstat[k][n]
						+ m_childWordTopicSstat[k][n]);
			}
			term4 -= Utils.lgamma(vocabulary_size * d_beta + m_parentSstat[k]
					+ m_childSstat[k]);
		}

		term1 = number_of_topics * Utils.lgamma(vocabulary_size * d_beta);
		term2 = -number_of_topics * (vocabulary_size * Utils.lgamma(d_beta));

		childLogLikelihood += term1 + term2 + term3 + term4;

		System.out.format("iter %d, parent log likelihood %.3f\n", iter,
				parentLogLikelihood);
		infoWriter.format("iter %d, parent log likelihood %.3f\n", iter,
				parentLogLikelihood);
		System.out.format("iter %d, child log likelihood %.3f\n", iter,
				childLogLikelihood);
		infoWriter.format("iter %d, child log likelihood %.3f\n", iter,
				childLogLikelihood);
		logLikelihood = parentLogLikelihood + childLogLikelihood;

		System.out
				.format("iter %d, log likelihood %.3f\n", iter, logLikelihood);
		infoWriter
				.format("iter %d, log likelihood %.3f\n", iter, logLikelihood);
	}

	// log space
	public double calParentLogLikelihood(_ParentDoc2 pDoc) {
		double term1 = 0.0;
		double term2 = 0.0;

		term1 = Utils.lgamma(number_of_topics * d_alpha) - number_of_topics
				* Utils.lgamma(d_alpha);

		for (int k = 0; k < number_of_topics; k++) {
			term2 += Utils.lgamma(pDoc.m_sstat[k] + d_alpha);
		}
		term2 -= Utils.lgamma((double) (number_of_topics * d_alpha + pDoc
				.getDocLength()));

		return term1 + term2;
	}

	// sum_x p(z|x)p(x)
	public double calChildLogLikelihood(_ChildDoc2 cDoc) {
		double tempLogLikelihood = 0.0;
		double tempLogLikelihood1 = 0.0;
		double tempLogLikelihood2 = 0.0;
		double term11 = 0.0;
		double term12 = 0.0;
		double term13 = 0.0;
		double term14 = 0.0;
		double weight1 = 0.0;
		double weight2 = 0.0;

		double term23 = 0.0;

		double parentDocLength = cDoc.m_parentDoc2.getTotalDocLength();

		term11 = Utils.lgamma(number_of_topics * d_alpha + m_mu);
		for (int k = 0; k < number_of_topics; k++) {
			term12 -= Utils.lgamma(d_alpha + m_mu
					* cDoc.m_parentDoc2.m_sstat[k] / parentDocLength);
			term13 += Utils.lgamma(d_alpha + m_mu
					* cDoc.m_parentDoc2.m_sstat[k] / parentDocLength
					+ cDoc.m_xTopicSstat[0][k]);

			term23 += Utils.lgamma(d_alpha + cDoc.m_xTopicSstat[1][k]);
		}

		term14 = -(Utils.lgamma(number_of_topics * d_alpha + m_mu
				+ cDoc.m_xSstat[0]));

		tempLogLikelihood1 = term11 + term12 + term13 + term14;

		tempLogLikelihood2 = Utils.lgamma(number_of_topics * d_alpha)
				- number_of_topics * Utils.lgamma(d_alpha) + term23
				- Utils.lgamma(number_of_topics * d_alpha + cDoc.m_xSstat[1]);

		weight1 = Utils.lgamma(m_gamma[0] + m_gamma[1])
				- Utils.lgamma(m_gamma[0]) - Utils.lgamma(m_gamma[1])
				+ Utils.lgamma(m_gamma[0] + cDoc.m_xSstat[0])
				+ Utils.lgamma(m_gamma[1])
				- Utils.lgamma(m_gamma[0] + m_gamma[1] + cDoc.m_xSstat[0]);

		weight2 = Utils.lgamma(m_gamma[0] + m_gamma[1])
				- Utils.lgamma(m_gamma[0]) - Utils.lgamma(m_gamma[1])
				+ Utils.lgamma(m_gamma[0])
				+ Utils.lgamma(m_gamma[1] + cDoc.m_xSstat[1])
				- Utils.lgamma(m_gamma[0] + m_gamma[1] + cDoc.m_xSstat[1]);

		// tempLogLikelihood = tempLogLikelihood1 * cDoc.m_xProportion[0]
		// + tempLogLikelihood2 * cDoc.m_xProportion[1];

		tempLogLikelihood = tempLogLikelihood1 + weight1 + tempLogLikelihood2
				+ weight2;

		return tempLogLikelihood;
	}

	// p(w, z)=p(w|z)p(z|d)
	public double calLogLikelihood2(int iter) {
		double logLikelihood = 0.0;
		double parentLogLikelihood = 0.0;
		double childLogLikelihood = 0.0;

		for (_Doc doc : m_trainSet) {
			if (doc instanceof _ParentDoc2)
				parentLogLikelihood += calParentLogLikelihood2((_ParentDoc2) doc);
			else if (doc instanceof _ChildDoc2)
				childLogLikelihood += calChildLogLikelihood2((_ChildDoc2) doc);
		}

		System.out.format("iter %d, parent log likelihood %.3f\n", iter,
				parentLogLikelihood);
		infoWriter.format("iter %d, parent log likelihood %.3f\n", iter,
				parentLogLikelihood);
		System.out.format("iter %d, child log likelihood %.3f\n", iter,
				childLogLikelihood);
		infoWriter.format("iter %d, child log likelihood %.3f\n", iter,
				childLogLikelihood);

		logLikelihood = parentLogLikelihood + childLogLikelihood;

		System.out
				.format("iter %d, log likelihood %.3f\n", iter, logLikelihood);
		infoWriter
				.format("iter %d, log likelihood %.3f\n", iter, logLikelihood);
		return logLikelihood;
	}

	public double calParentLogLikelihood2(_ParentDoc2 pDoc) {
		double likelihood = 0.0;

		int tid = 0;
		int wid = 0;
		double term1 = 0.0;
		double term2 = 0.0;
		for (int n = 0; n < pDoc.getTotalDocLength(); n++) {
			for (int k = 0; k < number_of_topics; k++) {
				wid = pDoc.m_words[n];
				// tid = pDoc.m_topicAssignment[n];
				// normalize
				term1 = (m_parentWordTopicSstat[k][wid] + m_childWordTopicSstat[k][wid])
						/ (double) (m_parentSstat[k] + m_childSstat[k]);
				term2 = pDoc.m_sstat[k] / (double) pDoc.getTotalDocLength();

				// if (term2 == 0)
				// System.out.println("term2 is zero");
				// if (term1 == 0)
				// System.out.println("term1 is zero");

				likelihood += Math.log(term1) + Math.log(term2);
			}
		}

		return likelihood;
	}

	public double calChildLogLikelihood2(_ChildDoc2 cDoc) {
		double likelihood = 0.0;

		int tid = 0;
		int wid = 0;
		double term1 = 0.0;
		double term2 = 0.0;

		for (int n = 0; n < cDoc.getTotalDocLength(); n++) {
			for (int k = 0; k < number_of_topics; k++) {
				wid = cDoc.m_words[n];
				// tid = cDoc.m_topicAssignment[n];

				term1 = (m_parentWordTopicSstat[k][wid] + m_childWordTopicSstat[k][wid])
						/ (double) (m_parentSstat[k] + m_childSstat[k]);
				term2 = (cDoc.m_xTopicSstat[0][k] + cDoc.m_xTopicSstat[1][k])
						/ (double) cDoc.getTotalDocLength();

				// if (term2 == 0)
				// System.out.println("term2 is zero");
				// if (term1 == 0)
				// System.out.println("term1 is zero");
				likelihood += Math.log(term1) + Math.log(term2);
			}
		}
		return likelihood;
	}
}
