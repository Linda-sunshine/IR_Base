package topicmodels.DCM;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._Doc4SparseDCMLDA;
import structures._RankItem;
import structures._Word;
import utils.Utils;

public class sparseClusterDCMLDA_test extends sparseClusterDCMLDA{
	public sparseClusterDCMLDA_test(int number_of_iteration, double converge,
			double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha,
			double burnIn, int lag, int newtonIter, double newtonConverge, double tParam, double sParam, int clusterNum, double gammaParam) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag, newtonIter, newtonConverge, tParam, sParam, clusterNum, gammaParam);

	}

	public void printTopWords(int k, String betaFile) {
		double logLikelihood = calculate_log_likelihood();
		System.out.format("final log likelihood %.3f\t", logLikelihood);

		String filePrefix = betaFile.replace("topWords.txt", "");
		debugOutput(k, filePrefix);

	}

	protected void debugOutput(int topK, String filePrefix) {

		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		String betaFile = filePrefix + "/topBeta.txt";
		printTopBeta(topK, betaFile);

		String topWordFile = filePrefix + "/topWords.txt";
		printTopWord(topK, topWordFile);

		File topicFolder = new File(filePrefix + "topicAssignment");

		if (!topicFolder.exists()) {
			System.out.println("creating directory" + topicFolder);
			topicFolder.mkdir();
		}

		File wordTopicDistributionFolder = new File(filePrefix
				+ "wordTopicDistribution");
		if (!wordTopicDistributionFolder.exists()) {
			System.out.println("creating word topic distribution folder\t"
					+ wordTopicDistributionFolder);
			wordTopicDistributionFolder.mkdir();
		}

		for (_Doc d : m_trainSet) {
			printParentTopicAssignment(d, topicFolder);
			printSparsity(d, topicFolder);
		}
		
		for(int c=0; c<m_clusterNum; c++){
			printWordTopicDistribution(c, wordTopicDistributionFolder, topK);
		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";

		printParameter(parentParameterFile, childParameterFile, m_trainSet);
		printSparsityRatio(filePrefix);
	}
	
	protected void printSparsityRatio(String filePrefix){
		String sparsityRatioFile = filePrefix+"sparsityRatio.txt";
		try {
			PrintWriter pw = new PrintWriter(new File(sparsityRatioFile));
			
			for(_Doc d:m_trainSet){
				_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;
				pw.print(DCMDoc.m_topicIndicator_distribution + "\n");
			}

			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	protected void printTopBeta(int k, String topBetaFile) {
		System.out.println("TopWord FilePath:" + topBetaFile);

		Arrays.fill(m_sstat, 0);
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < number_of_topics; i++)
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];
		}
		Utils.L1Normalization(m_sstat);

		try {
			PrintWriter topWordWriter = new PrintWriter(new File(topBetaFile));

			for (int i = 0; i < m_beta.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							m_beta[i][j]));

				topWordWriter.format("Topic %d(%.5f):\t", i, m_sstat[i]);
				for (_RankItem it : fVector)
					topWordWriter.format("%s(%.5f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				topWordWriter.write("\n");
			}
			topWordWriter.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}

	protected void printTopWord(int k, String topWordFile) {
		System.out.println("TopWord FilePath:" + topWordFile);

		Arrays.fill(m_sstat, 0);
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < number_of_topics; i++)
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];
		}
		Utils.L1Normalization(m_sstat);

		try {
			PrintWriter topWordWriter = new PrintWriter(new File(topWordFile));

			for (int i = 0; i < topic_term_probabilty.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							topic_term_probabilty[i][j]));

				topWordWriter.format("Topic %d(%.5f):\t", i, m_sstat[i]);
				for (_RankItem it : fVector)
					topWordWriter.format("%s(%.5f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				topWordWriter.write("\n");
			}
			topWordWriter.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}

	protected void printParentTopicAssignment(_Doc d, File topicFolder) {
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;
		String topicAssignmentFile = DCMDoc.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(topicFolder,
					topicAssignmentFile));
			pw.println("cluster\t" + DCMDoc.m_clusterIndicator);
			for (_Word w : DCMDoc.getWords()) {
				int index = w.getIndex();
				int topic = w.getTopic();

				String featureName = m_corpus.getFeature(index);
				pw.print(featureName + ":" + topic + "\t");
			}

			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	
	protected void printSparsity(_Doc d, File topicFolder) {
		String topicAssignmentFile = d.getName() + "sparsity.txt";
		try {
			PrintWriter pw = new PrintWriter(new File(topicFolder,
					topicAssignmentFile));

			_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;
			pw.print(DCMDoc.m_indicatorTrue_stat + "\t");
			for (int k = 0; k < number_of_topics; k++) {
				pw.print(DCMDoc.m_topicIndicator[k] + ":"
						+ DCMDoc.m_topicIndicator_prob[k] + "\t");
			}

			pw.print(DCMDoc.m_topicIndicator_distribution);

			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	protected void printParameter(String parentParameterFile,
			String childParameterFile, ArrayList<_Doc> docList) {
		System.out.println("printing parameter");

		try {
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);

			PrintWriter parentParaOut = new PrintWriter(new File(
					parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(
					childParameterFile));

			for (_Doc d : docList) {
				parentParaOut.print(d.getName() + "\t");
				parentParaOut.print("topicProportion\t");
				for (int k = 0; k < number_of_topics; k++) {
					parentParaOut.print(d.m_topics[k] + "\t");
				}

				parentParaOut.println();
			}

			parentParaOut.flush();
			parentParaOut.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	protected void printWordTopicDistribution(int cID,
			File wordTopicDistributionFolder, int k) {
		String wordTopicDistributionFile = cID  + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(
					wordTopicDistributionFolder, wordTopicDistributionFile));

			for (int i = 0; i < number_of_topics; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int v = 0; v < vocabulary_size; v++) {
					String featureName = m_corpus.getFeature(v);
					double wordProb = m_clusterTopicWordProb[cID][i][v];
					_RankItem ri = new _RankItem(featureName, wordProb);
					fVector.add(ri);
				}

				pw.format("Topic %d(%.5f):\t", i, m_clusterTopicProb[cID][i]);
				for (_RankItem it : fVector)
					pw.format("%s(%.5f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				pw.write("\n");
			}

			pw.flush();
			pw.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
