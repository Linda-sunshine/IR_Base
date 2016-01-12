/****train whole corpus using LDA without distinguishing child documents or parent documents
***/

package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;

import structures.MyPriorityQueue;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._RankItem;
import utils.Utils;

public class LDA_GibbsDebug extends LDA_Gibbs {
	public LDA_GibbsDebug(int number_of_iteration,
			double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha,
			double burnIn, int lag) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag);

	}

	public void printTopWords(int k, String betaFile) {

		// create folders to record topic assigment txts
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

		
		File topicFolder = new File(filePrefix + "topicAssignment");
		if (!topicFolder.exists()) {
			System.out.println("creating directory" + topicFolder);
			topicFolder.mkdir();
		}
		
		for (_Doc d : m_trainSet) {
		
			printTopicAssignment( d, topicFolder);	
		}

		String parameterFile = filePrefix + "parameter.txt";
		printParameter(parameterFile);

	}

	public void printTopicAssignment(_Doc d, File folder) {
		String topicAssignmentfile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(folder,
					topicAssignmentfile));

			for (int i = 0; i < d.m_words.length; i++) {
				int index = d.m_words[i];

				String featureName = m_corpus.getFeature(index);

				int topic = d.m_topicAssignment[i];

				pw.print(featureName + ":" + topic + "\t");
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void printCorpusTopWords(int k, String betaFile) {

		try {
			System.out.println("parent beta file");
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

	public void printParameter(String parameterFile) {

		try {
			System.out.println("print childParameterFile");
			PrintWriter paraOut = new PrintWriter(new File(parameterFile));

			for (_Doc d : m_trainSet) {
				
				paraOut.print(d.getName() + "\t");
				paraOut.print("topicProportion\t");
				for (int k = 0; k < number_of_topics; k++) {
					paraOut.print(d.m_topics[k] + "\t");
				}
				paraOut.println();
			}
			paraOut.flush();
			paraOut.close();

		} catch (Exception ex) {
			System.err.print("para File Not Found");
		}

	}

}
