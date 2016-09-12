package Analyzer;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import clustering.KMeansAlg4Vct;
import Classifier.BaseClassifier;
import Classifier.supervised.SVM;
import structures._Corpus;
import structures._Doc;

/***
 * The class performs feature selection-cross.
 * @author lin
 *
 */
public class CrossFeatureSelection {
	int m_kFold;
	int m_kMeans;
	int m_classNo;
	int m_featureSize;
	double m_C = 1.0; //penalty of SVM.
	double[][] m_weights;
	_Corpus m_corpus;
	ArrayList<ArrayList<_Doc>> m_trainSets;
	BaseClassifier m_classifier; 

	public CrossFeatureSelection(_Corpus c, int classNo, int featureSize, int kFold, int kMeans){
		m_corpus = c;
		m_kFold = kFold;
		m_kMeans = kMeans;
		m_classNo = classNo;
		m_featureSize = featureSize;
		m_trainSets = new ArrayList<ArrayList<_Doc>>();
	}
	
	public void init(){
		for(int i=0; i<m_kFold;i++)
			m_trainSets.add(new ArrayList<_Doc>());
	}
	
	//Split the whole collection into k folds.
	public void splitCorpus(){
		init();
		int fold = 0;
		m_corpus.shuffle(m_kFold);
		int[] masks = m_corpus.getMasks();
		ArrayList<_Doc> docs = m_corpus.getCollection();
		//Use this loop to iterate all the ten folders, set the train set and test set.
		for (int j = 0; j < masks.length; j++) {
			fold = masks[j];
			m_trainSets.get(fold).add(docs.get(j));
		}
	}
	
	// Train classifiers based on the splited training documents.
	public void train(){
		splitCorpus();
		m_weights = new double[m_kFold][];		
		for(int i=0; i < m_trainSets.size(); i++){
			m_classifier = new SVM(m_classNo, m_featureSize, m_C);
			m_classifier.train(m_trainSets.get(i));
			m_weights[i] = ((SVM) m_classifier).getWeights();
		}
		System.out.println(String.format("[Info]Finish training %d folds data!", m_kFold));
	}
	String m_filename;
	// Perform k-means on the features based on learned weights.
	public void kMeans(){
		KMeansAlg4Vct kmeans = new KMeansAlg4Vct(m_weights, m_kMeans);
		kmeans.init();
		kmeans.train();
		m_filename = String.format("CrossFeatures_%dfold_%dmeans_%dfvGroups.txt", m_kFold, m_kMeans, kmeans.getClusterSize());
		writeResults(kmeans.getClusters(), kmeans.getClusterSize());

	}

	public void writeResults(int[] clusterNos, int size){
		try{
			PrintWriter writer = new PrintWriter(new File(m_filename));
			for(int i=0; i<clusterNos.length-1; i++)
				writer.write(clusterNos[i] + ",");
			writer.write(clusterNos[clusterNos.length-1]+"\n");
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	public String getFilename(){
		return m_filename;
	}
}
