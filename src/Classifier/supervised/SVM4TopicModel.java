package Classifier.supervised;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import java.util.Collections;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;
import structures._Corpus;
import structures._Doc;
import structures._PerformanceStat;
import sun.security.util.Length;
import utils.Utils;

public class SVM4TopicModel extends SVM{
	public SVM4TopicModel(_Corpus c, double C, int classNo, int featureSize){
		super(classNo, featureSize, C);
		m_corpus = c;
		
	}
	
	public String toString(){
		return String.format("SVM4TopicModel[C:%d, F:%d, T:%s, c:%.3f]", m_classNo, m_featureSize, m_type, m_C);
	}
	
	public double train(Collection<_Doc> trainSet) {
		m_libModel = libSVMTrain(trainSet, 1+m_featureSize, m_type, m_C, 1);
		return 0;
	}
	
	public static Model libSVMTrain(Collection<_Doc> trainSet, int fSize, SolverType type, double C, double bias) {
		Feature[][] fvs = new Feature[trainSet.size()][];
		double[] y = new double[trainSet.size()];
		
		int fid = 0; // file id
//		ArrayList<Integer> labelList = new ArrayList<Integer>();
		for(_Doc d:trainSet) {
			if (bias>0)
				fvs[fid] = createLibLinearFV4TopicModel(d, fSize);
			else
				fvs[fid] = createLibLinearFV4TopicModel(d, 0);
			
//			if (!labelList.contains(d.getYLabel()))
//				labelList.add(d.getYLabel());
			y[fid] = d.getYLabel();
			fid ++;
		}
//		System.out.println("total documents\t"+fid);
		
		Problem libProblem = new Problem();
		libProblem.l = fid;
		libProblem.n = bias>=0?1+fSize:fSize;
		libProblem.x = fvs;
		libProblem.y = y;
		libProblem.bias = bias;
		
		return Linear.train(libProblem, new Parameter(type, C, SVM.EPS));
	}
	
	public int predict(_Doc doc) {
		return (int)Linear.predict(m_libModel, createLibLinearFV4TopicModel(doc, m_featureSize));
	}
	
	protected static Feature[] createLibLinearFV4TopicModel(_Doc doc, int fSize){
		Feature[] node;
		int featureNodeSize = doc.m_topics.length;
		if(fSize>0)
			node = new Feature[1+featureNodeSize];
		else
			node = new Feature[featureNodeSize];
		
		for(int i=0; i<featureNodeSize; i++){
//			if(doc.m_sstat[i]==0)
//				continue;
//			node[i] = new FeatureNode(i+1, doc.m_sstat[i]); //doc.m_topics
//			if(doc.m_topics[i]==0)
//				continue;
			node[i] = new FeatureNode(i+1, doc.m_topics[i]); //doc.m_topics

//			System.out.println("m_topics["+i+"]:\t"+doc.m_topics[i]);
		}
		if(fSize>0)
			node[featureNodeSize] = new FeatureNode(1+featureNodeSize, 1.0);
			
		return node;
	}
	
	
	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c){
		try {
			if (m_debugOutput!=null){
				m_debugWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(m_debugOutput, false), "UTF-8"));
				m_debugWriter.write(this.toString() + "\n");
			}
//				c.shuffle(k);

//				int[] masks = c.getMasks();
			ArrayList<_Doc> docs = c.getCollection();
			double[] accuracyArray = new double[k];
			//Use this loop to iterate all the ten folders, set the train set and test set.
			for (int i = 0; i < k; i++) {
//					for (int j = 0; j < masks.length; j++) {


					//more for training
//						if(masks[j]==i)
//							m_testSet.add(docs.get(j));
//						else
//							m_trainSet.add(docs.get(j));
//						if( masks[j]==(i+1)%k || masks[j]==(i+2)%k ) // || masks[j]==(i+3)%k
//							m_trainSet.add(docs.get(j));
//						else
//							m_testSet.add(docs.get(j));

//						//more for testing
//						if(masks[j]==i)
//							m_trainSet.add(docs.get(j));
//						else
//							m_testSet.add(docs.get(j));
//					}
				double trainingProportion = 0.01;
				splitData(c, trainingProportion);
				System.out.println("train set\t"+m_trainSet.size()+"\t test set\t"+m_testSet.size());
				long start = System.currentTimeMillis();
				train();
				double accuracy = test();
				accuracyArray[i] = accuracy;
				System.out.format("%s Train/Test finished in %.2f seconds with accuracy %.4f and F1 (%s)...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0, accuracy, getF1String());
				m_trainSet.clear();
				m_testSet.clear();
			}
			calculateMeanVariance(m_precisionsRecalls);

			double accuracyVar = Utils.getVariance(accuracyArray);
			double accuracyAvg = Utils.getMean(accuracyArray);
			System.out.println("accuracy avg:\t"+accuracyAvg+"\t var:\t"+accuracyVar);

			if (m_debugOutput!=null)
				m_debugWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	protected void splitData(_Corpus c, double trainingProportion){
		HashMap<Integer, ArrayList<_Doc>> labelDocMap = new HashMap<Integer, ArrayList<_Doc>>();
		for(_Doc d:c.getCollection()){
			int yLabel = d.getYLabel();
			if(labelDocMap.containsKey(yLabel)){
				ArrayList<_Doc>labelDocList = labelDocMap.get(yLabel);
				labelDocList.add(d);
			}else{
				ArrayList<_Doc> labelDocList = new ArrayList<_Doc>();
				labelDocList.add(d);
				labelDocMap.put(yLabel, labelDocList);
			}
		}
		for(int yLabel:labelDocMap.keySet()){
			ArrayList<_Doc> labelDocList = labelDocMap.get(yLabel);
			Collections.shuffle(labelDocList);
		}


		for(int yLabel:labelDocMap.keySet()){
			ArrayList<_Doc>labelDocList = labelDocMap.get(yLabel);

			int trainingNum = (int)(labelDocList.size()*trainingProportion);

			for(int i=0; i<trainingNum; i++)
				m_trainSet.add(labelDocList.get(i));

			for(int i=trainingNum; i<labelDocList.size(); i++)
				m_testSet.add(labelDocList.get(i));
		}


	}

}
