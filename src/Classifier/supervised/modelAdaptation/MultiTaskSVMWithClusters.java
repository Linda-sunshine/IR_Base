package Classifier.supervised.modelAdaptation;

import java.util.ArrayList;
import java.util.HashMap;

import structures._Review;
import structures._SparseFeature;
import structures._Review.rType;
import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;

public class MultiTaskSVMWithClusters extends MultiTaskSVM {
	
	int m_clusterNo; // The number of clusters.
	HashMap<Integer, Integer> m_userIndexClusterIndexMap; // Find each user's corresponding cluster no.
	public MultiTaskSVMWithClusters(int classNo, int featureSize, int clusterNo) {
		super(classNo, featureSize);
		m_clusterNo = clusterNo;
	}
	public void setUserIndexClusterIndexMap(HashMap<Integer, Integer> map){
		m_userIndexClusterIndexMap = map;
	}
	
	@Override
	public String toString() {
		return String.format("MT-SVM-Clusters[mu:%.3f,C:%.3f,clusters: %d,bias:%b]", m_u, m_C, m_bias);
	}
	
	//create a training instance of svm with cluster information.
	//for MT-SVM feature vector construction: we put user models in front of global model
	public Feature[] createLibLinearFV(_Review r, int userIndex, int clusterIndex){
		int fIndex; double fValue;
		_SparseFeature fv;
		_SparseFeature[] fvs = r.getSparse();
		
		int userOffset, clusterOffest, globalOffset;		
		Feature[] node;//0-th: x//sqrt(u); t-th: x.
		
		if (m_bias) {
			userOffset = (m_featureSize + 1) * userIndex;
			clusterOffest = (m_featureSize + 1) * (m_userSize + clusterIndex);
			globalOffset = (m_featureSize + 1) * (m_userSize + m_clusterNo);
			node = new Feature[(1+fvs.length) * 3]; // It consists of three parts.
		} else {
			userOffset = m_featureSize * userIndex;
			globalOffset = m_featureSize * m_userSize;
			node = new Feature[fvs.length * 2];
		}
		
		for(int i = 0; i < fvs.length; i++){
			fv = fvs[i];
			fIndex = fv.getIndex() + 1;//liblinear's feature index starts from one
			fValue = fv.getValue();
			
			//Construct the user part of the training instance.			
			node[i] = new FeatureNode(userOffset + fIndex, fValue);
			
			// Construct the cluster part of the training instance.
			node[]
			
			//Construct the global part of the training instance.
			if (m_bias)
				node[i + fvs.length + 1] = new FeatureNode(globalOffset + fIndex, fValue/m_u); // global model's bias term has to be moved to the last
			else
				node[i + fvs.length] = new FeatureNode(globalOffset + fIndex, fValue/m_u); // global model's bias term has to be moved to the last
		}
		
		if (m_bias) {//add the bias term		
			node[fvs.length] = new FeatureNode((m_featureSize + 1) * (userIndex + 1), 1.0);//user model's bias
			node[2*fvs.length+1] = new FeatureNode((m_featureSize + 1) * (m_userSize + 1), 1.0 / m_u);//global model's bias
		}
		return node;
	}
	

}
