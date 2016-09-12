package Classifier.supervised;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Random;

import Classifier.BaseClassifier;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;

public class KNN extends BaseClassifier{
	int m_k;
	int m_l;
	double[][] m_randomVcts;
	HashMap<Integer, ArrayList<_Doc>> m_buckets;
	
	public KNN(_Corpus c){
		super(c);
		m_k = 5;
		m_l = 10;
		m_buckets = new HashMap<Integer, ArrayList<_Doc>>();
	}
	
	public KNN(_Corpus c, int k, int l){
		super(c);
		m_k = k;
		m_l = l;
		m_randomVcts = new double[m_l][m_featureSize];
		m_buckets = new HashMap<Integer, ArrayList<_Doc>>();
	}
	
	@Override
	public String toString() {
		return String.format("kNN [k:%d, l:%d]", m_k, m_l);
	}

	//Initialize the random vectors.
	protected void init() {		
		m_buckets.clear();
		Random r = new Random();
		for(int i = 0; i < m_l; i++){
			for(int j = 0; j < m_featureSize; j++){
				m_randomVcts[i][j] = 2*r.nextDouble()-1;
			}
		}
	}
	
	public void setKL(int k, int l){
		m_k = k;
		m_l = l;
		m_randomVcts = new double[m_l][m_featureSize];
	}	
	
	//Group all the documents based on their hashcodes.
	@Override
	public double train(Collection<_Doc> trainSet) {
		init();
		
		if (m_l<=0)//no need to perform random projection
			return 0;
		
		for(_Doc d: trainSet){
			int hashCode = getHashCode(d);
			if(m_buckets.containsKey(hashCode)){
				m_buckets.get(hashCode).add(d);
			} else{
				ArrayList<_Doc> docs = new ArrayList<_Doc>();
				docs.add(d);
				m_buckets.put(hashCode, docs);
			}
		}
		return 0;
	}

	//Get the hashcode for every document.
	public int getHashCode(_Doc d){
		int[] hashArray = new int[m_l];
		for(int i = 0; i < m_l; i++)
			hashArray[i] = Utils.sgn(Utils.dotProduct(m_randomVcts[i], d.getSparse()));
		return Utils.encode(hashArray);
	}
	
	@Override
	public int predict(_Doc doc) {
		Collection<_Doc> docs;
		if (m_l<=0) {//no random projection
			docs = m_trainSet;
		} else {
			docs = m_buckets.get(getHashCode(doc));
			if(docs.size() < m_k) {
				System.err.println("L is set too large, tune the parameter.");
				return -1;
			} 
		}
		
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(m_k);
		for(_Doc d:docs)
			neighbors.add(new _RankItem(d.getYLabel(), Utils.dotProduct(d, doc)));
		
		Arrays.fill(m_cProbs, 0);
		for(_RankItem rt:neighbors)
			m_cProbs[rt.m_index] ++;//why don't we consider the similarity?
		
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	
	@Override
	public double score(_Doc doc, int label) {
		Collection<_Doc> docs;
		if (m_l<=0) {//no random projection
			docs = m_trainSet;
		} else {
			docs = m_buckets.get(getHashCode(doc));
			if(docs.size() < m_k) {
				System.err.println("L is set too large, tune the parameter.");
				return -1;
			} 
		}
		
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(m_k);
		for(_Doc d:docs)
			neighbors.add(new _RankItem(d.getYLabel(), Utils.dotProduct(d, doc)));
		
		Arrays.fill(m_cProbs, 0);
		for(_RankItem rt:neighbors)
			m_cProbs[rt.m_index] ++;
		
		return m_cProbs[label] - m_k;//to be consistent with the predict function
	}

	@Override
	protected void debug(_Doc d) {}

	@Override
	public void saveModel(String modelLocation) {}
}
